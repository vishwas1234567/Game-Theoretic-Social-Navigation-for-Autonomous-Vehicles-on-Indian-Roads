"""Stylised unprotected-merge/crossing simulator, parameterised from the night-clip statistics.

Ego (AV) approaches a conflict point and must pass a stream of K human-like opponents.
Opponents: noisy-rational, latent courtesy type psi, react to the ego's *claim* (inching / committing).
Planners: RIGID (fixed gap threshold), ASSERTIVE (small gap threshold), SOCIAL (Bayesian type inference +
          Stackelberg-style plan evaluation {wait, creep, go} with politeness cost) and ablations.
"""
import json, math, sys, numpy as np, pandas as pd
from multiprocessing import Pool

VS = json.load(open('data/video_stats.json'))
PD = pd.read_csv('data/perception_derived.csv')
DRV = PD[PD.t >= VS['t_first_move']]
SPEEDS = np.clip(np.array(VS['speed_moving']), 3.0, 11.0)          # video-derived opponent free speeds
HEADWAYS = np.clip(np.array(VS['lead_headways']), 0.8, 10.0)        # video-derived inter-vehicle headways
cm = {k: DRV.crowd[DRV.density == k].mean() for k in ['sparse', 'medium', 'dense']}
HW_SCALE = {k: cm['medium'] / cm[k] for k in cm}                     # denser scene -> shorter gaps
DT = 0.1
TYPES = np.array([0.0, 0.5, 1.0]); PRIOR = np.array([0.3, 0.4, 0.3])
BETA, THETA, KAPPA, DELTA = 6.0, 0.7, 0.8, 0.6
D_STOP, Z_IN, L_E, L_O, Z_OUT = -3.5, -1.0, 4.0, 3.0, 1.5
A_MAX, A_MIN, JMAX = 2.0, -3.0, 6.0


def sig(x): return 1.0 / (1.0 + math.exp(-x))


def claim_of(d, v):
    return max(0.0, min(1.0, 1 - (-d) / 6.0)) * min(1.0, (v + 0.6) / 1.5)


def p_yield_fn(psi, claim, urg):
    return sig(BETA * (psi + KAPPA * claim - DELTA * urg - THETA))


def t_enter(d, v, d_in=Z_IN):
    if d >= d_in: return 0.0
    return (d_in - d) / max(v, 0.3)


def ego_clear_time(d, v, v_go=6.0):
    """time for ego to clear the zone if it accelerates at A_MAX towards v_go."""
    dist = max(0.0, Z_OUT + L_E - d)
    if v >= v_go: return dist / v
    d_acc = (v_go ** 2 - v ** 2) / (2 * A_MAX)
    if dist <= d_acc:
        return (-v + math.sqrt(v * v + 2 * A_MAX * dist)) / A_MAX
    return (v_go - v) / A_MAX + (dist - d_acc) / v_go


def ego_enter_time_go(d, v, v_go=6.0):
    dist = max(0.0, Z_IN - d)
    if dist == 0: return 0.0
    return (-v + math.sqrt(v * v + 2 * A_MAX * dist)) / A_MAX


def a_required(d_o, v_o, T):
    """constant decel so the opponent arrives at the zone entry at time T (or stops before it)."""
    dist = max(0.1, Z_IN - d_o - 0.6)
    a = 2 * (dist - v_o * T) / (T * T)
    a_stop = v_o * v_o / (2 * dist)
    return -min(max(-a, 0.0), max(a_stop, 0.0) if v_o + a * T < 0 else 6.0, 6.0)


def pred_entry(o, horizon=6.0):
    """time until the opponent's front reaches the zone entry assuming it keeps its current acceleration."""
    d, v, t = o.d, o.v, 0.0
    a = min(o.a, 0.0) if o.yielding else 0.0
    if d >= Z_IN: return 0.0
    if v <= 0.05 and o.yielding:                       # stopped yielder: may lose patience and restart
        return max(0.0, o.pat - o.stopt) + 0.3 + math.sqrt(max(Z_IN - d, 0.1))
    while t < horizon:
        v = max(0.0, v + a * 0.1); d += v * 0.1; t += 0.1
        if d >= Z_IN: return t
        if v <= 0.05: return 99.0
    return 99.0


class Opp:
    def __init__(self, v0, T_arr, psi, rng):
        self.v0, self.v, self.psi = v0, v0, psi
        self.d = Z_IN - v0 * T_arr            # distance such that it reaches the zone at T_arr at v0
        self.u = rng.random(); self.yielding = False; self.a = 0.0; self.hist_a = []
        self.passed = False; self.t_in = None; self.t_out = None; self.dec_peak = 0.0; self.delay = 0.0
        self.induced = []
        self.pat = {0.0: 1.5, 0.5: 2.5, 1.0: 4.0}[float(psi)]; self.stopt = 0.0; self.gave_up = False


def run_episode(args):
    planner, density, seed, keep = args[:4]
    odd = args[4] if len(args) > 4 else None          # (q_true, policy) -> ODD-conditioned perception degradation
    rng = np.random.default_rng(seed)
    mult, vgo, tracks = 1.0, 6.0, {}
    if odd is not None:
        q, pol = odd
        q_hat = float(np.clip(q + rng.normal(0, 0.05), 0, 1))
        if (pol == 'strict' and q_hat < 0.85) or (pol in ('track', 'margin') and q_hat < 0.15):     # outside ODD -> minimal-risk stop / no trip
            return dict(planner=planner, density=density, seed=seed, collision=0, done=0, min_pet=9.0, rms_jerk=0.0, peak_decel=0.0,
                        delay=60.0, stall=0.0, jain=1 / 7, opp_mean_decel=0.0, opp_peak_decel=0.0, opp_hard=0, opp_delay=0.0,
                        available=0, q=q, policy=pol)
        if pol == 'margin': mult = 1 + 1.5 * (1 - q_hat)
        sig_d, sig_v, miss, lat = 0.3 + 3.0 * (1 - q), 0.1 + 1.0 * (1 - q), 0.35 * (1 - q) ** 2, 0.5 * (1 - q)

        def percept_fn(lv, t):
            out = []
            for o in lv:
                k = id(o); tr = tracks.get(k)
                if rng.random() > miss:
                    dd = o.d - o.v * lat + rng.normal(0, sig_d); vv = max(0.0, o.v + rng.normal(0, sig_v)); tracks[k] = (dd, vv, t); seen = (dd, vv)
                elif pol in ('track', 'margin') and tr is not None and t - tr[2] <= 1.5:
                    seen = (tr[0] + tr[1] * (t - tr[2]), tr[1])
                else:
                    continue
                pr = Opp.__new__(Opp); pr.__dict__.update(o.__dict__); pr.d, pr.v = seen; pr.orig = o; out.append(pr)
            return out
    K = 6
    hs = rng.choice(HEADWAYS, K) * HW_SCALE[density]
    T1 = rng.uniform(2.0, 6.0)
    arr = T1 + np.concatenate([[0], np.cumsum(hs[:-1] + 0.8)])
    psis = rng.choice(TYPES, K, p=PRIOR)
    opps = [Opp(float(rng.choice(SPEEDS)), arr[k], psis[k], rng) for k in range(K)]
    d_e, v_e, a_e = -20.0, float(np.clip(0.6 * rng.choice(SPEEDS), 2.0, 5.0)), 0.0
    belief = PRIOR.copy(); belief_k = 0
    plan = 'go'; committed = False
    t = 0.0; log = []; ego_in = None; ego_out = None
    jerks = []; accs = []; stall = 0.0
    hist = []                                    # delayed observation buffer of (d_e, v_e)
    next_plan_t = 0.0; creeping_since = None
    free_t = None
    while t < 60.0:
        hist.append((d_e, v_e))
        d_obs, v_obs = hist[max(0, len(hist) - 1 - 5)]   # 0.5 s opponent reaction delay
        # ---------------- ego planning -----------------
        live = [o for o in opps if not o.passed]
        nxt = live[0] if live else None
        if t >= next_plan_t and d_e < Z_IN + 0.5 and not committed:
            next_plan_t = t + 0.2
            if odd is not None:
                live = percept_fn(live, t); nxt = live[0] if live else None
            if nxt is None:
                plan = 'go'
            elif planner in ('rigid', 'assertive'):
                gap_req = 5.0 if planner == 'rigid' else 1.5
                T_clr = ego_clear_time(d_e, max(v_e, 0.5), vgo)
                ok = True
                for o in live:
                    To = t_enter(o.d, o.v, Z_IN)
                    To_out = (Z_OUT + L_O - o.d) / max(o.v, 0.3)
                    if To - T_clr < gap_req and To_out > ego_enter_time_go(d_e, max(v_e, 0.5)) - gap_req:
                        ok = False; break
                plan = 'go' if ok else 'wait'
            else:                                    # ----- SOCIAL game-theoretic planner -----
                if belief_k != id(getattr(nxt, 'orig', nxt)):
                    belief = PRIOR.copy(); belief_k = id(getattr(nxt, 'orig', nxt))
                lam_s = SOC['lam_s']; use_inf = SOC['infer']; use_creep = SOC['creep']
                b = belief if use_inf else PRIOR
                T_clr_go = ego_clear_time(d_e, max(v_e, 0.5), vgo)
                T_ent_go = ego_enter_time_go(d_e, max(v_e, 0.5))
                To = t_enter(nxt.d, nxt.v); To_out = (Z_OUT + L_O - nxt.d) / max(nxt.v, 0.3)
                later = [(t_enter(o.d, o.v), (Z_OUT + L_O - o.d) / max(o.v, 0.3)) for o in live[1:]]
                T_wait = To_out + 1.0
                ev = To_out
                for (a_in, a_out) in later:
                    if a_in - ev >= 2.5 + T_clr_go: break
                    ev = a_out; T_wait = a_out + 1.0
                c_wait = T_wait + 2.0
                urg = min(1.0, max(0.0, (3.0 - To) / 3.0))
                # ---- safety shield: 'go' only if every live opponent is predicted (const-accel, observed a) to
                #      be clear of / not yet in the zone with >= 0.7 s margin
                shield = True
                for o in live:
                    ent = pred_entry(o); ext = ent + (L_O + Z_OUT - Z_IN) / max(o.v, 1.0) if ent < 90 else 99.0
                    if ent < 90 and not (ent >= mult * (1.2 * T_clr_go + 0.7) or ext <= T_ent_go - 0.7 * mult): shield = False; break
                a_req = a_required(nxt.d, nxt.v, T_clr_go + 0.8) if To < T_clr_go + 0.8 and To > T_ent_go - 0.7 else 0.0
                c_go = (T_clr_go + lam_s * a_req ** 2) if shield else 1e9
                if use_creep and d_e < -1.5 and not shield:
                    c_cr = 0.0
                    for pt, bt in zip(TYPES, b):
                        py = p_yield_fn(pt, 0.62, urg)
                        c_cr += bt * (py * (T_clr_go + 0.8 + lam_s * a_req ** 2) + (1 - py) * (c_wait + 0.6))
                else:
                    c_cr = 1e9
                plan = min((c_go, 'go'), (c_wait, 'wait'), (c_cr, 'creep'))[1]
                if DEBUG is not None: DEBUG.append((round(t,1), round(d_e,1), round(v_e,1), plan, shield, round(c_go,1), round(c_wait,1), round(c_cr,1), [(round(o.d,1), round(o.v,1), round(pred_entry(o),1)) for o in live[:3]]))
                # Bayesian update from observed opponent response (yield vs hold) at claim-dependent likelihood
                if To < 6.0 and To_out > 0:
                    yielded_obs = nxt.a < -0.8
                    cl = claim_of(d_obs, v_obs)
                    like = np.array([p_yield_fn(pt, cl, urg) for pt in TYPES])
                    like = like if yielded_obs else 1 - like
                    belief = belief * like + 1e-6; belief /= belief.sum()
        # ---------------- ego low-level control ----------------
        if plan == 'go' or committed:
            v_t = vgo; a_cmd = float(np.clip((v_t - v_e) / 0.8, A_MIN, A_MAX))
        else:
            if plan == 'creep': d_t, vcap = -2.0, 1.2
            else: d_t, vcap = D_STOP, 99.0
            v_t = min(vcap, math.sqrt(max(0.0, 2 * 1.5 * (d_t - d_e))))   # 0 once at/after the hold point
            if plan == 'creep' and d_e < d_t - 0.4: v_t = max(v_t, 0.6)
            a_cmd = float(np.clip((v_t - v_e) / 0.8, A_MIN, A_MAX))
            if v_e > v_t + 0.05:      # braking: make sure we really stop before the zone (-1.3 m)
                a_cmd = min(a_cmd, max(A_MIN, -v_e * v_e / (2 * max(max(d_t, -1.3 if d_e > d_t else d_t) - d_e, 0.05))))
        a_new = a_e + float(np.clip(a_cmd - a_e, -JMAX * DT, JMAX * DT))
        jerks.append((a_new - a_e) / DT); accs.append(a_new)
        a_e = a_new
        v_e = max(0.0, v_e + a_e * DT); d_e += v_e * DT
        # point of no return: cannot stop before the zone with the comfortable-hard limit A_MIN
        if d_e + v_e ** 2 / (2 * abs(A_MIN)) > Z_IN - 0.2: committed = True
        # ---------------- opponents ----------------
        for o in opps:
            if o.passed: continue
            To = t_enter(o.d, o.v); To_out = (Z_OUT + L_O - o.d) / max(o.v, 0.3)
            ego_clr_obs = ego_clear_time(d_obs, max(v_obs, 0.5))
            ego_ent_obs = ego_enter_time_go(d_obs, max(v_obs, 0.5))
            ego_done = d_obs > Z_OUT + L_E
            conflict = (not ego_done) and (To > 0) and (To < ego_clr_obs + 1.5) and (To_out > ego_ent_obs - 1.0 or d_obs > Z_IN)
            if conflict:
                urg_o = min(1.0, max(0.0, (3.0 - To) / 3.0))
                py = p_yield_fn(o.psi, claim_of(d_obs, v_obs), urg_o)
                if o.u < py and not o.gave_up: o.yielding = True
            if ego_done: o.yielding = False
            if o.yielding and o.v < 0.3:
                o.stopt += DT
                if o.stopt > o.pat: o.yielding = False; o.gave_up = True      # patience runs out -> opponent takes the right of way
            a_o = (o.v0 - o.v) / 1.0
            if o.yielding and To > 0:
                T_goal = (1.25 * ego_clr_obs if d_obs < Z_OUT + L_E else 0.0) + 0.8
                a_o = min(a_o, a_required(o.d, o.v, max(T_goal, To)))
                dist_s = Z_IN - 0.6 - o.d
                if o.v < 2.0 and dist_s < 4.0:          # yielders come to a stop short of the zone, not creep into it
                    a_o = min(a_o, -min(6.0, max(0.3, o.v * o.v / (2 * max(dist_s, 0.1)))) if o.v > 0.02 else 0.0)
            # emergency backstop
            ego_in_zone = (d_e > Z_IN) and (d_e - L_E < Z_OUT)
            if To > 0 and To < 1.6 and ego_in_zone and not o.yielding:
                a_o = -6.0
            a_o = float(np.clip(a_o, -6.0, 2.0))
            o.a = a_o
            o.v = max(0.0, o.v + a_o * DT); o.d += o.v * DT
            o.induced.append(max(0.0, -a_o) if (a_o < -0.8) else 0.0)
            if o.t_in is None and o.d > Z_IN: o.t_in = t
            if o.t_out is None and o.d - L_O > Z_OUT: o.t_out = t; o.passed = True
            if o.v < 0.2 and o.d < Z_IN - 0.5:
                pass
        if ego_in is None and d_e > Z_IN: ego_in = t
        if ego_out is None and d_e - L_E > Z_OUT: ego_out = t
        # stall (hesitation): ego nearly stopped before the zone while the next opponent is > 3 s away
        if v_e < 0.3 and d_e < Z_IN:
            nn = [o for o in opps if not o.passed]
            if not nn or t_enter(nn[0].d, nn[0].v) > 3.0 or nn[0].v < 0.3: stall += DT
        if keep: log.append((t, d_e, *[o.d for o in opps]))
        t += DT
        if ego_out is not None and all(o.passed for o in opps): break
        if t > 59.9: break
    # ---------------- metrics ----------------
    done = ego_out is not None
    pets = []; collided = False
    for o in opps:
        if o.t_in is None or ego_in is None: continue
        if o.t_out is None: o.t_out = t
        e_out = ego_out if ego_out is not None else t
        if (ego_in < o.t_out) and (o.t_in < e_out): collided = True; pets.append(-0.1)
        else: pets.append(o.t_in - e_out if ego_in < o.t_in else ego_in - o.t_out)
    free_T = (6.5 + 15.0 + 0) / max(5.0, 1.0)
    delay = (ego_out if done else 60.0) - 6.0
    ind = [x for o in opps for x in o.induced]
    ind_arr = np.array(ind) if ind else np.zeros(1)
    opp_delay = []
    for o in opps:
        T_free = arr[opps.index(o)] + (Z_OUT + L_O) / o.v0
        opp_delay.append(max(0.0, (o.t_out if o.t_out is not None else t) - T_free))
    all_d = np.array([max(delay, 0.0)] + opp_delay) + 1e-3
    jain = float(all_d.sum() ** 2 / (len(all_d) * (all_d ** 2).sum()))
    res = dict(planner=planner, density=density, seed=seed, collision=int(collided), done=int(done),
               min_pet=float(min(pets)) if pets else 9.0, rms_jerk=float(np.sqrt(np.mean(np.square(jerks)))),
               peak_decel=float(-min(accs)), delay=float(max(delay, 0)), stall=float(stall),
               jain=jain, opp_mean_decel=float(ind_arr[ind_arr > 0].mean()) if (ind_arr > 0).any() else 0.0,
               opp_peak_decel=float(ind_arr.max()), opp_hard=int((ind_arr > 3.0).any()),
               opp_delay=float(np.mean(opp_delay)))
    if odd is not None: res.update(available=1, q=odd[0], policy=odd[1])
    if keep: res['log'] = log
    return res


SOC = dict(lam_s=2.0, infer=True, creep=True)
DEBUG = None
VARIANTS = {
    'rigid': dict(planner='rigid'), 'assertive': dict(planner='assertive'),
    'social': dict(planner='social', soc=dict(lam_s=2.0, infer=True, creep=True)),
    'social_noinfer': dict(planner='social', soc=dict(lam_s=2.0, infer=False, creep=True)),
    'social_nocreep': dict(planner='social', soc=dict(lam_s=2.0, infer=True, creep=False)),
    'social_impolite': dict(planner='social', soc=dict(lam_s=0.0, infer=True, creep=True)),
}


def worker(a):
    name, density, seed, keep = a
    global SOC
    v = VARIANTS[name]; SOC = v.get('soc', SOC)
    r = run_episode((v['planner'], density, seed, keep)); r['variant'] = name
    return r


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    jobs = [(n, d, 1000 + s, False) for n in VARIANTS for d in ['sparse', 'medium', 'dense'] for s in range(N)]
    with Pool(2) as p: out = p.map(worker, jobs, chunksize=20)
    pd.DataFrame(out).to_csv('data/sim_results.csv', index=False)
    print('hw scale', HW_SCALE)
    print(pd.DataFrame(out).groupby(['variant', 'density'])[['collision', 'done', 'delay', 'stall', 'min_pet', 'rms_jerk', 'opp_mean_decel', 'jain']].mean().round(2))
