"""Turn per-frame perception into behaviour statistics + figures + JSON used by the simulator & report."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter1d

plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .3, 'figure.dpi': 150})
D = pd.read_csv('data/perception.csv')
dt = 1 / 15.0
F_PX, CAM_H = 160 / np.tan(np.deg2rad(35)), 1.2     # ASSUMED: 70 deg HFOV at 320 px, 1.2 m camera height
SCALE = F_PX * CAM_H / dt

# consensus of two independent flow-based estimators (ground-plane flow ratio & radial expansion rate)
va = D.v_med.interpolate(limit=15, limit_direction='both').fillna(0).values
vb = D.k_exp.interpolate(limit=15, limit_direction='both').fillna(0).values
va = median_filter(va, 9); vb = median_filter(vb, 9)
mv = (va > 0.0008) & (vb > 0.01)
ratio = np.median(va[mv] / vb[mv])              # puts expansion-rate on the ground-plane scale
va_s, vb_s = va * SCALE, vb * ratio * SCALE
v = np.clip(0.5 * (va_s + vb_s), 0, 14)
v = gaussian_filter1d(v, 10)
est_corr = float(np.corrcoef(gaussian_filter1d(va_s, 10), gaussian_filter1d(vb_s, 10))[0, 1])
t = D.t.values
a = np.gradient(v, dt); a = gaussian_filter1d(a, 2)
j = np.gradient(a, dt); j = gaussian_filter1d(j, 2)
D['v'], D['a'], D['jerk'] = v, a, j

# lead vehicle range from tail-light pair spacing (assumed 1.5 m lamp spacing)
Z = F_PX * 1.5 / D.lamp_sep_px
D['lead_Z'] = Z.where((Z > 3) & (Z < 60))
D['headway'] = D.lead_Z / np.maximum(v, 1.0)
Zs = D.lead_Z.interpolate(limit=3)
closing = -np.gradient(gaussian_filter1d(Zs.ffill().fillna(0).values, 2), dt)
D['ttc'] = np.where((D.lead_Z.notna()) & (closing > 0.5), D.lead_Z / np.maximum(closing, 1e-3), np.nan)

# agent-in-corridor and crowding proxies
D['crowd'] = gaussian_filter1d(D.indep_all.fillna(0).values, 8)
D['corr_occ'] = gaussian_filter1d(D.indep_corr.fillna(0).values, 4)
D['lights'] = gaussian_filter1d((D.n_red + 0.25 * D.n_white).values, 8)

moving = v > 1.0
t_move = float(t[np.argmax(moving)])
drv = t >= t_move
stop = v < 0.8
# stop segments (> 1 s)
seg = []; s0 = None
for i, s in enumerate(stop):
    if s and s0 is None: s0 = i
    if (not s or i == len(stop) - 1) and s0 is not None:
        if (t[i] - t[s0]) >= 1.0 and t[s0] >= t_move - 1: seg.append((t[s0], t[i]))
        s0 = None
# decel episodes: a < -1.0 for >= 0.5 s, label 'yield' if corridor occupied or lead present
dec = a < -1.0
ep = []; s0 = None
for i, d in enumerate(dec):
    if d and s0 is None: s0 = i
    if (not d or i == len(dec) - 1) and s0 is not None:
        if (t[i] - t[s0]) >= 0.5 and t[s0] >= t_move:
            sl = slice(s0, i + 1)
            ep.append(dict(t0=float(t[s0]), t1=float(t[i]), peak=float(a[sl].min()),
                           occ=float(D.corr_occ.values[sl].mean()),
                           lead=bool(D.lead_Z.iloc[sl].notna().any())))
        s0 = None
E = pd.DataFrame(ep)
occ_thr = float(np.nanpercentile(D.corr_occ, 60))
E['interactive'] = (E.occ > occ_thr) | E.lead
stats = dict(
    duration_s=float(t[-1]), speed_estimator_corr=est_corr, t_first_move=t_move, frames_sampled=int(len(D)),
    mean_speed_mps=float(v[drv].mean()), moving_mean_speed_mps=float(v[moving].mean()),
    p90_speed_mps=float(np.percentile(v, 90)),
    frac_time_stopped=float(stop[drv].mean()), parked_prefix_s=t_move, n_stops=len(seg), mean_stop_s=float(np.mean([b - a_ for a_, b in seg])) if seg else 0,
    rms_jerk=float(np.sqrt(np.mean(j[drv] ** 2))), rms_acc=float(np.sqrt(np.mean(a[drv] ** 2))),
    p5_acc=float(np.percentile(a[drv], 5)), p95_acc=float(np.percentile(a[drv], 95)),
    max_decel=float(-a[drv].min()),
    n_decel_episodes=int(len(E)), n_interactive_decel=int(E.interactive.sum()),
    lead_visible_frac=float(D.lead_Z.notna().mean()), median_lead_range_m=float(D.lead_Z.median()),
    median_headway_s=float(D.headway.dropna().median()), p10_headway_s=float(D.headway.dropna().quantile(.1)),
    min_ttc_s=float(D.ttc.dropna().quantile(.02)) if D.ttc.notna().any() else None,
    mean_independent_motion=float(D.indep_all.mean()), mean_corridor_occ=float(D.indep_corr.mean()),
    mean_red=float(D.n_red.mean()), mean_white=float(D.n_white.mean()),
    ped_frames_frac=float((D.n_ped > 0).mean()),
    lead_headways=D.headway.dropna().clip(0, 15).round(2).tolist()[:4000],
    speed_moving=np.round(v[moving][::5], 2).tolist(),
    crowd_p33=float(np.percentile(D.crowd, 33)), crowd_p66=float(np.percentile(D.crowd, 66)),
    assumptions=dict(hfov_deg=70, cam_height_m=1.2, lamp_spacing_m=1.5),
)
# density terciles -> speed/stop stats (shows interaction between crowding and ego behaviour)
q1, q2 = np.percentile(D.crowd[drv], 33), np.percentile(D.crowd[drv], 66)
lab = np.where(D.crowd < q1, 'sparse', np.where(D.crowd < q2, 'medium', 'dense'))
D['density'] = lab
by = {}
for k in ['sparse', 'medium', 'dense']:
    m = (lab == k) & drv
    by[k] = dict(frac=float(m.mean()), mean_speed=float(v[m].mean()), stopped=float(stop[m].mean()),
                 rms_jerk=float(np.sqrt(np.mean(j[m] ** 2))), mean_lights=float(D.lights[m].mean()),
                 decel_rate_per_min=float(sum(1 for e in ep if m[min(np.searchsorted(t, e['t0']), len(m)-1)]) / (m.sum() * dt / 60 + 1e-9)))
stats['by_density'] = by
corr = float(np.corrcoef(D.crowd[drv], v[drv])[0, 1]); stats['corr_crowd_speed'] = corr
json.dump(stats, open('data/video_stats.json', 'w'), indent=1)
D.to_csv('data/perception_derived.csv', index=False); E.to_csv('data/decel_episodes.csv', index=False)

# ------------------ figures ------------------
fig, ax = plt.subplots(4, 1, figsize=(7.2, 7.2), sharex=True)
ax[0].plot(t, v * 3.6, lw=.8); ax[0].set_ylabel('ego speed (km/h)')
ax[0].axvspan(0, t_move, color='gray', alpha=.15)
for s, e in seg: ax[0].axvspan(s, e, color='r', alpha=.2)
ax[0].set_title('Ego behaviour reconstructed from the night clip (grey = parked at start, red = stops > 1 s)')
ax[1].plot(t, a, lw=.6, color='C1'); ax[1].axhline(-1, ls='--', c='k', lw=.6); ax[1].set_ylabel('accel (m/s$^2$)')
ax[2].plot(t, D.crowd, c='C2', label='independent-motion fraction'); ax[2].plot(t, D.corr_occ, c='C3', label='ego-corridor occupancy', lw=.7)
ax[2].set_ylabel('crowding proxy'); ax[2].legend(fontsize=7, loc='upper right')
ax[3].plot(t, D.headway, '.', ms=1.5, c='C4'); ax[3].set_ylim(0, 8); ax[3].set_ylabel('lead headway (s)'); ax[3].set_xlabel('time (s)')
plt.tight_layout(); plt.savefig('figures/video_timeline.png'); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
ax[0].hist(v[moving] * 3.6, 40, color='C0'); ax[0].set_xlabel('speed when moving (km/h)')
ax[1].hist(a, 60, color='C1'); ax[1].set_xlabel('acceleration (m/s$^2$)'); ax[1].set_yscale('log')
h = D.headway.dropna().clip(0, 10); ax[2].hist(h, 40, color='C4'); ax[2].set_xlabel('time headway to lead (s)')
plt.tight_layout(); plt.savefig('figures/video_distributions.png'); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
ks = ['sparse', 'medium', 'dense']
ax[0].bar(ks, [by[k]['mean_speed'] * 3.6 for k in ks], color=['C2', 'C1', 'C3']); ax[0].set_ylabel('mean speed (km/h)')
ax[1].bar(ks, [by[k]['rms_jerk'] for k in ks], color=['C2', 'C1', 'C3']); ax[1].set_ylabel('RMS jerk (m/s$^3$)')
ax[2].bar(ks, [by[k]['decel_rate_per_min'] for k in ks], color=['C2', 'C1', 'C3']); ax[2].set_ylabel('hard-decel events / min')
plt.tight_layout(); plt.savefig('figures/video_density.png'); plt.close()
print(json.dumps({k: v_ for k, v_ in stats.items() if not isinstance(v_, list)}, indent=1))
