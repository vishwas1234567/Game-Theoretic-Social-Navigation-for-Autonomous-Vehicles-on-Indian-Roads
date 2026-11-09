"""Full evaluation: planners x density, ablations, politeness sweep, SOME-style 5-dimension scorecard, figures, LaTeX tables."""
import json, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool
sys.path.insert(0, 'code')
import socialsim as S

plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .3, 'figure.dpi': 150})
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
DENS = ['sparse', 'medium', 'dense']
for lam in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
    S.VARIANTS[f'sweep_{lam}'] = dict(planner='social', soc=dict(lam_s=lam, infer=True, creep=True))
MAIN = ['rigid', 'assertive', 'social', 'social_noinfer', 'social_nocreep', 'social_impolite']
SWEEP = [k for k in S.VARIANTS if k.startswith('sweep_')]


def run(jobs):
    with Pool(2) as p: return pd.DataFrame(p.map(S.worker, jobs, chunksize=25))


df = run([(n, d, 1000 + s, False) for n in MAIN for d in DENS for s in range(N)])
sw = run([(n, d, 5000 + s, False) for n in SWEEP for d in ['medium', 'dense'] for s in range(N // 2)])
df.to_csv('data/sim_results.csv', index=False); sw.to_csv('data/sim_sweep.csv', index=False)

# ------------------------- scorecard -------------------------
K = 6
clip = lambda x: np.clip(x, 0, 1)


def score(d):
    m = pd.DataFrame(index=d.index)
    m['collision'] = 1 - d.collision
    m['PET'] = clip(d.min_pet / 2.0)
    m['jerk'] = clip(1 - (d.rms_jerk - 0.5) / 3.0)
    m['peak_decel'] = clip(1 - (d.peak_decel - 1.5) / 4.5)
    m['delay'] = clip(1 - d.delay / 25.0)
    m['hesitation'] = clip(1 - d.stall / 6.0)
    m['fairness'] = clip((d.jain - 1 / (K + 1)) / (1 - 1 / (K + 1)))
    m['induced_decel'] = clip(1 - d.opp_mean_decel / 3.0)
    m['opp_delay'] = clip(1 - d.opp_delay / 10.0)
    dim = pd.DataFrame({'Safety': m[['collision', 'PET']].mean(1), 'Comfort': m[['jerk', 'peak_decel']].mean(1),
                        'Efficiency': m['delay'], 'Social': m[['hesitation', 'fairness']].mean(1),
                        'Impact': m[['induced_decel', 'opp_delay']].mean(1)})
    return m, dim


W = dict(Safety=.30, Comfort=.15, Efficiency=.20, Social=.20, Impact=.15)
metrics, dims = score(df)
df = pd.concat([df, dims], axis=1)
df['AHP'] = sum(df[k] * w for k, w in W.items())
df['Balanced'] = 1.0 / sum(w / np.maximum(df[k], 0.02) for k, w in W.items())      # weakest-link aware (weighted harmonic mean)
sm, sd_ = score(sw); sw = pd.concat([sw, sd_], axis=1)
sw['lam'] = sw.variant.str.replace('sweep_', '').astype(float)
sw['AHP'] = sum(sw[k] * w for k, w in W.items())
sw['Balanced'] = 1.0 / sum(w / np.maximum(sw[k], 0.02) for k, w in W.items())
df.to_csv('data/sim_scored.csv', index=False)

rng = np.random.default_rng(0)
def ci(x, B=1000):
    x = np.asarray(x); bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return np.percentile(bs, [2.5, 97.5])
def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / den, (c + h) / den

NAMES = {'rigid': 'Rigid-safety', 'assertive': 'Assertive', 'social': 'Social game-theoretic (ours)',
         'social_noinfer': '\\quad w/o type inference', 'social_nocreep': '\\quad w/o inching signal', 'social_impolite': '\\quad w/o politeness term'}
import os; os.makedirs('report/tables', exist_ok=True)

# Table 1: headline (all densities pooled)
rows = []
for v in MAIN:
    d = df[df.variant == v]
    k = int(d.collision.sum()); lo, hi = wilson(k, len(d))
    rows.append([NAMES[v], f"{100 * d.collision.mean():.1f} [{100 * lo:.1f},{100 * hi:.1f}]", f"{d.min_pet.mean():.2f}", f"{d.delay.mean():.1f}",
                 f"{d.stall.mean():.2f}", f"{d.rms_jerk.mean():.2f}", f"{d.opp_mean_decel.mean():.2f}", f"{d.jain.mean():.2f}"])
t1 = pd.DataFrame(rows, columns=['Planner', 'Coll. \\% [95\\% CI]', 'Min PET (s)', 'Delay (s)', 'Stall (s)', 'RMS jerk', 'Induced dec.', 'Jain'])
open('report/tables/headline.tex', 'w').write(t1.to_latex(index=False, escape=False, column_format='lrrrrrrr'))

# Table 2: five-dimension scorecard
rows = []
for v in MAIN:
    d = df[df.variant == v]
    rows.append([NAMES[v]] + [f"{d[k].mean():.2f}" for k in W] + [f"{d.AHP.mean():.3f}", f"{d.Balanced.mean():.3f}"])
t2 = pd.DataFrame(rows, columns=['Planner'] + list(W) + ['AHP-style', 'Balanced'])
open('report/tables/scorecard.tex', 'w').write(t2.to_latex(index=False, escape=False, column_format='lrrrrrrr'))

# Table 3: by density (delay, induced decel, collision) for the three main planners
rows = []
for dn in DENS:
    for v in ['rigid', 'assertive', 'social']:
        d = df[(df.variant == v) & (df.density == dn)]
        rows.append([dn, NAMES[v], f"{100 * d.collision.mean():.1f}", f"{d.delay.mean():.1f}", f"{d.opp_mean_decel.mean():.2f}", f"{d.AHP.mean():.3f}", f"{d.Balanced.mean():.3f}"])
t3 = pd.DataFrame(rows, columns=['Density', 'Planner', 'Coll. \\%', 'Delay (s)', 'Induced dec.', 'AHP', 'Balanced'])
open('report/tables/bydensity.tex', 'w').write(t3.to_latex(index=False, escape=False, column_format='llrrrrr'))

# Table 4: politeness sweep
rows = []
for lam, g in sw.groupby('lam'):
    rows.append([f"{lam:g}", f"{100 * g.collision.mean():.1f}", f"{g.delay.mean():.1f}", f"{g.opp_mean_decel.mean():.2f}", f"{g.opp_hard.mean() * 100:.1f}", f"{g.Balanced.mean():.3f}"])
t4 = pd.DataFrame(rows, columns=['$\\lambda_s$', 'Coll. \\%', 'Delay (s)', 'Induced dec.', 'Opp. hard-brake \\%', 'Balanced'])
open('report/tables/sweep.tex', 'w').write(t4.to_latex(index=False, escape=False, column_format='rrrrrr'))

# paired comparison social vs rigid / assertive (same seeds)
cmp = {}
for base in ['rigid', 'assertive']:
    a = df[df.variant == 'social'].sort_values(['density', 'seed']); b = df[df.variant == base].sort_values(['density', 'seed'])
    for col in ['delay', 'AHP', 'Balanced', 'opp_mean_decel', 'Comfort']:
        diff = a[col].values - b[col].values
        cmp[f'{base}:{col}'] = dict(mean=float(diff.mean()), lo=float(ci(diff)[0]), hi=float(ci(diff)[1]))
json.dump(cmp, open('data/paired_cmp.json', 'w'), indent=1)

# ------------------------- figures -------------------------
cols = {'rigid': 'C3', 'assertive': 'C1', 'social': 'C0', 'social_noinfer': 'C2', 'social_nocreep': 'C4', 'social_impolite': 'C5'}
lab = {'rigid': 'Rigid', 'assertive': 'Assertive', 'social': 'Social (ours)', 'social_noinfer': 'no inference', 'social_nocreep': 'no inching', 'social_impolite': 'no politeness'}
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
x = np.arange(len(W)); wd = .26
for i, v in enumerate(['rigid', 'assertive', 'social']):
    d = df[df.variant == v]
    ax[0].bar(x + (i - 1) * wd, [d[k].mean() for k in W], wd, label=lab[v], color=cols[v], yerr=[[d[k].mean() - ci(d[k])[0] for k in W], [ci(d[k])[1] - d[k].mean() for k in W]], capsize=1.5, error_kw=dict(lw=.6))
ax[0].set_xticks(x); ax[0].set_xticklabels(list(W), rotation=20); ax[0].set_ylim(0, 1.05); ax[0].set_ylabel('dimension score'); ax[0].legend(fontsize=7, loc='lower left')
ax[0].set_title('Five-dimension scorecard')
xs = np.arange(2);
for i, v in enumerate(['rigid', 'assertive', 'social']):
    d = df[df.variant == v]
    ax[1].bar(xs + (i - 1) * wd, [d.AHP.mean(), d.Balanced.mean()], wd, color=cols[v], yerr=[[d[k].mean() - ci(d[k])[0] for k in ['AHP', 'Balanced']], [ci(d[k])[1] - d[k].mean() for k in ['AHP', 'Balanced']]], capsize=2, error_kw=dict(lw=.6))
ax[1].set_xticks(xs); ax[1].set_xticklabels(['AHP-style\n(fixed weights)', 'Balanced\n(weakest-link aware)']); ax[1].set_ylim(0, 1.05); ax[1].set_title('Overall score (95% bootstrap CI)')
plt.tight_layout(); plt.savefig('figures/scorecard.png'); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5), sharex=True)
for j, (col, yl) in enumerate([('delay', 'ego delay (s)'), ('opp_mean_decel', 'induced decel. (m/s$^2$)'), ('Balanced', 'balanced score')]):
    for i, v in enumerate(['rigid', 'assertive', 'social']):
        m = [df[(df.variant == v) & (df.density == dn)][col].mean() for dn in DENS]
        ax[j].plot(DENS, m, 'o-', color=cols[v], label=lab[v])
    ax[j].set_ylabel(yl)
ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig('figures/by_density.png'); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.8))
for dn, mk in [('medium', 'o'), ('dense', 's')]:
    g = sw[sw.density == dn].groupby('lam')
    ax[0].plot(g.opp_mean_decel.mean(), g.delay.mean(), mk + '-', label=dn)
    for lam, (xx, yy) in zip(g.size().index, zip(g.opp_mean_decel.mean(), g.delay.mean())): ax[0].annotate(f'{lam:g}', (xx, yy), fontsize=6, xytext=(3, 3), textcoords='offset points')
ax[0].set_xlabel('mean induced deceleration of others (m/s$^2$)'); ax[0].set_ylabel('ego delay (s)'); ax[0].legend(fontsize=7); ax[0].set_title('Politeness-efficiency frontier ($\\lambda_s$ labelled)')
g = sw.groupby('lam'); ax[1].plot(g.Balanced.mean(), 'o-', label='balanced'); ax[1].plot(g.AHP.mean(), 's--', label='AHP-style'); ax[1].set_xscale('symlog', linthresh=.5); ax[1].set_xlabel('politeness weight $\\lambda_s$'); ax[1].set_ylabel('score'); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig('figures/politeness_sweep.png'); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.8))
for j, col in enumerate(['Balanced', 'delay']):
    vals = [df[df.variant == v][col].mean() for v in MAIN]; errs = [[df[df.variant == v][col].mean() - ci(df[df.variant == v][col])[0] for v in MAIN], [ci(df[df.variant == v][col])[1] - df[df.variant == v][col].mean() for v in MAIN]]
    ax[j].bar(range(len(MAIN)), vals, color=[cols[v] for v in MAIN], yerr=errs, capsize=2, error_kw=dict(lw=.6))
    ax[j].set_xticks(range(len(MAIN))); ax[j].set_xticklabels([lab[v] for v in MAIN], rotation=35, ha='right', fontsize=7); ax[j].set_ylabel({'Balanced': 'balanced score', 'delay': 'ego delay (s)'}[col])
plt.tight_layout(); plt.savefig('figures/ablation.png'); plt.close()

# space-time diagrams: same dense scenario for the three planners
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.8), sharey=True)
seed = 1005
for j, v in enumerate(['rigid', 'assertive', 'social']):
    S.SOC = S.VARIANTS[v].get('soc', S.SOC)
    r = S.run_episode((S.VARIANTS[v]['planner'], 'dense', seed, True)); L = np.array(r['log'])
    for k in range(2, L.shape[1]): ax[j].plot(L[:, 0], L[:, k], color='gray', lw=.9)
    ax[j].plot(L[:, 0], L[:, 1], color=cols[v], lw=2); ax[j].axhspan(-1.0, 1.5 + 4, color='k', alpha=.08)
    ax[j].set_ylim(-25, 12); ax[j].set_title(lab[v] + f"  (delay {r['delay']:.1f}s)", fontsize=8); ax[j].set_xlabel('time (s)')
ax[0].set_ylabel('distance to conflict point (m)')
plt.tight_layout(); plt.savefig('figures/spacetime.png'); plt.close()

# human reference from video vs simulated planner comfort
V = json.load(open('data/video_stats.json'))
summ = {v: dict(collision=float(df[df.variant == v].collision.mean()), delay=float(df[df.variant == v].delay.mean()),
                balanced=float(df[df.variant == v].Balanced.mean()), ahp=float(df[df.variant == v].AHP.mean()),
                rms_jerk=float(df[df.variant == v].rms_jerk.mean()), stall=float(df[df.variant == v].stall.mean()),
                induced=float(df[df.variant == v].opp_mean_decel.mean()), n=int((df.variant == v).sum())) for v in MAIN}
json.dump(summ, open('data/sim_summary.json', 'w'), indent=1)
print(json.dumps(summ, indent=1)); print(t2.to_string()); print(json.dumps(cmp, indent=1)); print(t4.to_string())
