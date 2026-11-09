"""ODD experiment: (1) ODD conformance of the clip, (2) planner behaviour as perception quality q leaves the ODD."""
import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.ndimage import gaussian_filter1d
sys.path.insert(0, 'code'); import socialsim as S
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .3, 'figure.dpi': 150})

# ---------------- (1) ODD conformance from the video ----------------
F = pd.read_csv('data/odd_features.csv'); P = pd.read_csv('data/perception_derived.csv')
D = pd.merge_asof(F.sort_values('t'), P[['t', 'crowd', 'v', 'a', 'jerk', 'n_white']].sort_values('t'), on='t')
drv = D.t >= 49.1
SPEC = dict(lane=0.15, dark=0.12, glare=0.02, crowd=0.15, sharp=800.0)       # illustrative structured-ODD thresholds (assumed)


Qf = pd.DataFrame({c: gaussian_filter1d(v, 8) for c, v in {
    'Lane markings': np.clip(D.lane_score / SPEC['lane'], 0, 1),
    'Lighting': np.clip(1 - (D.dark_frac - SPEC['dark']) / SPEC['dark'], 0, 1),
    'Glare': np.clip(1 - (D.glare_frac - SPEC['glare']) / (2 * SPEC['glare']), 0, 1),
    'Traffic complexity': np.clip(1 - (D.crowd - SPEC['crowd']) / SPEC['crowd'], 0, 1),
    'Image sharpness': np.clip(D.sharp / SPEC['sharp'], 0, 1)}.items()})
Qf['t'] = D.t.values
PERC = ['Lighting', 'Glare', 'Traffic complexity', 'Image sharpness']
Qf['q_all'] = Qf[['Lane markings'] + PERC].min(axis=1)
Qf['q_perc'] = Qf[PERC].min(axis=1)
RAW = pd.DataFrame({'Lane markings': D.lane_score >= SPEC['lane'], 'Lighting': D.dark_frac <= SPEC['dark'], 'Glare': D.glare_frac <= SPEC['glare'],
                    'Traffic complexity': D.crowd <= SPEC['crowd'], 'Image sharpness': D.sharp >= SPEC['sharp']})
inside_raw = RAW.all(axis=1)
from scipy.ndimage import median_filter
inside = pd.Series(median_filter(inside_raw.values.astype(int), 15).astype(bool), index=inside_raw.index)   # 1 s debounce for exit counting
m = drv.values
res = dict(frac_inside_all=float(inside_raw[m].mean()),
           viol={c: float(1 - RAW[c][m].mean()) for c in RAW.columns},
           mean_q_all=float(Qf.q_all[m].mean()), mean_q_perc=float(Qf.q_perc[m].mean()),
           frac_inside_perc=float(RAW[PERC][m].all(axis=1).mean()))
ins = inside[m].values.astype(int); trans = np.diff(ins); T_drv = m.sum() / 15 / 60
res['exits_per_min'] = float((trans == -1).sum() / T_drv)
res['mean_dwell_inside_s'] = float(ins.sum() / 15 / max((trans == 1).sum() + (ins[0] == 1), 1))
# sensitivity: scale all thresholds
sens = []
for sc in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
    qq = pd.DataFrame({
        'l': np.clip(D.lane_score / (SPEC['lane'] / sc if sc < 1 else SPEC['lane'] / sc), 0, 1) if False else np.clip(D.lane_score / (SPEC['lane'] / sc), 0, 1),
        'd': (D.dark_frac <= SPEC['dark'] * sc), 'g': (D.glare_frac <= SPEC['glare'] * sc),
        'c': (D.crowd <= SPEC['crowd'] * sc), 's': (D.sharp >= SPEC['sharp'] / sc)})
    ok = (D.lane_score >= SPEC['lane'] / sc) & qq.d & qq.g & qq.c & qq.s
    sens.append(dict(scale=sc, inside=float(ok[drv].mean()), inside_no_lane=float((qq.d & qq.g & qq.c & qq.s)[drv].mean())))
res['sensitivity'] = sens
# does the human driver behave differently outside the ODD? (speed, jerk) -- descriptive only
out = ~inside_raw
res['human_in_odd'] = dict(frac=float(inside_raw[m].mean()), speed=float(D.v[m & inside_raw.values].mean()) if (m & inside_raw.values).any() else None)
res['human_out_odd'] = dict(speed=float(D.v[m & out.values].mean()), rms_jerk=float(np.sqrt((D.jerk[m & out.values] ** 2).mean())))
if (m & inside_raw.values).sum() > 30: res['human_in_odd']['rms_jerk'] = float(np.sqrt((D.jerk[m & inside_raw.values] ** 2).mean()))
res['q_perc_quantiles'] = {str(k): float(Qf.q_perc[m].quantile(k)) for k in [.1, .25, .5, .75, .9]}
json.dump(res, open('data/odd_video.json', 'w'), indent=1)
Qf.to_csv('data/odd_quality.csv', index=False)

fig, ax = plt.subplots(3, 1, figsize=(7.2, 6.2), sharex=True)
cols = ['C0', 'C1', 'C2', 'C3', 'C4']
for c, col in zip(['Lane markings'] + PERC, cols): ax[0].plot(Qf.t, Qf[c], lw=.8, color=col, label=c)
ax[0].set_ylabel('attribute quality'); ax[0].legend(fontsize=6, ncol=3, loc='lower right'); ax[0].set_title('ODD attribute qualities from the clip (1 = inside the illustrative ODD)')
ax[1].plot(Qf.t, Qf.q_all, color='k', lw=.8, label='all attributes (strict ODD)'); ax[1].plot(Qf.t, Qf.q_perc, color='C3', lw=.8, label='perception-relevant only')
ax[1].axhline(.85, ls='--', c='gray', lw=.7); ax[1].set_ylabel('ODD quality $q$'); ax[1].legend(fontsize=7)
ax[2].fill_between(Qf.t, 0, inside_raw.astype(float), step='mid', color='C2', alpha=.5); ax[2].set_yticks([0, 1]); ax[2].set_yticklabels(['outside', 'inside']); ax[2].set_xlabel('time (s)')
ax[2].set_ylabel('strict ODD status')
for a in ax: a.axvspan(0, 49.1, color='gray', alpha=.12)
plt.tight_layout(); plt.savefig('figures/odd_timeline.png'); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.8))
names = ['Lane markings'] + PERC
ax[0].barh(names, [res['viol'][c] * 100 for c in names], color=cols); ax[0].set_xlabel('% of driving time outside spec'); ax[0].invert_yaxis()
ax[1].plot([s['scale'] for s in sens], [s['inside'] * 100 for s in sens], 'o-', label='all five attributes'); ax[1].plot([s['scale'] for s in sens], [s['inside_no_lane'] * 100 for s in sens], 's--', label='without lane markings')
ax[1].set_xscale('log'); ax[1].set_xlabel('threshold relaxation factor'); ax[1].set_ylabel('% time inside ODD'); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig('figures/odd_violations.png'); plt.close()

# ---------------- (2) planner behaviour vs ODD quality ----------------
POLS = ['strict', 'naive', 'track', 'margin']


def job(a):
    pol, q, seed = a
    S.SOC = dict(lam_s=2.0, infer=True, creep=True)
    r = S.run_episode(('social', 'medium', seed, False, (q, pol)))
    r['pol'] = pol; return r


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    qs = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    with Pool(2) as p:
        sweep = pd.DataFrame(p.map(job, [(pol, q, 7000 + s) for pol in POLS for q in qs for s in range(N)], chunksize=30))
        # q sampled from the video's perception-quality distribution (driving portion)
        pool_q = Qf.q_perc[m].values; rng = np.random.default_rng(3)
        vid = pd.DataFrame(p.map(job, [(pol, float(rng.choice(pool_q)), 9000 + s) for pol in POLS for s in range(N * 3)], chunksize=30))
    sweep.to_csv('data/odd_sweep.csv', index=False); vid.to_csv('data/odd_video_episodes.csv', index=False)
    sweep['success'] = ((sweep.done == 1) & (sweep.collision == 0)).astype(int)
    vid['success'] = ((vid.done == 1) & (vid.collision == 0)).astype(int)
    g = sweep.groupby(['pol', 'q']).agg(coll=('collision', 'mean'), avail=('available', 'mean'), succ=('success', 'mean'), delay=('delay', lambda x: x[sweep.loc[x.index, 'available'] == 1].mean())).reset_index()
    lab = {'strict': 'Strict-ODD (disengage if $q<0.85$)', 'naive': 'ODD-unaware', 'track': 'ODD-aware: track persistence', 'margin': 'ODD-aware: persistence + margin inflation'}
    colr = {'strict': 'C7', 'naive': 'C3', 'track': 'C0', 'margin': 'C1'}
    fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.8))
    for pol in POLS:
        d = g[g.pol == pol].sort_values('q')
        ax[0].plot(d.q, d.coll * 100, 'o-', ms=3, color=colr[pol], label=lab[pol]); ax[1].plot(d.q, d.avail * 100, 'o-', ms=3, color=colr[pol]); ax[2].plot(d.q, d.succ * 100, 'o-', ms=3, color=colr[pol])
    ax[0].set_ylabel('conflicts per 100 trips'); ax[1].set_ylabel('availability (% trips started)'); ax[2].set_ylabel('trip success (% done, no conflict)')
    for a in ax: a.set_xlabel('ODD quality $q$'); a.invert_xaxis()
    fig.legend(*ax[0].get_legend_handles_labels(), fontsize=6.5, loc='lower center', ncol=2, frameon=False)
    plt.tight_layout(rect=(0, .14, 1, 1)); plt.savefig('figures/odd_sweep.png'); plt.close()
    rows = []
    for pol in POLS:
        d = vid[vid.pol == pol]; k = int(d.collision.sum()); n = len(d)
        rows.append(dict(pol=pol, n=n, coll=100 * d.collision.mean(), avail=100 * d.available.mean(), succ=100 * d.success.mean(), delay=d[d.available == 1].delay.mean(), pet=d[d.available == 1].min_pet.mean(),
                         coll_given_started=100 * d[d.available == 1].collision.mean()))
    R = pd.DataFrame(rows); R.to_csv('data/odd_video_summary.csv', index=False)
    NAMES = {'strict': 'Strict-ODD (geofence-style)', 'naive': 'ODD-unaware', 'track': 'ODD-aware: persistence', 'margin': 'ODD-aware: persistence + margins'}
    tb = pd.DataFrame({'Policy': [NAMES[p] for p in R.pol], 'Avail. \\%': R.avail.map('{:.1f}'.format), 'Coll. \\% (of all)': R.coll.map('{:.2f}'.format), 'Coll. \\% (if started)': R.coll_given_started.map('{:.2f}'.format),
                       'Success \\%': R.succ.map('{:.1f}'.format), 'Delay (s)': R.delay.map('{:.1f}'.format)})
    open('report/tables/odd_video.tex', 'w').write(tb.to_latex(index=False, escape=False, column_format='lrrrrr'))
    g.to_csv('data/odd_sweep_summary.csv', index=False)
    print(json.dumps({k: v for k, v in res.items() if k != 'q_perc_quantiles'}, indent=1)); print(res['q_perc_quantiles']); print(R.round(3).to_string())
    print(g[g.q.isin([1.0, 0.7, 0.5, 0.3, 0.1])].round(3).to_string())
