#!/usr/bin/env python3
"""
Offline verification from canonical IRT parameters.
Does NOT re-fit the 2PL model — loads saved canonical theta/b.

Usage:
    python src/reproduce_from_derived.py              # ~15s full verification
    python src/reproduce_from_derived.py --quick      # ~15s rankings + crisis (skips LLTM)
    python src/reproduce_from_derived.py --stress-refit  # ~60s re-runs 1PL stress

Note: --stress-refit may produce 6/10 or 7/10 IRT wins depending on
numpy/scipy version. The canonical result (7/10) was obtained with
numpy 1.26 / scipy 1.12.

Writes:
    outputs/results_from_derived.json
    paper/figures_verification/*.pdf
"""
import numpy as np, pandas as pd, json, argparse
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr, pearsonr, kendalltau, mannwhitneyu
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--quick', action='store_true')
parser.add_argument('--stress-refit', action='store_true')
args = parser.parse_args()
np.random.seed(42)

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
    'font.size':9,'axes.labelsize':10,'axes.titlesize':11,'xtick.labelsize':8,
    'ytick.labelsize':8,'legend.fontsize':8,'figure.dpi':300,
    'savefig.bbox':'tight','savefig.pad_inches':0.08,
    'pdf.fonttype':42,'ps.fonttype':42,'mathtext.fontset':'dejavusans'})

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT/'paper'/'figures_verification'; OUTDIR.mkdir(parents=True, exist_ok=True)
print("="*60); print("OFFLINE VERIFICATION (canonical params)"); print("="*60)

# ─── LOAD ───
R = pd.read_csv(ROOT/'data'/'response_matrix.csv', index_col=0)
R = R.apply(pd.to_numeric, errors='coerce').dropna(axis=1,how='all').dropna(axis=0,how='all')
feat = pd.read_csv(ROOT/'data'/'period_features.csv')
th_df = pd.read_csv(ROOT/'data'/'canonical_theta.csv')
pr_df = pd.read_csv(ROOT/'data'/'canonical_period_params.csv')
nf, ni = R.shape
theta = th_df.set_index('fund')['theta']
b_irt = pr_df['b'].values
print(f"Loaded: {nf}x{ni} matrix, canonical params")

# ─── RANKINGS ───
avg_r = R.mean(axis=1).rank(ascending=False)
irt_r = theta.reindex(R.index).rank(ascending=False)
rd = avg_r - irt_r
rho, _ = spearmanr(avg_r, irt_r); tau, _ = kendalltau(avg_r, irt_r)
print(f"rho={rho:.4f} tau={tau:.4f} max|D|={rd.abs().max():.1f}")

META = {'SPY':('S&P 500','US'),'QQQ':('Nasdaq 100','US'),'IWM':('Russell 2000','US'),
    'DIA':('Dow Jones','US'),'ARKK':('ARK Innov.','US'),'VTI':('Total US','US'),
    'VOO':('Vgd S&P500','US'),'EWG':('Germany','EU'),'EWU':('UK','EU'),
    'EWQ':('France','EU'),'EWI':('Italy','EU'),'EWP':('Spain','EU'),
    'EWL':('Switz.','EU'),'EWN':('Neth.','EU'),'EWD':('Sweden','EU'),
    'EWK':('Belgium','EU'),'EWJ':('Japan','Asia'),'EWY':('S. Korea','Asia'),
    'EWA':('Australia','Asia'),'EWT':('Taiwan','Asia'),'EWS':('Singapore','Asia'),
    'EWH':('Hong Kong','Asia'),'MCHI':('China','Asia'),'EEM':('EM Broad','EM'),
    'EWZ':('Brazil','EM'),'INDA':('India','EM'),'EWW':('Mexico','EM'),
    'TUR':('Turkey','EM'),'EZA':('S. Africa','EM'),'ECH':('Chile','EM'),
    'EPOL':('Poland','EM'),'XLK':('Tech','US'),'XLF':('Financials','US'),
    'XLE':('Energy','US'),'XLV':('Healthcare','US'),'XLRE':('Real Est.','US'),
    'SMH':('Semicons','US'),'XLU':('Utilities','US'),'XLI':('Industrials','US'),
    'XLP':('Cons.Stpl','US'),'XLB':('Materials','US'),'XLC':('Comm.Svcs','US'),
    'GLD':('Gold','Glb'),'SLV':('Silver','Glb'),'TLT':('Long Bonds','US'),
    'HYG':('High Yield','US'),'LQD':('Inv.Grade','US'),'TIP':('TIPS','US'),
    'VNQ':('US REITs','US'),'IBIT':('Bitcoin','Glb')}
df = pd.DataFrame({'name':[META.get(t,(t,'?'))[0] for t in R.index],
    'region':[META.get(t,(t,'?'))[1] for t in R.index],
    'beat':R.mean(axis=1),'theta':theta.reindex(R.index),'irt_r':irt_r,'delta':rd}).sort_values('irt_r')

# ─── CRISIS (10k perms, vectorized) ───
print("Crisis validation...")
CRISES = {'GFC':('2008-09','2009-03'),'Flash':('2010-05','2010-07'),
    'EUDebt':('2011-08','2011-10'),'China':('2015-08','2016-02'),
    'Volmag':('2018-10','2018-12'),'COVID':('2020-02','2020-04'),
    'RateHike':('2022-01','2022-10')}
periods = feat['period'].values[:ni]
beat_vals = np.array([float(R.iloc[:,i].mean()) for i in range(ni)])
crisis_flags = feat['is_crisis'].values[:ni].astype(int)
cb = b_irt[crisis_flags==1]; nb = b_irt[crisis_flags==0]
_, p_mw = mannwhitneyu(cb, nb, alternative='two-sided')
r_beat_b, _ = pearsonr(beat_vals, b_irt[:ni])

# Precompute episode lengths (pandas ONCE, then pure numpy)
ep_lens = []
ep_means_obs = []
for cn,(cs,ce) in CRISES.items():
    mask = (periods >= cs) & (periods <= ce)
    el = int(mask.sum())
    ep_lens.append(el)
    if el > 0: ep_means_obs.append(b_irt[mask].mean())
obs_ep = np.mean(ep_means_obs)
n_all = len(b_irt)

# Pure numpy permutation loop
perms = np.empty(10000)
for i in range(10000):
    pm = []
    for el in ep_lens:
        if el == 0: continue
        si = np.random.randint(0, max(1, n_all - el))
        pm.append(b_irt[si:si+el].mean())
    perms[i] = np.mean(pm)
p_block = float(np.mean(np.abs(perms - perms.mean()) >= np.abs(obs_ep - perms.mean())))
print(f"  MW p={p_mw:.4f} Block p={p_block:.4f} r={r_beat_b:.4f}")

# ─── LLTM ───
r2_lltm = cv_mean = None; etas = None
if not args.quick:
    print("LLTM...")
    Q = feat[['vix_avg','dispersion','acwi_return','credit_spread','is_crisis']].values[:ni]
    Qs = StandardScaler().fit_transform(Q); vld = np.all(np.isfinite(Qs), axis=1)
    lltm = Ridge(alpha=1.0); lltm.fit(Qs[vld], b_irt[vld])
    r2_lltm = r2_score(b_irt[vld], lltm.predict(Qs[vld]))
    cv_sc = cross_val_score(Ridge(alpha=1.0), Qs[vld], b_irt[vld], cv=10, scoring='r2')
    cv_mean = float(cv_sc.mean())
    etas = sorted(zip(['vix_avg','dispersion','acwi_return','credit_spread','is_crisis'],
                      lltm.coef_), key=lambda x:abs(x[1]), reverse=True)
    print(f"  R2={r2_lltm:.4f} CV={cv_mean:.4f}")
else:
    print("LLTM: skipped (--quick)")

# ─── STRESS ───
if args.stress_refit:
    print("Stress test (1PL refit)...")
    R_v = R.values.copy(); om = ~np.isnan(R_v)
    tr = pd.Series(theta.reindex(R.index).values, index=R.index).rank(ascending=False)
    sr_list = []
    for s in [0,.1,.2,.3,.4,.5,.6,.7,.8,.9]:
        ras, ris = [], []
        for seed in range(2):
            rng = np.random.RandomState(42+seed)
            Rs = R_v.copy(); Rs[om & (rng.random(R_v.shape)<s)] = np.nan
            ro = np.any(~np.isnan(Rs),1); co = np.any(~np.isnan(Rs),0)
            if ro.sum()<5 or co.sum()<5: continue
            Rss = Rs[np.ix_(ro,co)]; idx = R.index[ro]
            ar = pd.Series(np.nanmean(Rss,1), index=idx).rank(ascending=False)
            try:
                f2,i2 = Rss.shape; A=Rss.copy(); M=~np.isnan(A); A[~M]=0
                def rnll(p):
                    pr = np.clip(expit(p[:f2,None]-p[f2:][None,:]),1e-10,1-1e-10)
                    return -(A*np.log(pr)+(1-A)*np.log(1-pr))[M].sum()
                r2 = minimize(rnll, np.zeros(f2+i2), method='L-BFGS-B', options={'maxiter':200})
                ts = r2.x[:f2]; ts -= ts.mean()
                ir = pd.Series(ts, index=idx).rank(ascending=False)
            except: ir = ar
            c = tr.index.intersection(idx)
            if len(c)<5: continue
            ra,_ = spearmanr(tr[c],ar.reindex(c)); ri,_ = spearmanr(tr[c],ir.reindex(c))
            if np.isfinite(ra): ras.append(ra)
            if np.isfinite(ri): ris.append(ri)
        ram = np.mean(ras) if ras else np.nan; rim = np.mean(ris) if ris else np.nan
        sr_list.append({'s':s,'rho_avg':ram,'rho_irt':rim,
            'adv':rim-ram if np.isfinite(rim) and np.isfinite(ram) else np.nan})
    sr = pd.DataFrame(sr_list); vs = sr.dropna(subset=['adv'])
    irt_wins = int((vs.adv>0).sum())
    print(f"  1PL wins: {irt_wins}/{len(vs)}")
else:
    with open(ROOT/'data'/'canonical_stress.json') as f: sc = json.load(f)
    irt_wins = sc['irt_wins']; sr = pd.DataFrame()
    print(f"Stress: canonical={irt_wins}/{sc['total']} (--stress-refit to re-run)")

# ─── FIGURES ───
print("Figures...")
movers = rd.abs().sort_values(ascending=False)
top12 = df.loc[movers.head(12).index].sort_values('delta')
fig,ax = plt.subplots(figsize=(5.5,4))
colors = ['#2166ac' if d>0 else '#b2182b' for d in top12['delta']]
ax.barh(range(len(top12)), top12['delta'], color=colors, height=0.7)
ax.set_yticks(range(len(top12)))
ax.set_yticklabels([f"{top12.index[i]} ({top12.iloc[i]['name']})" for i in range(len(top12))], fontsize=7)
ax.set_xlabel('Rank change'); ax.axvline(0, color='gray', lw=0.5); ax.set_xlim(-36,36)
for i,v in enumerate(top12['delta']):
    ax.text(v+(1.2 if v>0 else -1.2), i, f'{v:+.1f}', va='center',
            ha='left' if v>0 else 'right', fontsize=6.5, clip_on=False)
plt.subplots_adjust(left=0.24, right=0.92)
plt.savefig(OUTDIR/'fig1_rank_corrections.pdf'); plt.close()

fig,axes = plt.subplots(1,2,figsize=(5.5,2.8))
axes[0].boxplot([cb,nb], tick_labels=[f'Crisis\n(n={len(cb)})',f'Normal\n(n={len(nb)})'],
    patch_artist=True, widths=0.5)
axes[0].set_ylabel('IRT difficulty (b)'); axes[0].set_title(f'MW p={p_mw:.4f}\nBlock p={p_block:.4f}', fontsize=9)
axes[1].scatter(beat_vals, b_irt[:ni], c=['#d6604d' if c else '#4393c3' for c in crisis_flags],
    s=12, alpha=0.5, edgecolors='white', linewidth=0.3)
z = np.polyfit(beat_vals, b_irt[:ni], 1)
xl = np.linspace(beat_vals.min(), beat_vals.max(), 100)
axes[1].plot(xl, np.polyval(z,xl), 'k--', lw=0.8, alpha=0.5)
axes[1].set_xlabel('Beat rate'); axes[1].set_ylabel('IRT difficulty (b)')
axes[1].set_title(f'r = {r_beat_b:.3f}', fontsize=9)
plt.tight_layout(); plt.savefig(OUTDIR/'fig2_crisis_validation.pdf'); plt.close()

if etas:
    fig,ax = plt.subplots(figsize=(5.5,2.5))
    fn=[e[0] for e in etas]; fw=[e[1] for e in etas]
    ax.barh(range(len(fn)), fw, color=['#2166ac' if w>=0 else '#b2182b' for w in fw], height=0.6)
    ax.set_yticks(range(len(fn))); ax.set_yticklabels(fn, fontsize=8)
    ax.axvline(0, color='gray', lw=0.5); ax.set_xlabel('LLTM weight')
    ax.set_title(f'LLTM: R2={r2_lltm:.3f}, CV={cv_mean:.3f}', fontsize=9)
    plt.tight_layout(); plt.savefig(OUTDIR/'fig3_lltm_weights.pdf'); plt.close()

if args.stress_refit and len(sr)>0:
    fig,ax = plt.subplots(figsize=(5.5,3))
    vs2 = sr.dropna(subset=['rho_avg','rho_irt'])
    ax.plot(vs2['s']*100, vs2['rho_avg'], 'o-', color='#b2182b', ms=4, lw=1.5, label='Simple average')
    ax.plot(vs2['s']*100, vs2['rho_irt'], 's-', color='#2166ac', ms=4, lw=1.5, label='1PL IRT')
    ax.set_xlabel('Artificial sparsity (%)'); ax.set_ylabel('Spearman rho')
    ax.legend(frameon=False); ax.set_ylim(0,1)
    ax.set_title(f'Stress test (1PL, IRT wins {irt_wins}/{len(vs)})')
    plt.tight_layout(); plt.savefig(OUTDIR/'fig5_stress_test.pdf'); plt.close()

reg_stats = {}
for reg in ['US','EU','Asia','EM','Glb']:
    rf = [t for t in df.index if df.loc[t,'region']==reg]
    if len(rf)>=2:
        reg_stats[reg] = {'n':len(rf),'avg_theta':float(df.loc[rf,'theta'].mean()),
            'avg_abs_delta':float(df.loc[rf,'delta'].abs().mean())}
fig,ax = plt.subplots(figsize=(5.5,2.8))
regs = [r for r in ['US','EU','Asia','EM','Glb'] if r in reg_stats]; x = range(len(regs)); w = 0.35
ax.bar([i-w/2 for i in x], [reg_stats[r]['avg_theta'] for r in regs], w, color='#4393c3', label='Avg theta')
ax2 = ax.twinx()
ax2.bar([i+w/2 for i in x], [reg_stats[r]['avg_abs_delta'] for r in regs], w, color='#d6604d', label='Avg |delta|')
ax.set_xticks(x); ax.set_xticklabels([f"{r}\n(n={reg_stats[r]['n']})" for r in regs], fontsize=8)
ax.set_ylabel('Avg theta', color='#4393c3'); ax2.set_ylabel('Avg |delta|', color='#d6604d')
ax.axhline(0, color='gray', lw=0.5)
l1,la1 = ax.get_legend_handles_labels(); l2,la2 = ax2.get_legend_handles_labels()
ax.legend(l1+l2, la1+la2, loc='upper right', frameon=False, fontsize=7)
plt.tight_layout(); plt.savefig(OUTDIR/'fig6_regions.pdf'); plt.close()

# ─── SAVE (stress_irt_wins as int) ───
results = {'mode':'offline_canonical_params','n_funds':nf,'n_items':ni,
    'rho_sp':float(rho),'tau':float(tau),'max_abs_delta':float(rd.abs().max()),
    'crisis_p_mw':float(p_mw),'crisis_p_block':p_block,'crisis_permutations':10000,
    'r_beat_b':float(r_beat_b),'stress_irt_wins':int(irt_wins) if isinstance(irt_wins,int) else irt_wins}
if r2_lltm is not None:
    results['lltm_r2'] = float(r2_lltm); results['lltm_cv_mean'] = cv_mean
with open(ROOT/'outputs'/'results_from_derived.json','w') as f:
    json.dump(results, f, indent=2)
print(f"\nDone. outputs/results_from_derived.json + paper/figures_verification/")
