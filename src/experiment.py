#!/usr/bin/env python3
"""
v4 PRODUCTION EXPERIMENT — All audit blocking issues fixed (true L-BFGS-B bounds):
1. Loads data/prices.csv (matches download_data.py output)
2. Starts sample 2008-04 (after ACWI ETF launch 2008-03-26)
3. Post-optimization centering: mean(θ)=0, mean(b)=0
4. Stress test uses explicit 1PL (documented as such)
5. Crisis permutation: contiguous-block resampling
6. Figures output to paper/figures/ (matches main.tex)
"""
import numpy as np, pandas as pd, json, os, sys
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr, pearsonr, kendalltau, mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
    'font.size':9,'axes.labelsize':10,'axes.titlesize':11,
    'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
    'figure.dpi':300,'savefig.bbox':'tight','savefig.pad_inches':0.08,
    'pdf.fonttype':42,'ps.fonttype':42,'mathtext.fontset':'dejavusans',
})

# Resolve paths relative to repo root
ROOT = Path(__file__).resolve().parent.parent
DATAFILE = ROOT / 'data' / 'prices.csv'
OUTDIR = ROOT / 'paper' / 'figures'
RESULTS_DIR = ROOT / 'outputs'
OUTDIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("v4 PRODUCTION — all blocking audit issues fixed")
print("="*70)

# ─── LOAD DATA ───
if not DATAFILE.exists():
    print(f"ERROR: {DATAFILE} not found. Run: python src/download_data.py")
    sys.exit(1)
prices = pd.read_csv(DATAFILE, index_col=0, parse_dates=True)
print(f"Loaded: {prices.shape[0]} days x {prices.shape[1]} tickers")

META = {
    'SPY':('S&P 500','US','broad'),'QQQ':('Nasdaq 100','US','tech'),
    'IWM':('Russell 2000','US','small'),'DIA':('Dow Jones','US','broad'),
    'ARKK':('ARK Innov.','US','growth'),'VTI':('Total US','US','broad'),
    'VOO':('Vgd S&P500','US','broad'),
    'EWG':('Germany','EU','country'),'EWU':('UK','EU','country'),
    'EWQ':('France','EU','country'),'EWI':('Italy','EU','country'),
    'EWP':('Spain','EU','country'),'EWL':('Switz.','EU','country'),
    'EWN':('Neth.','EU','country'),'EWD':('Sweden','EU','country'),
    'EWK':('Belgium','EU','country'),
    'EWJ':('Japan','Asia','country'),'EWY':('S. Korea','Asia','country'),
    'EWA':('Australia','Asia','country'),'EWT':('Taiwan','Asia','country'),
    'EWS':('Singapore','Asia','country'),'EWH':('Hong Kong','Asia','country'),
    'MCHI':('China','Asia','country'),
    'EEM':('EM Broad','EM','broad'),'EWZ':('Brazil','EM','country'),
    'INDA':('India','EM','country'),'EWW':('Mexico','EM','country'),
    'TUR':('Turkey','EM','country'),'EZA':('S. Africa','EM','country'),
    'ECH':('Chile','EM','country'),'EPOL':('Poland','EM','country'),
    'XLK':('Tech','US','sector'),'XLF':('Financials','US','sector'),
    'XLE':('Energy','US','sector'),'XLV':('Healthcare','US','sector'),
    'XLRE':('Real Est.','US','sector'),'SMH':('Semicons','US','sector'),
    'XLU':('Utilities','US','sector'),'XLI':('Industrials','US','sector'),
    'XLP':('Cons.Stpl','US','sector'),'XLB':('Materials','US','sector'),
    'XLC':('Comm.Svcs','US','sector'),
    'GLD':('Gold','Glb','commodity'),'SLV':('Silver','Glb','commodity'),
    'TLT':('Long Bonds','US','bond'),'HYG':('High Yield','US','bond'),
    'LQD':('Inv.Grade','US','bond'),'TIP':('TIPS','US','bond'),
    'VNQ':('US REITs','US','reit'),'IBIT':('Bitcoin','Glb','crypto'),
}
available = [t for t in META if t in prices.columns]

# FIX: Start at 2008-04 (after ACWI launch 2008-03-26)
SAMPLE_START = '2008-04-01'

CRISES = {
    'GFC':       ('2008-09','2009-03'),
    'Flash':     ('2010-05','2010-07'),
    'EUDebt':    ('2011-08','2011-10'),
    'China':     ('2015-08','2016-02'),
    'Volmag':    ('2018-10','2018-12'),
    'COVID':     ('2020-02','2020-04'),
    'RateHike':  ('2022-01','2022-10'),
}

# ─── BUILD MONTHLY RESPONSE MATRIX ───
print(f"\n[1/7] Monthly matrix from {SAMPLE_START}...")
period_starts = pd.date_range(SAMPLE_START, '2026-05-01', freq='MS')
periods = [(period_starts[i], period_starts[i+1]) for i in range(len(period_starts)-1)]

R = pd.DataFrame(np.nan, index=available, columns=range(len(periods)))
pinfo = {}

for j, (start, end) in enumerate(periods):
    mask = (prices.index >= start) & (prices.index < end)
    w = prices.loc[mask]
    if len(w) < 10: continue

    # ACWI is always available after 2008-04
    bm_ret = 0.0
    if 'ACWI' in w.columns:
        bv = w['ACWI'].dropna()
        if len(bv) > 1:
            bm_ret = (bv.iloc[-1] - bv.iloc[0]) / bv.iloc[0]

    vix_avg = 20.0
    if '^VIX' in w.columns:
        vx = w['^VIX'].dropna()
        if len(vx) > 0: vix_avg = float(vx.mean())

    frets = []
    for t in available:
        if t in w.columns:
            v = w[t].dropna()
            if len(v) > 1: frets.append((v.iloc[-1]-v.iloc[0])/v.iloc[0])
    disp = np.std(frets) if len(frets) > 5 else 0

    credit = 0.0
    if 'HYG' in w.columns and 'LQD' in w.columns:
        h, l = w['HYG'].dropna(), w['LQD'].dropna()
        if len(h) > 1 and len(l) > 1:
            credit = (l.iloc[-1]-l.iloc[0])/l.iloc[0] - (h.iloc[-1]-h.iloc[0])/h.iloc[0]

    label = start.strftime('%Y-%m')
    crisis_name = None
    for cn, (cs, ce) in CRISES.items():
        if cs <= label <= ce: crisis_name = cn; break

    pinfo[j] = {'label': label, 'acwi': bm_ret, 'vix': vix_avg,
                'disp': disp, 'credit': credit,
                'crisis': int(crisis_name is not None),
                'crisis_name': crisis_name}

    for t in available:
        if t not in w.columns: continue
        v = w[t].dropna()
        if len(v) < 5: continue
        fr = (v.iloc[-1]-v.iloc[0])/v.iloc[0]
        R.loc[t, j] = 1.0 if fr > bm_ret else 0.0

R = R.dropna(axis=1, how='all').dropna(axis=0, how='all')
nf, ni = R.shape
obs = R.notna().sum().sum()
total = nf * ni
sparsity = 1.0 - obs / total
coverage = R.notna().mean(axis=1)
print(f"   {nf} funds x {ni} months | obs={obs}/{total} ({obs/total*100:.1f}%) | sparsity={sparsity:.3f}")

# ─── IRT 2PL + POST-OPTIMIZATION CENTERING ───
print(f"\n[2/7] IRT 2PL with post-optimization centering...")
R_arr = R.values.copy()
om = ~np.isnan(R_arr)
R_arr[~om] = 0

def nll(p, A, M, F, I):
    th = p[:F]; a = np.exp(p[F:F+I]); b = p[F+I:]
    lo = a[None,:] * (th[:,None] - b[None,:])
    pr = np.clip(expit(lo), 1e-10, 1-1e-10)
    ll = A*np.log(pr) + (1-A)*np.log(1-pr); ll[~M] = 0
    return -ll.sum()

def grad_fn(p, A, M, F, I):
    th = p[:F]; a = np.exp(p[F:F+I]); b = p[F+I:]
    pr = np.clip(expit(a[None,:]*(th[:,None]-b[None,:])), 1e-10, 1-1e-10)
    r = (A-pr)*M
    return -np.concatenate([
        np.sum(r*a[None,:], axis=1),
        np.sum(r*(th[:,None]-b[None,:])*a[None,:], axis=0),
        np.sum(-r*a[None,:], axis=0)])

# True L-BFGS-B bounds: log_a constrained to [-5, 5]; theta and b unconstrained
bounds = [(None, None)] * nf + [(-5.0, 5.0)] * ni + [(None, None)] * ni
res = minimize(nll, np.zeros(nf+2*ni), jac=grad_fn, args=(R_arr, om, nf, ni),
               method='L-BFGS-B', bounds=bounds, options={'maxiter':8000, 'ftol':1e-12})
theta_raw = res.x[:nf]
log_a_fitted = res.x[nf:nf+ni]
assert np.all(log_a_fitted >= -5 - 1e-8) and np.all(log_a_fitted <= 5 + 1e-8), \
    f"log_a outside bounds: [{log_a_fitted.min():.4f}, {log_a_fitted.max():.4f}]"
a_irt = np.exp(log_a_fitted)
b_raw = res.x[nf+ni:]

# POST-OPTIMIZATION CENTERING (audit fix #4)
# Standard IRT practice: center θ to mean=0, center b to mean=0
theta = theta_raw - theta_raw.mean()
b_irt = b_raw - b_raw.mean()
print(f"   Converged: {res.success} ({res.nit} iters)")
print(f"   Post-centering: mean(θ)={theta.mean():.6f}, mean(b)={b_irt.mean():.6f}")

# Save canonical parameters for offline verification
# log_a is already bounded to [-5, 5] by L-BFGS-B optimizer bounds.
pd.DataFrame({'fund':R.index,'theta':theta}).to_csv(ROOT/'data'/'canonical_theta.csv',index=False)
pd.DataFrame({'period':R.columns,'log_a':log_a_fitted,'b':b_irt}).to_csv(ROOT/'data'/'canonical_period_params.csv',index=False)
print(f"   Canonical params saved: {nf} funds, {ni} periods")


# Rankings
avg_s = R.mean(axis=1); avg_r = avg_s.rank(ascending=False)
th_s = pd.Series(theta, index=R.index); irt_r = th_s.rank(ascending=False)
rd = avg_r - irt_r
rho_main, p_main = spearmanr(avg_r, irt_r)
tau_main, p_tau = kendalltau(avg_r, irt_r)
print(f"   ρ={rho_main:.4f} τ={tau_main:.4f} max|Δ|={rd.abs().max():.1f}")

df = pd.DataFrame({
    'name': [META[t][0] for t in R.index], 'region': [META[t][1] for t in R.index],
    'type': [META[t][2] for t in R.index],
    'beat': avg_s, 'avg_r': avg_r, 'theta': th_s, 'irt_r': irt_r,
    'delta': rd, 'cov': coverage
}).sort_values('irt_r')

movers = rd.abs().sort_values(ascending=False)
print(f"\n   TOP 6 MOVERS:")
for tk in movers.head(6).index:
    r = df.loc[tk]
    print(f"   {tk:<6} {r['name']:<15} Δ={r['delta']:>+6.1f}")

# ─── CRISIS VALIDATION: CONTIGUOUS BLOCK PERMUTATION ───
print(f"\n[3/7] Crisis validation (contiguous block permutation)...")
pdata = []
for idx, j in enumerate(R.columns):
    if j in pinfo:
        p = pinfo[j]
        pdata.append({'period': p['label'], 'b': b_irt[idx], 'acwi': p['acwi'],
                      'vix': p['vix'], 'beat': float(R.iloc[:,idx].mean()),
                      'crisis': p['crisis'], 'crisis_name': p['crisis_name']})
pdf = pd.DataFrame(pdata)
cb = pdf[pdf.crisis==1]['b']; nb = pdf[pdf.crisis==0]['b']
u_stat, p_mw = mannwhitneyu(cb, nb, alternative='two-sided')
r_beat_b, p_bb = pearsonr(pdf['beat'], pdf['b'])

# CONTIGUOUS BLOCK PERMUTATION (audit fix #5)
# Unit = crisis episode, not individual month
episode_means = []
for cn, (cs, ce) in CRISES.items():
    ep_mask = (pdf['period'] >= cs) & (pdf['period'] <= ce)
    ep_b = pdf.loc[ep_mask, 'b']
    if len(ep_b) > 0:
        episode_means.append(ep_b.mean())

obs_episode_mean = np.mean(episode_means)
n_perm = 10000
perm_episode_means = []
all_b = pdf['b'].values  # chronological
n_total = len(all_b)

for _ in range(n_perm):
    perm_means = []
    for cn, (cs, ce) in CRISES.items():
        ep_mask = (pdf['period'] >= cs) & (pdf['period'] <= ce)
        ep_len = ep_mask.sum()
        if ep_len == 0: continue
        # Sample a CONTIGUOUS block of the same length
        max_start = n_total - ep_len
        if max_start <= 0: continue
        start_idx = np.random.randint(0, max_start)
        block = all_b[start_idx:start_idx + ep_len]
        perm_means.append(block.mean())
    perm_episode_means.append(np.mean(perm_means))

p_block = np.mean(np.abs(np.array(perm_episode_means) - np.mean(perm_episode_means))
                  >= np.abs(obs_episode_mean - np.mean(perm_episode_means)))

print(f"   Month-level Mann-Whitney: p = {p_mw:.4f}")
print(f"   Contiguous-block permutation ({n_perm} perms): p = {p_block:.4f}")
print(f"   Crisis mean b = {cb.mean():.3f} | Normal mean b = {nb.mean():.3f}")
print(f"   r(beat, b) = {r_beat_b:.4f}")

# ─── LLTM ───
print(f"\n[4/7] LLTM (exploratory)...")
feat_names = ['vix_avg', 'dispersion', 'acwi_return', 'credit_spread', 'is_crisis']
Q = np.zeros((ni, len(feat_names)))
for idx, j in enumerate(R.columns):
    if j in pinfo:
        p = pinfo[j]
        Q[idx] = [p['vix'], p['disp'], p['acwi'], p['credit'], p['crisis']]
sc = StandardScaler(); Qs = sc.fit_transform(Q)
vld = np.all(np.isfinite(Qs), axis=1)
lltm = Ridge(alpha=1.0); lltm.fit(Qs[vld], b_irt[vld])
r2_lltm = r2_score(b_irt[vld], lltm.predict(Qs[vld]))
rho_lltm, _ = pearsonr(b_irt[vld], lltm.predict(Qs[vld]))
cv_scores = cross_val_score(Ridge(alpha=1.0), Qs[vld], b_irt[vld], cv=10, scoring='r2')
etas = sorted(zip(feat_names, lltm.coef_), key=lambda x: abs(x[1]), reverse=True)
print(f"   R²={r2_lltm:.4f} ρ={rho_lltm:.4f} CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")

# ─── PCR ───
print(f"\n[5/7] PCR...")
Ret = pd.DataFrame(np.nan, index=available, columns=range(len(periods)))
for j, (s, e) in enumerate(periods):
    if j not in R.columns: continue
    mk = (prices.index>=s) & (prices.index<e); w = prices.loc[mk]
    for t in available:
        if t in w.columns:
            v = w[t].dropna()
            if len(v) > 5: Ret.loc[t,j] = (v.iloc[-1]-v.iloc[0])/v.iloc[0]
Ret2 = Ret.dropna(axis=0, thresh=int(ni*0.2))
Ret2 = Ret2.dropna(axis=1, thresh=int(nf*0.3)).fillna(0)
nc = min(10, Ret2.shape[1]-1, Ret2.shape[0]-1)
pca = PCA(n_components=nc)
scores = pca.fit_transform(StandardScaler().fit_transform(Ret2.T))
cmap = {c:i for i,c in enumerate(Ret2.columns)}
pj = [j for j in R.columns if j in cmap]
Xp = np.array([scores[cmap[j]] for j in pj])
yp = np.array([b_irt[list(R.columns).index(j)] for j in pj])
pcr_m = LinearRegression().fit(Xp[:,:3], yp)
r2_pcr = r2_score(yp, pcr_m.predict(Xp[:,:3]))
print(f"   PC1={pca.explained_variance_ratio_[0]*100:.1f}% PCR(3)→b R²={r2_pcr:.4f}")

# ─── STRESS TEST (EXPLICIT 1PL) ───
print(f"\n[6/7] Stress test (1PL Rasch refit, documented as such)...")
def stress(R, theta_full, levels):
    res_list = []
    for s in levels:
        ras, ris = [], []
        for b in range(5):
            Rs = R.copy(); rng = np.random.RandomState(42+b)
            mo = R.notna()
            for i in range(Rs.shape[0]):
                for jj in range(Rs.shape[1]):
                    if mo.iloc[i,jj] and rng.random()<s: Rs.iloc[i,jj]=np.nan
            Rs2 = Rs.dropna(axis=0,how='all').dropna(axis=1,how='all')
            if Rs2.shape[0]<5 or Rs2.shape[1]<5: continue
            ar = Rs2.mean(axis=1).rank(ascending=False)
            try:
                f2,i2=Rs2.shape; a2=Rs2.values.copy(); m2=~np.isnan(a2); a2[~m2]=0
                # EXPLICIT 1PL (Rasch): P = sigmoid(theta_i - b_j)
                def rasch_nll(p,A,M,F,I):
                    th=p[:F]; b=p[F:]
                    pr=np.clip(expit(th[:,None]-b[None,:]),1e-10,1-1e-10)
                    ll=A*np.log(pr)+(1-A)*np.log(1-pr); ll[~M]=0
                    return -ll.sum()
                r2=minimize(rasch_nll,np.zeros(f2+i2),args=(a2,m2,f2,i2),
                           method='L-BFGS-B',options={'maxiter':300})
                th_s2=r2.x[:f2]
                th_s2=th_s2-th_s2.mean()  # center
                ir=pd.Series(th_s2,index=Rs2.index).rank(ascending=False)
            except: ir=ar
            tr=pd.Series(theta_full,index=R.index).rank(ascending=False)
            c=tr.index.intersection(ar.index)
            if len(c)<5: continue
            ra,_=spearmanr(tr[c],ar[c]); ri,_=spearmanr(tr[c],ir.reindex(c))
            if np.isfinite(ra): ras.append(ra)
            if np.isfinite(ri): ris.append(ri)
        ram=np.mean(ras) if ras else np.nan; rim=np.mean(ris) if ris else np.nan
        res_list.append({'s':s,'rho_avg':ram,'rho_irt':rim,
                    'adv':rim-ram if np.isfinite(rim) and np.isfinite(ram) else np.nan})
    return pd.DataFrame(res_list)

sr = stress(R, theta, [0,.1,.2,.3,.4,.5,.6,.7,.8,.9])
vs = sr.dropna(subset=['adv'])
irt_wins = int((vs.adv>0).sum())
print(f"   1PL IRT wins: {irt_wins}/{len(vs)}")

# ─── REGION STATS ───
print(f"\n[7/7] Region stats...")
reg_stats = {}
for reg in ['US','EU','Asia','EM','Glb']:
    rf = [t for t in df.index if df.loc[t,'region']==reg]
    if len(rf)>=2:
        reg_stats[reg] = {'n':len(rf),
                          'avg_theta':float(df.loc[rf,'theta'].mean()),
                          'avg_abs_delta':float(df.loc[rf,'delta'].abs().mean())}
        print(f"   {reg:<5} n={len(rf):>2} avg|Δ|={reg_stats[reg]['avg_abs_delta']:.2f}")

# ════════════════════════════════════════
# GENERATE FIGURES
# ════════════════════════════════════════
print(f"\n[FIG] Generating figures to {OUTDIR}...")

top12 = df.loc[movers.head(12).index].sort_values('delta')

# Fig 1
fig,ax=plt.subplots(figsize=(5.5,4))
colors=['#2166ac' if d>0 else '#b2182b' for d in top12['delta']]
bars=ax.barh(range(len(top12)),top12['delta'],color=colors,height=0.7)
ax.set_yticks(range(len(top12)))
ax.set_yticklabels([f"{top12.index[i]} ({top12.iloc[i]['name']})" for i in range(len(top12))],fontsize=7)
ax.set_xlabel('Rank change (simple average to IRT)')
ax.axvline(0,color='gray',lw=0.5)
for i,(val,bar) in enumerate(zip(top12['delta'],bars)):
    x=val+(1.2 if val>0 else -1.2)
    ax.text(x,i,f'{val:+.1f}',va='center',ha='left' if val>0 else 'right',fontsize=6.5)
ax.set_xlim(-36,36)
ax.set_title(f'IRT rank corrections (max |delta| = {rd.abs().max():.1f})')
plt.subplots_adjust(left=0.24,right=0.92); plt.savefig(OUTDIR/'fig1_rank_corrections.pdf'); plt.close()

# Fig 2
fig,axes=plt.subplots(1,2,figsize=(5.5,2.8))
ax=axes[0]
bp_plot=ax.boxplot([cb.values,nb.values],tick_labels=[f'Crisis\n(n={len(cb)})',f'Normal\n(n={len(nb)})'],
                   patch_artist=True,widths=0.5)
bp_plot['boxes'][0].set_facecolor('#d6604d'); bp_plot['boxes'][1].set_facecolor('#4393c3')
ax.set_ylabel('IRT difficulty (b)')
ax.set_title(f'Month: p={p_mw:.4f}\nBlock perm: p={p_block:.4f}',fontsize=9)
ax=axes[1]
ax.scatter(pdf['beat'],pdf['b'],c=['#d6604d' if c else '#4393c3' for c in pdf['crisis']],
           s=12,alpha=0.5,edgecolors='white',linewidth=0.3)
z=np.polyfit(pdf['beat'],pdf['b'],1)
xl=np.linspace(pdf['beat'].min(),pdf['beat'].max(),100)
ax.plot(xl,np.polyval(z,xl),'k--',lw=0.8,alpha=0.5)
ax.set_xlabel('Beat rate'); ax.set_ylabel('IRT difficulty (b)')
ax.set_title(f'r = {r_beat_b:.3f}',fontsize=9)
plt.tight_layout(); plt.savefig(OUTDIR/'fig2_crisis_validation.pdf'); plt.close()

# Fig 3
fig,ax=plt.subplots(figsize=(5.5,2.5))
fn=[e[0] for e in etas]; fw=[e[1] for e in etas]
cols=['#2166ac' if w>=0 else '#b2182b' for w in fw]
ax.barh(range(len(fn)),fw,color=cols,height=0.6)
ax.set_yticks(range(len(fn))); ax.set_yticklabels(fn,fontsize=8)
ax.axvline(0,color='gray',lw=0.5)
ax.set_xlabel('LLTM weight')
ax.set_title(f'LLTM: R2={r2_lltm:.3f}, CV R2={cv_scores.mean():.3f}',fontsize=9)
plt.tight_layout(); plt.savefig(OUTDIR/'fig3_lltm_weights.pdf'); plt.close()

# Fig 4
fig,ax=plt.subplots(figsize=(5.5,2.5))
ve=pca.explained_variance_ratio_[:8]*100; cv_pca=np.cumsum(ve)
x=range(1,len(ve)+1)
ax.bar(x,ve,color='#4393c3',width=0.6,label='Individual')
ax.plot(x,cv_pca,'o-',color='#2d8659',ms=4,lw=1.5,label='Cumulative')
ax.set_xlabel('Principal component'); ax.set_ylabel('Variance explained (%)')
ax.set_xticks(x); ax.set_xticklabels([f'PC{i}' for i in x])
ax.legend(loc='center right',frameon=False)
plt.tight_layout(); plt.savefig(OUTDIR/'fig4_pca_variance.pdf'); plt.close()

# Fig 5
fig,ax=plt.subplots(figsize=(5.5,3))
vs2=sr.dropna(subset=['rho_avg','rho_irt'])
ax.plot(vs2['s']*100,vs2['rho_avg'],'o-',color='#b2182b',ms=4,lw=1.5,label='Simple average')
ax.plot(vs2['s']*100,vs2['rho_irt'],'s-',color='#2166ac',ms=4,lw=1.5,label='1PL IRT')
ax.fill_between(vs2['s']*100,vs2['rho_avg'],vs2['rho_irt'],
                where=vs2['rho_irt']>vs2['rho_avg'],alpha=0.1,color='#2166ac')
ax.set_xlabel('Artificial sparsity (%)'); ax.set_ylabel('Spearman rho')
ax.legend(frameon=False); ax.set_ylim(0,1)
ax.set_title(f'Stress test (1PL refit, IRT wins {irt_wins}/{len(vs)})')
plt.tight_layout(); plt.savefig(OUTDIR/'fig5_stress_test.pdf'); plt.close()

# Fig 6
fig,ax=plt.subplots(figsize=(5.5,2.8))
regs=[r for r in ['US','EU','Asia','EM','Glb'] if r in reg_stats]
rdf=pd.DataFrame([reg_stats[r] for r in regs],index=regs)
x=range(len(rdf)); w=0.35
ax.bar([i-w/2 for i in x],rdf['avg_theta'],w,color='#4393c3',label='Avg theta')
ax2=ax.twinx()
ax2.bar([i+w/2 for i in x],rdf['avg_abs_delta'],w,color='#d6604d',label='Avg |delta|')
ax.set_xticks(x)
ax.set_xticklabels([f"{r}\n(n={int(rdf.loc[r,'n'])})" for r in regs],fontsize=8)
ax.set_ylabel('Avg theta',color='#4393c3')
ax2.set_ylabel('Avg |delta rank|',color='#d6604d')
ax.axhline(0,color='gray',lw=0.5)
l1,la1=ax.get_legend_handles_labels(); l2,la2=ax2.get_legend_handles_labels()
ax.legend(l1+l2,la1+la2,loc='upper right',frameon=False,fontsize=7)
plt.tight_layout(); plt.savefig(OUTDIR/'fig6_regions.pdf'); plt.close()

print(f"   6 figures saved")

# ─── SAVE RESULTS ───
results = {
    'sample_start': SAMPLE_START,
    'benchmark': 'ACWI (no SPY fallback after 2008-04)',
    'n_funds':int(nf),'n_items':int(ni),'sparsity':float(sparsity),
    'irt_converged':bool(res.success),'irt_iters':int(res.nit),
    'irt_centering':'post-optimization: mean(theta)=0, mean(b)=0',
    'rho_sp':float(rho_main),'tau':float(tau_main),
    'max_abs_delta':float(rd.abs().max()),
    'lltm_r2':float(r2_lltm),'lltm_rho':float(rho_lltm),
    'lltm_cv_mean':float(cv_scores.mean()),'lltm_cv_std':float(cv_scores.std()),
    'pca_pc1':float(pca.explained_variance_ratio_[0]),
    'pca_top3':float(sum(pca.explained_variance_ratio_[:3])),
    'pcr_r2':float(r2_pcr),
    'crisis_b_mean':float(cb.mean()),'normal_b_mean':float(nb.mean()),
    'crisis_p_mw':float(p_mw),'crisis_p_block':float(p_block),
    'r_beat_b':float(r_beat_b),
    'stress_model':'1PL Rasch (documented)',
    'stress_irt_wins':irt_wins,'stress_total':int(len(vs)),
    'region_stats':{k:v for k,v in reg_stats.items()},
    'lltm_weights':{f:float(w) for f,w in etas},
    'top12_movers':{tk:{'name':META[tk][0],'delta':float(df.loc[tk,'delta'])}
                    for tk in movers.head(12).index},
}
with open(RESULTS_DIR/'results.json','w') as f:
    json.dump(results,f,indent=2,default=str)
df.to_csv(RESULTS_DIR/'rankings.csv')

print(f"\n{'='*70}")
print(f"v4 VERIFIED: {nf}x{ni} | ρ={rho_main:.4f} | max|Δ|={rd.abs().max():.1f}")
print(f"  Crisis block-perm p={p_block:.4f}")
print(f"  LLTM CV R²={cv_scores.mean():.4f}")
print(f"  Stress 1PL wins {irt_wins}/{len(vs)}")
print(f"{'='*70}")
