"""N03: paired statistics for every reported difference (answers R2.3 / R3.4).

For two systems A and B evaluated on the SAME rows:
  * dF1 = F1(A) - F1(B) and the difference in actual error counts;
  * 95% CI from a paired, class-stratified bootstrap (10,000 resamples; rows are resampled within each true
    class, and the SAME resampled rows are used for A and B and for every comparison);
  * exact two-sided McNemar test on the discordant rows (b = A right & B wrong, c = A wrong & B right), reported
    for one seed only; it is not the test behind the Holm-adjusted p-values.
For seed-matched comparisons (training seeds or meta-seeds) the CI also carries the seed spread (compare_seeds,
D20/D25): the p-value comes from the Student-t statistic with Welch-Satterthwaite df, and n06 Holm-adjusts these
t-based p-values within the pre-specified family. A primary difference is resolved only if the Holm-adjusted p < 0.05
and all per-seed differences share its sign. Single-run comparisons: resolved when the 95% CI excludes 0 (D8).
Scope (D37): rows are resampled with all fitted models fixed, so the intervals are conditional on the folds, the fitted
predictions and the stated seeds; no model is refitted and no event-level dependence is resampled.

Library use: from n03_stats import Bootstrapper; bs = Bootstrapper(y); bs.compare(predA, predB) or
bs.compare_seeds([predA_s...], [predB_s...]).
"""
import numpy as np
from scipy.stats import binomtest, norm
from scipy.stats import t as student_t

K = 8


def macro_f1_from_cm(cm):
    """cm: (..., K*K) flattened confusion counts (rows = true, cols = predicted)."""
    cm = cm.reshape(*cm.shape[:-1], K, K)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    fp = cm.sum(-2) - tp
    fn = cm.sum(-1) - tp
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.maximum(d, 1e-12), 0.0).mean(-1)


class Bootstrapper:
    def __init__(self, y, n_boot=10_000, seed=20261118):
        self.y = np.asarray(y).astype(int)
        n = len(self.y)
        rng = np.random.default_rng(seed)
        W = np.zeros((n_boot, n), np.float32)
        for c in range(K):
            idx = np.where(self.y == c)[0]
            draws = rng.integers(0, len(idx), size=(n_boot, len(idx)))
            for b in range(n_boot):
                np.add.at(W[b], idx[draws[b]], 1.0)
        self.W = W
        self.n_boot = n_boot

    def boot_f1(self, pred):
        onehot = np.zeros((len(self.y), K * K), np.float32)
        onehot[np.arange(len(self.y)), self.y * K + np.asarray(pred).astype(int)] = 1.0
        return macro_f1_from_cm(self.W @ onehot)

    def point(self, pred):
        cm = np.bincount(self.y * K + np.asarray(pred).astype(int), minlength=K * K).astype(float)
        return float(macro_f1_from_cm(cm)), int((np.asarray(pred) != self.y).sum())

    def compare(self, pa, pb):
        fa, ea = self.point(pa)
        fb, eb = self.point(pb)
        d = self.boot_f1(pa) - self.boot_f1(pb)
        lo, hi = np.percentile(d, [2.5, 97.5])
        ra, rb = np.asarray(pa) == self.y, np.asarray(pb) == self.y
        b, c = int((ra & ~rb).sum()), int((~ra & rb).sum())
        p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c > 0 else 1.0
        return dict(f1_a=fa, f1_b=fb, d_f1=fa - fb, ci_lo=float(lo), ci_hi=float(hi), err_a=ea, err_b=eb,
                    d_err=ea - eb, b_a_only_right=b, c_b_only_right=c, mcnemar_p=float(p),
                    resolved=bool(lo > 0 or hi < 0))

    def compare_seeds(self, pas, pbs):
        """Seed-matched comparison (training seeds or meta-seeds), DECISION_LOG D20.
        d_s = F1(A_s) - F1(B_s) per seed; dbar = mean_s d_s.
        Var(dbar) = Var_rowboot(dbar) + s_d^2 / S   (row resampling with one shared weight matrix, plus seed spread).
        D25 (amends D20, logged before any G/P result): with a = Var_rowboot and b = s_d^2 / S, the reference
        distribution is Student t with Welch-Satterthwaite df = (a + b)^2 (S - 1) / b^2 (df = inf, i.e. normal, when
        b = 0 or S = 1). 95% CI = dbar +/- t_{0.975,df} sqrt(Var); two-sided p from t = dbar / sqrt(Var) -> key 'p'.
        The D20 normal-approximation values stay available as ci_lo_normal / ci_hi_normal / p_normal (reference only).
        resolved = CI excludes 0 AND every per-seed difference has the same sign as dbar (and is non-zero).
        The row-only percentile CI is also returned for reference. McNemar (exact) is given for the first seed only."""
        assert len(pas) == len(pbs) and len(pas) >= 1
        S = len(pas)
        per = np.array([self.point(a)[0] - self.point(b)[0] for a, b in zip(pas, pbs)])
        dboot = np.mean([self.boot_f1(a) - self.boot_f1(b) for a, b in zip(pas, pbs)], axis=0)
        var_row = float(np.var(dboot, ddof=1))
        sd_seed = float(np.std(per, ddof=1)) if S > 1 else 0.0
        b_seed = sd_seed ** 2 / S
        se = float(np.sqrt(var_row + b_seed))
        dbar = float(per.mean())
        df_satt = float((var_row + b_seed) ** 2 * (S - 1) / b_seed ** 2) if (S > 1 and b_seed > 0) else np.inf
        q = float(student_t.ppf(0.975, df_satt)) if np.isfinite(df_satt) else float(norm.ppf(0.975))
        lo, hi = dbar - q * se, dbar + q * se
        lo_n, hi_n = dbar - 1.96 * se, dbar + 1.96 * se
        if se > 0:
            p_n = float(2 * norm.sf(abs(dbar / se)))
            p = float(2 * student_t.sf(abs(dbar / se), df_satt)) if np.isfinite(df_satt) else p_n
        else:  # identical predictions in every seed and resample
            p = p_n = 1.0 if dbar == 0 else 0.0
        same_sign = bool(np.all(np.sign(per) == np.sign(dbar)) and dbar != 0)
        rlo, rhi = np.percentile(dboot, [2.5, 97.5])
        ea = [self.point(a)[1] for a in pas]
        eb = [self.point(b)[1] for b in pbs]
        m0 = self.compare(pas[0], pbs[0])
        return dict(n_seeds=S, d_f1_mean=dbar, d_f1_sd_seed=sd_seed if S > 1 else np.nan,
                    d_f1_per_seed=[round(float(x), 5) for x in per], se_row=float(np.sqrt(var_row)), se_total=se,
                    df_satt=df_satt, t_quantile=q, ci_lo=float(lo), ci_hi=float(hi), p=p,
                    ci_lo_normal=float(lo_n), ci_hi_normal=float(hi_n), p_normal=p_n,
                    ci_row_lo=float(rlo), ci_row_hi=float(rhi),
                    same_sign=same_sign, err_a=ea, err_b=eb, d_err_mean=float(np.mean(ea) - np.mean(eb)),
                    mcnemar_seed0_b=m0["b_a_only_right"], mcnemar_seed0_c=m0["c_b_only_right"],
                    mcnemar_seed0_p=m0["mcnemar_p"], resolved=bool((lo > 0 or hi < 0) and same_sign))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - r) * p[i]))
        adj[i] = running
    return adj
