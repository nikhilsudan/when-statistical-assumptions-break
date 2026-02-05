import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.mixture import GaussianMixture

from distributions import (
    generate_lognormal,
    generate_student_t,
    generate_mixture,
)
from confidence_intervals import standard_95_ci

SAMPLE_SIZES = [50, 200, 500, 1000, 5000]
N_REPLICATES = 800   # lower than before to keep runtime reasonable


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def lognormal_coverage_curves():
    """Return (n, original_coverage, logspace_coverage)."""

    orig = []
    fixed = []

    for n in SAMPLE_SIZES:
        covers_orig = []
        covers_fixed = []

        for i in range(N_REPLICATES):
            x, true_mean, _ = generate_lognormal(n, seed=i)

            # ---- Original CI ----
            lo, hi = standard_95_ci(x)
            covers_orig.append(lo <= true_mean <= hi)

            # ---- Log-space CI ----
            logx = np.log(x)
            mu = np.mean(logx)
            s = np.std(logx, ddof=1)
            z = 1.96
            m = z * s / np.sqrt(n)

            lo_log, hi_log = mu - m, mu + m
            lo2, hi2 = np.exp(lo_log), np.exp(hi_log)

            covers_fixed.append(lo2 <= true_mean <= hi2)

        orig.append(np.mean(covers_orig))
        fixed.append(np.mean(covers_fixed))

    return np.array(SAMPLE_SIZES), np.array(orig), np.array(fixed)


def studentt_robust_curves():
    """Return (n, mean_cov, median_cov, trimmed_cov)."""

    def bootstrap_median_ci(samples, B=600):
        n = len(samples)
        meds = [
            np.median(np.random.choice(samples, size=n, replace=True))
            for _ in range(B)
        ]
        return np.percentile(meds, 2.5), np.percentile(meds, 97.5)

    def trimmed_ci(samples, trim=0.1):
        tmean = stats.trim_mean(samples, proportiontocut=trim)
        se = stats.sem(samples)
        z = 1.96
        return tmean - z * se, tmean + z * se

    mean_cov, med_cov, trim_cov = [], [], []

    for n in SAMPLE_SIZES:
        m, md, t = [], [], []

        for i in range(N_REPLICATES):
            x, true_mean, _ = generate_student_t(n, seed=i)

            lo, hi = standard_95_ci(x)
            m.append(lo <= true_mean <= hi)

            lo2, hi2 = bootstrap_median_ci(x)
            md.append(lo2 <= true_mean <= hi2)

            lo3, hi3 = trimmed_ci(x)
            t.append(lo3 <= true_mean <= hi3)

        mean_cov.append(np.mean(m))
        med_cov.append(np.mean(md))
        trim_cov.append(np.mean(t))

    return (
        np.array(SAMPLE_SIZES),
        np.array(mean_cov),
        np.array(med_cov),
        np.array(trim_cov),
    )


def mixture_before_after_points(n=1200, seed=0):
    """Return single mean + two GMM means + CIs for each cluster."""

    x, true_mean, _ = generate_mixture(n, seed=seed)

    single_mean = np.mean(x)
    single_lo, single_hi = standard_95_ci(x)

    gmm = GaussianMixture(n_components=2, random_state=seed)
    labels = gmm.fit_predict(x.reshape(-1, 1))

    means, cis = [], []

    for k in range(2):
        grp = x[labels == k]
        mu = np.mean(grp)
        lo, hi = standard_95_ci(grp)
        means.append(mu)
        cis.append((lo, hi))

    return single_mean, (single_lo, single_hi), means, cis


# ============================================================
# SINGLE HIGH-GRADE FIGURE
# ============================================================

def plot_three_panel_remediation():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # -------------------------------------------------------
    # PANEL A — LOGNORMAL FIX
    # -------------------------------------------------------
    n, orig, fixed = lognormal_coverage_curves()

    ax = axes[0]
    ax.plot(n, orig, marker="o", label="Classical CI (bad)")
    ax.plot(n, fixed, marker="o", label="Log-space CI (fixed)")
    ax.axhline(0.95, linestyle="--", label="Nominal 95%")

    ax.set_xscale("log")
    ax.set_ylim(0.85, 1.02)
    ax.set_xlabel("Sample size (log scale)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(
        "A) Skewed data → Log transform restores valid CIs"
    )
    ax.legend()

    # -------------------------------------------------------
    # PANEL B — STUDENT-t ROBUST FIX
    # -------------------------------------------------------
    n, mean_cov, med_cov, trim_cov = studentt_robust_curves()

    ax = axes[1]
    ax.plot(n, mean_cov, marker="o", label="Mean CI (fragile)")
    ax.plot(n, med_cov, marker="o", label="Median CI (robust)")
    ax.plot(n, trim_cov, marker="o", label="Trimmed mean CI")
    ax.axhline(0.95, linestyle="--")

    ax.set_xscale("log")
    ax.set_ylim(0.85, 1.02)
    ax.set_xlabel("Sample size (log scale)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(
        "B) Heavy tails → Robust estimators outperform the mean"
    )
    ax.legend()

    # -------------------------------------------------------
    # PANEL C — MIXTURE + GMM
    # -------------------------------------------------------
    single_mean, (s_lo, s_hi), means, cis = mixture_before_after_points()

    ax = axes[2]

    # plot population scatter as faint background
    x, _, _ = generate_mixture(2000, seed=1)
    ax.hist(x, bins=60, density=True, alpha=0.2, color="gray")

    # single mean + CI
    ax.axvline(single_mean, linestyle="--", color="black", label="Single mean")
    ax.plot([s_lo, s_hi], [0.01, 0.01], color="black")

    # two cluster means + CIs
    colors = ["tab:blue", "tab:orange"]
    for i, (mu, (lo, hi)) in enumerate(zip(means, cis)):
        ax.axvline(mu, color=colors[i], label=f"Cluster {i+1} mean")
        ax.plot([lo, hi], [0.015 + 0.002 * i, 0.015 + 0.002 * i], color=colors[i])

    ax.set_title(
        "C) Mixture → GMM reveals two real populations"
    )
    ax.set_xlabel("Value")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("outputs/remediation_three_panel.png")
    plt.close()


def main():
    print("Generating single three-panel remediation figure...")
    plot_three_panel_remediation()
    print("Saved: outputs/remediation_three_panel.png")


if __name__ == "__main__":
    main()
