import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from distributions import (
    generate_normal,
    generate_lognormal,
    generate_student_t,
    generate_mixture,
)

from estimation import run_estimation_experiment
from confidence_intervals import run_coverage_experiment
from convergence import run_convergence_experiment
from hypothesis_testing import run_testing_experiment

# Make figures clean and publication-style
plt.rcParams.update({
    "figure.dpi": 180,
    "font.size": 10,
    "axes.grid": True,
})


# -------------------------------------------------------------
# FIGURE 1 — Distribution panels (population vs sampling mean)
# -------------------------------------------------------------

def plot_distribution_panels():
    """
    For each distribution:
    Left: population density with true vs sample mean marked
    Right: sampling distribution of the mean

    Saves: outputs/distribution_panels.png
    """

    np.random.seed(0)
    n = 200
    reps = 5000

    fig, axes = plt.subplots(4, 2, figsize=(10, 14))

    generators = {
        "Normal": generate_normal,
        "Lognormal": generate_lognormal,
        "Student-t": generate_student_t,
        "Mixture": generate_mixture,
    }

    for i, (name, gen) in enumerate(generators.items()):

        # ----- Left panel: population shape -----
        x, true_mean, _ = gen(5000)

        kde = gaussian_kde(x)
        xs = np.linspace(min(x), max(x), 400)

        ax = axes[i, 0]
        ax.plot(xs, kde(xs))
        ax.axvline(true_mean, linestyle="--", label="True mean")

        sample_mean = np.mean(x[:n])
        ax.axvline(sample_mean, linestyle=":", label="Sample mean")

        ax.set_title(f"{name}: Population distribution")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        if i == 0:
            ax.legend()

        # ----- Right panel: sampling distribution of the mean -----
        means = []
        for r in range(reps):
            s, _, _ = gen(n, seed=r)
            means.append(np.mean(s))

        ax2 = axes[i, 1]
        ax2.hist(means, bins=60, density=True)
        ax2.set_title(f"{name}: Sampling distribution of the mean (n={n})")
        ax2.set_xlabel("Sample mean")
        ax2.set_ylabel("Density")

    plt.tight_layout()
    plt.savefig("outputs/distribution_panels.png")
    plt.close()


# -------------------------------------------------------------
# FIGURE 2 — CI coverage + width (two-panel research plot)
# -------------------------------------------------------------

def plot_ci_coverage_with_width():
    """
    Two-panel figure:
    Top: empirical coverage vs n
    Bottom: average CI width vs n

    Saves: outputs/ci_coverage_research.png
    """

    df = run_coverage_experiment()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

    for dist in df["distribution"].unique():
        sub = df[df["distribution"] == dist]
        ax1.plot(sub["n"], sub["coverage"], marker="o", label=dist)
        ax2.plot(sub["n"], sub["avg_ci_width"], marker="o")

    ax1.axhline(0.95, linestyle="--", label="Nominal 95%")
    ax1.set_xscale("log")
    ax1.set_ylabel("Empirical coverage")
    ax1.set_title("Confidence Interval Reliability")
    ax1.legend()

    ax2.set_xscale("log")
    ax2.set_xlabel("Sample size (log scale)")
    ax2.set_ylabel("Average CI width")
    ax2.set_title("Precision of uncertainty estimates")

    plt.tight_layout()
    plt.savefig("outputs/ci_coverage_research.png")
    plt.close()


# -------------------------------------------------------------
# FIGURE 3 — Absolute error convergence (better than raw means)
# -------------------------------------------------------------

def plot_absolute_error_convergence():
    """
    Plots absolute error from true mean vs sample size.

    Saves: outputs/mean_error_convergence.png
    """

    df = run_convergence_experiment()

    plt.figure(figsize=(8, 6))

    for dist in df["distribution"].unique():
        sub = df[df["distribution"] == dist]
        plt.plot(
            sub["n"],
            sub["absolute_error"],
            marker="o",
            label=dist,
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Sample size (log scale)")
    plt.ylabel("Absolute error from true mean (log scale)")
    plt.title("Convergence of the sample mean")
    plt.legend()
    plt.tight_layout()

    plt.savefig("outputs/mean_error_convergence.png")
    plt.close()


# -------------------------------------------------------------
# FIGURE 4 — t-test miscalibration (centered at zero)
# -------------------------------------------------------------

def plot_ttest_miscalibration():
    """
    Plots (empirical Type I error - 0.05).

    Zero = perfectly calibrated test.

    Saves: outputs/ttest_miscalibration.png
    """

    df = run_testing_experiment()
    df["miscalibration"] = df["type_i_error"] - 0.05

    plt.figure(figsize=(8, 6))

    for dist in df["distribution"].unique():
        sub = df[df["distribution"] == dist]
        plt.plot(
            sub["n"],
            sub["miscalibration"],
            marker="o",
            label=dist,
        )

    plt.axhline(0.0, linestyle="--")
    plt.xscale("log")
    plt.xlabel("Sample size (log scale)")
    plt.ylabel("Type I miscalibration (error - 0.05)")
    plt.title("Mis-calibration of the classical t-test")
    plt.legend()
    plt.tight_layout()

    plt.savefig("outputs/ttest_miscalibration.png")
    plt.close()


# -------------------------------------------------------------
# FIGURE 5 — Capstone summary heatmap
# -------------------------------------------------------------

def plot_summary_heatmap():
    """
    One synthesis figure summarizing failures across distributions.

    Saves: outputs/summary_heatmap.png
    """

    coverage = run_coverage_experiment().groupby("distribution")["coverage"].mean()
    conv = run_convergence_experiment().groupby("distribution")["absolute_error"].mean()
    ttest = run_testing_experiment().groupby("distribution")["type_i_error"].mean()

    summary = pd.DataFrame({
        "CI coverage": coverage,
        "Mean error": conv,
        "t-test error": ttest,
    }).loc[
        ["normal", "lognormal", "student_t", "mixture"]
    ]

    plt.figure(figsize=(7, 6))
    im = plt.imshow(summary, aspect="auto")

    plt.xticks(range(3), summary.columns, rotation=20)
    plt.yticks(range(4), summary.index)

    for i in range(4):
        for j in range(3):
            plt.text(j, i, f"{summary.iloc[i, j]:.3f}",
                     ha="center", va="center")

    plt.title("When statistical assumptions break: summary")
    plt.colorbar(im, label="Performance metric")
    plt.tight_layout()

    plt.savefig("outputs/summary_heatmap.png")
    plt.close()


# -------------------------------------------------------------
# MASTER RUNNER
# -------------------------------------------------------------

def main():
    print("Generating research-grade visuals...")

    plot_distribution_panels()
    print("Saved: outputs/distribution_panels.png")

    plot_ci_coverage_with_width()
    print("Saved: outputs/ci_coverage_research.png")

    plot_absolute_error_convergence()
    print("Saved: outputs/mean_error_convergence.png")

    plot_ttest_miscalibration()
    print("Saved: outputs/ttest_miscalibration.png")

    plot_summary_heatmap()
    print("Saved: outputs/summary_heatmap.png")

    print("All high-grade visuals generated.")


if __name__ == "__main__":
    main()
