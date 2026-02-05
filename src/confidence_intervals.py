import numpy as np
import pandas as pd
from scipy import stats

from distributions import (
    generate_normal,
    generate_lognormal,
    generate_student_t,
    generate_mixture,
)


SAMPLE_SIZES = [50, 200, 500, 1000, 5000]
N_REPLICATES = 1000   # number of repeated experiments per setting


def standard_95_ci(samples):
    """
    Construct a classical 95% confidence interval for the mean
    using the normal approximation.

    Returns: (lower_bound, upper_bound)
    """
    n = len(samples)
    sample_mean = np.mean(samples)
    sample_std = np.std(samples, ddof=1)

    z = 1.96  # 95% critical value

    margin = z * sample_std / np.sqrt(n)
    return sample_mean - margin, sample_mean + margin


def coverage_experiment(generator_fn, true_mean, n, seed_start=0):
    """
    Run repeated sampling to estimate empirical CI coverage.

    Args:
        generator_fn: function that generates samples
        true_mean: true population mean
        n: sample size
        seed_start: starting random seed for reproducibility

    Returns:
        coverage_rate: fraction of intervals that contain true_mean
        avg_width: average width of the CI
    """
    covers = []
    widths = []

    for i in range(N_REPLICATES):
        samples, _, _ = generator_fn(n, seed=seed_start + i)
        lower, upper = standard_95_ci(samples)

        covers.append(lower <= true_mean <= upper)
        widths.append(upper - lower)

    coverage_rate = float(np.mean(covers))
    avg_width = float(np.mean(widths))

    return coverage_rate, avg_width


def run_coverage_experiment():
    """
    Compute empirical 95% CI coverage for each distribution
    and each sample size.

    Returns:
        pandas DataFrame with:
        ['distribution', 'n', 'coverage', 'avg_ci_width']
    """
    records = []

    for n in SAMPLE_SIZES:

        # ---- Normal ----
        coverage, width = coverage_experiment(
            generate_normal, true_mean=0.0, n=n
        )
        records.append({
            "distribution": "normal",
            "n": n,
            "coverage": coverage,
            "avg_ci_width": width,
        })

        # ---- Lognormal ----
        coverage, width = coverage_experiment(
            generate_lognormal, true_mean=np.exp(0.5), n=n
        )
        records.append({
            "distribution": "lognormal",
            "n": n,
            "coverage": coverage,
            "avg_ci_width": width,
        })

        # ---- Student-t ----
        coverage, width = coverage_experiment(
            generate_student_t, true_mean=0.0, n=n
        )
        records.append({
            "distribution": "student_t",
            "n": n,
            "coverage": coverage,
            "avg_ci_width": width,
        })

        # ---- Mixture ----
        coverage, width = coverage_experiment(
            generate_mixture, true_mean=0.0, n=n
        )
        records.append({
            "distribution": "mixture",
            "n": n,
            "coverage": coverage,
            "avg_ci_width": width,
        })

    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    df = run_coverage_experiment()
    print(df)
