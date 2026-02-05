import numpy as np
import pandas as pd

from distributions import (
    generate_normal,
    generate_lognormal,
    generate_student_t,
    generate_mixture,
)


SAMPLE_SIZES = [50, 200, 500, 1000, 5000]


def estimate_mean_and_variance(samples):
    """
    Compute sample mean and sample variance (unbiased).
    """
    sample_mean = float(np.mean(samples))
    sample_variance = float(np.var(samples, ddof=1))
    return sample_mean, sample_variance


def run_estimation_experiment():
    """
    For each distribution and each sample size:
      - draw one sample
      - compute sample mean and variance
      - compare to true values

    Returns:
        results_df: pandas DataFrame with columns:
        ['distribution', 'n', 'sample_mean', 'true_mean',
         'mean_bias', 'sample_variance', 'true_variance',
         'variance_error']
    """

    records = []

    for n in SAMPLE_SIZES:

        # ---- Normal ----
        x, true_mean, true_var = generate_normal(n)
        sm, sv = estimate_mean_and_variance(x)

        records.append({
            "distribution": "normal",
            "n": n,
            "sample_mean": sm,
            "true_mean": true_mean,
            "mean_bias": sm - true_mean,
            "sample_variance": sv,
            "true_variance": true_var,
            "variance_error": sv - true_var,
        })

        # ---- Lognormal ----
        x, true_mean, true_var = generate_lognormal(n)
        sm, sv = estimate_mean_and_variance(x)

        records.append({
            "distribution": "lognormal",
            "n": n,
            "sample_mean": sm,
            "true_mean": true_mean,
            "mean_bias": sm - true_mean,
            "sample_variance": sv,
            "true_variance": true_var,
            "variance_error": sv - true_var,
        })

        # ---- Student-t ----
        x, true_mean, true_var = generate_student_t(n)
        sm, sv = estimate_mean_and_variance(x)

        records.append({
            "distribution": "student_t",
            "n": n,
            "sample_mean": sm,
            "true_mean": true_mean,
            "mean_bias": sm - true_mean,
            "sample_variance": sv,
            "true_variance": true_var,
            "variance_error": sv - true_var,
        })

        # ---- Mixture ----
        x, true_mean, true_var = generate_mixture(n)
        sm, sv = estimate_mean_and_variance(x)

        records.append({
            "distribution": "mixture",
            "n": n,
            "sample_mean": sm,
            "true_mean": true_mean,
            "mean_bias": sm - true_mean,
            "sample_variance": sv,
            "true_variance": true_var,
            "variance_error": sv - true_var,
        })

    results_df = pd.DataFrame.from_records(records)
    return results_df


if __name__ == "__main__":
    df = run_estimation_experiment()
    print(df)
