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
N_REPLICATES = 1000      # repeated experiments per setting
ALPHA = 0.05             # 5% significance level


def run_one_sample_ttest(samples, mu0):
    """
    Classical one-sample t-test:
    H0: mean = mu0
    H1: mean != mu0

    Returns:
        p_value: float
        reject: bool (True if H0 rejected)
    """
    t_stat, p_value = stats.ttest_1samp(samples, popmean=mu0)
    reject = p_value < ALPHA
    return float(p_value), bool(reject)


def type_i_error_experiment(generator_fn, true_mean, n, seed_start=0):
    """
    Estimate empirical Type I error rate:
    - We simulate data where H0 is TRUE
    - We see how often the t-test incorrectly rejects H0

    Returns:
        type_i_rate: fraction of false rejections
    """
    rejections = []

    for i in range(N_REPLICATES):
        samples, _, _ = generator_fn(n, seed=seed_start + i)
        _, reject = run_one_sample_ttest(samples, mu0=true_mean)
        rejections.append(reject)

    return float(np.mean(rejections))


def run_testing_experiment():
    """
    For each distribution and sample size,
    estimate the empirical Type I error rate
    of the classical t-test.

    Returns:
        DataFrame with columns:
        ['distribution', 'n', 'type_i_error']
    """
    records = []

    for n in SAMPLE_SIZES:

        # ---- Normal (benchmark) ----
        err = type_i_error_experiment(
            generate_normal, true_mean=0.0, n=n
        )
        records.append({
            "distribution": "normal",
            "n": n,
            "type_i_error": err,
        })

        # ---- Lognormal ----
        err = type_i_error_experiment(
            generate_lognormal, true_mean=np.exp(0.5), n=n
        )
        records.append({
            "distribution": "lognormal",
            "n": n,
            "type_i_error": err,
        })

        # ---- Student-t ----
        err = type_i_error_experiment(
            generate_student_t, true_mean=0.0, n=n
        )
        records.append({
            "distribution": "student_t",
            "n": n,
            "type_i_error": err,
        })

        # ---- Mixture ----
        err = type_i_error_experiment(
            generate_mixture, true_mean=0.0, n=n
        )
        records.append({
            "distribution": "mixture",
            "n": n,
            "type_i_error": err,
        })

    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    df = run_testing_experiment()
    print(df)
