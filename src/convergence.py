import numpy as np
import pandas as pd

from distributions import (
    generate_normal,
    generate_lognormal,
    generate_student_t,
    generate_mixture,
)

SAMPLE_SIZES = [50, 200, 500, 1000, 5000]


def track_mean_convergence(generator_fn, true_mean, seed=0):
    """
    For a single distribution, track how the sample mean evolves
    as sample size increases.

    Returns:
        pandas DataFrame with columns:
        ['n', 'sample_mean', 'true_mean', 'absolute_error']
    """
    records = []

    for n in SAMPLE_SIZES:
        samples, _, _ = generator_fn(n, seed=seed)

        sample_mean = float(np.mean(samples))
        abs_error = abs(sample_mean - true_mean)

        records.append({
            "n": n,
            "sample_mean": sample_mean,
            "true_mean": true_mean,
            "absolute_error": abs_error,
        })

    return pd.DataFrame.from_records(records)


def run_convergence_experiment():
    """
    Run convergence tracking for all four distributions.

    Returns:
        pandas DataFrame with a column 'distribution' added.
    """
    frames = []

    # ---- Normal ----
    df_norm = track_mean_convergence(
        generate_normal, true_mean=0.0, seed=0
    )
    df_norm["distribution"] = "normal"
    frames.append(df_norm)

    # ---- Lognormal ----
    df_lognorm = track_mean_convergence(
        generate_lognormal, true_mean=np.exp(0.5), seed=0
    )
    df_lognorm["distribution"] = "lognormal"
    frames.append(df_lognorm)

    # ---- Student-t ----
    df_t = track_mean_convergence(
        generate_student_t, true_mean=0.0, seed=0
    )
    df_t["distribution"] = "student_t"
    frames.append(df_t)

    # ---- Mixture ----
    df_mix = track_mean_convergence(
        generate_mixture, true_mean=0.0, seed=0
    )
    df_mix["distribution"] = "mixture"
    frames.append(df_mix)

    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = run_convergence_experiment()
    print(df)
