import numpy as np


def generate_normal(n, seed=42):
    """
    Generate samples from a standard normal distribution.

    Returns:
        samples: numpy array of shape (n,)
        true_mean: float
        true_variance: float
    """
    rng = np.random.default_rng(seed)

    samples = rng.normal(loc=0.0, scale=1.0, size=n)
    true_mean = 0.0
    true_variance = 1.0

    return samples, true_mean, true_variance


def generate_lognormal(n, seed=42):
    """
    Generate samples from a lognormal distribution.

    We generate X ~ LogNormal(mu=0, sigma=1).

    Returns:
        samples: numpy array of shape (n,)
        true_mean: float
        true_variance: float
    """
    rng = np.random.default_rng(seed)

    mu = 0.0
    sigma = 1.0

    samples = rng.lognormal(mean=mu, sigma=sigma, size=n)

    # True mean and variance of lognormal distribution
    true_mean = np.exp(mu + (sigma**2) / 2)
    true_variance = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)

    return samples, true_mean, true_variance


def generate_student_t(n, df=4, seed=42):
    """
    Generate samples from a Student-t distribution.

    df=4 gives heavy tails while keeping finite variance.

    Returns:
        samples: numpy array of shape (n,)
        true_mean: float
        true_variance: float
    """
    rng = np.random.default_rng(seed)

    samples = rng.standard_t(df=df, size=n)

    # For Student-t:
    # Mean = 0 (if df > 1)
    # Variance = df / (df - 2) if df > 2
    true_mean = 0.0
    true_variance = df / (df - 2)

    return samples, true_mean, true_variance


def generate_mixture(n, seed=42):
    """
    Generate a bimodal Gaussian mixture:
    50% from N(-2, 1) and 50% from N(2, 1).

    Returns:
        samples: numpy array of shape (n,)
        true_mean: float
        true_variance: float
    """
    rng = np.random.default_rng(seed)

    n1 = n // 2
    n2 = n - n1

    samples1 = rng.normal(loc=-2.0, scale=1.0, size=n1)
    samples2 = rng.normal(loc=2.0, scale=1.0, size=n2)

    samples = np.concatenate([samples1, samples2])

    # True mean of symmetric mixture is 0
    true_mean = 0.0

    # True variance = variance within components + variance between components
    # Each component variance = 1
    # Means are -2 and 2, so between-component variance = 4
    true_variance = 1.0 + 4.0

    return samples, true_mean, true_variance
