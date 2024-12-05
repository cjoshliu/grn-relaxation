# bootstrap_argmax.py: find peak location CI using bootstrap

import jax
from jax import numpy as jnp

def bootstrap_argmax(data: jnp.ndarray,
                     num_bootstrap: int = 10000,
                     ci: float = 95,
                     seed: int = 0):
    """
    Compute the confidence interval for the argmax of a 1D array using bootstrapping with JAX.

    Parameters
    ----------
    data : jnp.ndarray
        1D array of data points.

    num_bootstrap : int
        Number of bootstrap samples.

    ci : float
        Confidence interval percentage (e.g., 95 for 95% CI).

    seed : int
        Seed for reproducibility.

    Returns
    -------
    argmax_original : int
        Argmax index of the original data.

    ci_lower : float
        Lower bound of the confidence interval for the argmax index.

    ci_upper : float
        Upper bound of the confidence interval for the argmax index.
    """
    key = jax.random.PRNGKey(seed)
    n = data.shape[0]

    # Generate all random keys for bootstrapping
    keys = jax.random.split(key, num_bootstrap)

    # Define a function to perform bootstrapping
    def bootstrap_sample(key):
        # Generate bootstrap indices
        indices = jax.random.randint(key, shape=(n,), minval=0, maxval=n)
        # Resample data
        resampled_data = data[jnp.sort(indices)]
        # Compute argmax
        return jnp.argmax(resampled_data)

    # Vectorize the bootstrap sampling
    bootstrap_argmax_indices = jax.vmap(bootstrap_sample)(keys)

    # Argmax of the original data
    argmax_original = int(jnp.argmax(data))

    # # Convert to NumPy for percentile calculation
    # bootstrap_argmax_indices_np = np.array(bootstrap_argmax_indices)

    # Compute confidence interval
    alpha = 100 - ci
    lower_percentile = alpha / 2
    upper_percentile = 100 - (alpha / 2)

    ci_lower = jnp.percentile(bootstrap_argmax_indices, lower_percentile)
    ci_upper = jnp.percentile(bootstrap_argmax_indices, upper_percentile)

    return argmax_original, int(jnp.round(ci_lower)), int(jnp.round(ci_upper))+1
