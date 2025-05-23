import argparse
import os

import time

import numpy as np
import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
from flax import nnx
from pathlib import Path

# Local/fojax imports
from fojax.nn import LogReg
from fojax.data import get_cifar10_dataloaders_no_tf
from fojax.sampler import (
    make_mala_step,
    make_fmala_step,
    make_line_fmala_step,
    make_precon_fmala_step,
    make_precon_line_fmala_step,
)
from functools import partial

# CNN
class CNN_CIFAR(nnx.Module):
  def __init__(self, num_classes: int = 10, *, rngs: nnx.Rngs):
    # --- Block 1 (conv_layer_b1) ---
    self.conv1_1 = nnx.Conv(3, 32, kernel_size=(3, 3), padding="SAME", rngs=rngs)
    # self.bn1_1   = nnx.BatchNorm(32, rngs=rngs)
    self.conv1_2 = nnx.Conv(32, 32, kernel_size=(3, 3), padding="SAME", rngs=rngs)
    # self.bn1_2   = nnx.BatchNorm(32, rngs=rngs)
    # MaxPool with window (2,2) and stride (2,2)
    self.maxpool1 = partial(nnx.max_pool, window_shape=(2, 2), strides=(2, 2))
    
    # --- Block 2 (first part of conv_layer_b2) ---
    self.conv2_1 = nnx.Conv(32, 64, kernel_size=(3, 3), padding="SAME", rngs=rngs)
    # self.bn2_1   = nnx.BatchNorm(64, rngs=rngs)
    self.conv2_2 = nnx.Conv(64, 64, kernel_size=(3, 3), padding="SAME", rngs=rngs)
    # self.bn2_2   = nnx.BatchNorm(64, rngs=rngs)
    self.maxpool2 = partial(nnx.max_pool, window_shape=(2, 2), strides=(2, 2))
    
    # --- Block 3 (second part of conv_layer_b2) ---
    self.conv3_1 = nnx.Conv(64, 128, kernel_size=(3, 3), padding="SAME", rngs=rngs)
    # self.bn3_1   = nnx.BatchNorm(128, rngs=rngs)
    self.conv3_2 = nnx.Conv(128, 128, kernel_size=(3, 3), padding="SAME", rngs=rngs)
    # self.bn3_2   = nnx.BatchNorm(128, rngs=rngs)
    self.maxpool3 = partial(nnx.max_pool, window_shape=(2, 2), strides=(2, 2))
    
    # --- Fully Connected Layers ---
    # Note: Two separate dropout layers are used, mirroring the original usage.
    # self.dropout1 = nnx.Dropout(rate=0.5, rngs=rngs)
    self.linear1  = nnx.Linear(128 * 4 * 4, 1024, rngs=rngs)
    # self.dropout2 = nnx.Dropout(rate=0.5, rngs=rngs)
    self.linear2  = nnx.Linear(1024, num_classes, rngs=rngs)

  def __call__(self, x: jax.Array):
    # --- Block 1 ---
    x = self.conv1_1(x)
    # x = self.bn1_1(x)
    x = nnx.relu(x)
    x = self.conv1_2(x)
    # x = self.bn1_2(x)
    x = nnx.relu(x)
    x = self.maxpool1(x)

    # --- Block 2 ---
    x = self.conv2_1(x)
    # x = self.bn2_1(x)
    x = nnx.relu(x)
    x = self.conv2_2(x)
    # x = self.bn2_2(x)
    x = nnx.relu(x)
    x = self.maxpool2(x)

    # --- Block 3 ---
    x = self.conv3_1(x)
    # x = self.bn3_1(x)
    x = nnx.relu(x)
    x = self.conv3_2(x)
    # x = self.bn3_2(x)
    x = nnx.relu(x)
    x = self.maxpool3(x)

    # Flatten the features
    x = x.reshape(x.shape[0], -1)
    
    # --- Fully Connected Layers ---
    # x = self.dropout1(x)
    x = self.linear1(x)
    x = nnx.relu(x)
    # x = self.dropout2(x)
    x = self.linear2(x)
    return x

# ----------------------------------------------------------------------
# Data Handling
# ----------------------------------------------------------------------
def tf_to_jax(t):
    """Convert a TensorFlow EagerTensor to a JAX array."""
    return jnp.array(np.array(t))

def get_full_dataset(dataloader, steps):
    """Concatenate a TF DataLoader into full (x,y) arrays."""
    xs, ys = [], []
    for step, batch in zip(range(steps), dataloader.as_numpy_iterator()):
        if isinstance(batch, dict):
            x, y = batch["image"], batch["label"]
        else:
            x, y = batch
        
        x_jax = tf_to_jax(x)
        y_jax = tf_to_jax(y)
        
        xs.append(x_jax)
        ys.append(y_jax)

    x_full = jnp.concatenate(xs, axis=0)
    y_full = jnp.concatenate(ys, axis=0)
    return x_full, y_full

# ----------------------------------------------------------------------
# Build log_prob for a logistic regression (with Gaussian prior)
# ----------------------------------------------------------------------
def make_log_prob_fn(
    graphdef: nnx.GraphDef,
    state: nnx.State,
    x: jax.Array,
    y: jax.Array,
    prior_sigma: float = 1.0,
):
    """
    Create a log-prob function for MCMC sampling on an `nnx` model (e.g., LogReg).
    
    Returns:
      log_prob_fn: A function taking a State (parameters) and returning
                   the scalar log-prob = -NLL(theta) + logPrior(theta).
    """
    # 1) Merge the graphdef + state so we have a model to inspect
    model_full = nnx.merge(graphdef, state)

    # 2) Split out only the parameter portion from the state
    _, param_state, mutable_state = nnx.split(model_full, nnx.Param, ...)

    @jax.jit
    def log_prob_fn(theta):
        # a) Merge new parameters into model
        model_candidate = nnx.merge(graphdef, theta, mutable_state)
        # b) Forward pass
        logits = model_candidate(x)  # shape (batch_size, n_classes)
        # c) Negative log-likelihood
        log_softmax = jax.nn.log_softmax(logits, axis=-1)
        nll = -jnp.sum(log_softmax[jnp.arange(logits.shape[0]), y])
        # d) Gaussian prior on parameters
        def sum_of_squares(p):
            return jnp.sum(p**2)
        sq_leaves = jax.tree.map(sum_of_squares, theta)
        sum_sq_params = jax.tree_util.tree_reduce(lambda a, b: a + b, sq_leaves, 0.0)
        log_prior = -(0.5 / (prior_sigma**2)) * sum_sq_params
        # e) Combine data + prior => total log-prob
        return -nll + log_prior

    return log_prob_fn

# ----------------------------------------------------------------------
# Make a plain log-likelihood fn (no prior)
# ----------------------------------------------------------------------
def make_ll(
    graphdef: nnx.GraphDef,
    state: nnx.State,
    x: jax.Array,
    y: jax.Array,
):
    """
    Similar to make_log_prob_fn, but returns only the log-likelihood 
    (i.e. excluding prior).
    """
    model_full = nnx.merge(graphdef, state)
    _, param_state, mutable_state = nnx.split(model_full, nnx.Param, ...)

    @jax.jit
    def ll_fn(theta):
        # Merge new params
        model_candidate = nnx.merge(graphdef, theta, mutable_state)
        logits = model_candidate(x)  # shape (batch_size, n_classes)
        log_softmax = jax.nn.log_softmax(logits, axis=-1)
        nll = -jnp.sum(log_softmax[jnp.arange(logits.shape[0]), y])
        return -nll  # i.e. log-likelihood
    return ll_fn

def run_sampler_scan(sampler_step, theta_init, rng_key_init, epsilon, n_steps):
    """
    Single-chain version. 
    Returns all thetas, all accepted flags.
    """
    def one_step(carry, _):
        (theta, rng_key) = carry
        rng_key, subkey = jax.random.split(rng_key)
        theta_next, accepted = sampler_step(theta, subkey, epsilon)
        new_carry = (theta_next, rng_key)
        return new_carry, (theta_next, accepted)

    init_carry = (theta_init, rng_key_init)
    final_carry, (thetas, accepts) = jax.lax.scan(
        f=one_step,
        init=init_carry,
        xs=None,
        length=n_steps
    )
    return thetas, accepts

def run_sampler_scan_no_save_thetas(sampler_step, theta_init, rng_key_init, epsilon, n_steps):

    def one_step(carry, _):
        """
        carry = (theta, rng_key, step_idx, accepts)
        We'll overwrite 'accepts[step_idx]' at each iteration.
        """
        (theta, rng_key, step_idx, accepts) = carry
        rng_key, subkey = jax.random.split(rng_key)

        # Perform a single MCMC update
        theta_next, accepted = sampler_step(theta, subkey, epsilon)

        # Record acceptance at the current step
        accepts = accepts.at[step_idx].set(accepted)

        # Move to the next iteration
        new_carry = (theta_next, rng_key, step_idx + 1, accepts)
        return new_carry, None  # We don't collect any output in scan's 'ys'

    # Initialize acceptance array (n_steps in length)
    accepts_init = jnp.zeros(n_steps, dtype=jnp.bool_)

    # The carry stores current theta, RNG key, current step index, and the acceptance array
    init_carry = (theta_init, rng_key_init, 0, accepts_init)

    # Run the scan
    final_carry, _ = jax.lax.scan(
        f=one_step,
        init=init_carry,
        xs=None,
        length=n_steps
    )

    final_theta, _, _, final_accepts = final_carry

    return final_theta, final_accepts

# ----------------------------------------------------------------------
# Compute NLL (just for final analysis) on a list of states
# ----------------------------------------------------------------------
def compute_nll_of_samples(log_prob_fn, thetas, subsample=1):
    """
    Given a log-likelihood or log-prob function and a list of States, 
    compute the negative log-likelihood for each (subsampled) theta.
    
    Args:
      log_prob_fn:  function(theta) -> scalar log-likelihood or log-prob
      thetas:       list of length n_steps, each a nnx.State
      subsample:    keep every 'subsample' step for thinning
    
    Returns:
      A dictionary with:
        'theta_indices':  the step indices used
        'nlls':           NLL values for each selected sample
    """
    sel_indices = np.arange(0, len(thetas), subsample)
    nll_values = []
    for i in sel_indices:
        lp = log_prob_fn(thetas[i])  # log-likelihood or log-prob
        nll = -lp                    # negative of that
        nll_values.append(np.asarray(nll))  # to NumPy scalar

    return {
        'theta_indices': sel_indices,
        'nlls': np.array(nll_values),
    }


# ----------------------------------------------------------------------
# Main Script
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MCMC sampling with logistic regression model (nnx).")
    parser.add_argument('--sampler',
                        type=str,
                        default='make_mala_step',
                        choices=[
                            'make_mala_step',
                            'make_fmala_step',
                            'make_line_fmala_step',
                            'make_precon_fmala_step',
                            'make_precon_line_fmala_step',
                        ],
                        help="Which sampler function to use.")
    parser.add_argument('--epsilon', type=float, default=0.001, help="Step size.")
    parser.add_argument('--num_samples', type=int, default=10, help="Number of MCMC steps.")
    parser.add_argument('--subsample', type=int, default=1, help="Keep every nth sample for NLL.")
    parser.add_argument('--gpu_index', type=int, default=0, help="Which GPU device to use (fallback to CPU if none).")
    parser.add_argument('--prior_sigma', type=float, default=10.0, help="Std dev for Gaussian prior.")
    parser.add_argument('--save_path', type=str, default="mcmc_results_gpu_50000_even_longer.npy",
                        help="Where to save the dictionary-of-dictionaries.")
    parser.add_argument('--force', action='store_true',
                        help="If present, overwrite existing data for this sampler.")
    parser.add_argument('--n_time_iterations', type=int, default=1,
                        help="Number of times to run the chain for timing purposes.")
    parser.add_argument('--chunk', type=int, default=50000,
                        help="Number of times to run the chain for timing purposes.")
    args = parser.parse_args()

    sampler_name = args.sampler

    # ------------------------------------------------
    # 1) Select device
    # ------------------------------------------------
    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device):
        train_ds, test_ds, steps_per_epoch, _ = get_cifar10_dataloaders_no_tf(batch_size=1000, seed=0, buffer=1024)
        x_train, y_train = get_full_dataset(train_ds, steps_per_epoch)
        # Extract the last 10,000 examples from the dataset
        x_train_cpu = x_train[:args.chunk]#000]
        y_train_cpu = y_train[:args.chunk]#000]
     
    devices = jax.devices("gpu")
    device = devices[0]
    sampler = args.sampler #'make_fmala_step' #'make_mala_step'
    epsilon = args.epsilon
    num_samples = args.num_samples
    
    with jax.default_device(device):
        rng_key = jax.random.PRNGKey(0)
        model = CNN_CIFAR(rngs=nnx.Rngs(0))
        graphdef, state = nnx.split(model)
        x_train = jax.device_put(x_train_cpu, device)
        y_train = jax.device_put(y_train_cpu, device)
        log_prob_fn = make_log_prob_fn(graphdef, state, x_train, y_train)
    
        sampler_builder = eval(sampler)(log_prob_fn)
    
        @jax.jit
        def sampler_step_jitted(theta, rng_key, epsilon):
            """One MCMC update (JIT-compiled)."""
            return sampler_builder(theta, rng_key, epsilon)
    
        # First: "warm-up" / compile a single step to ensure 
        # we don't measure compilation time:
        dummy_theta, dummy_accepted = sampler_step_jitted(state, rng_key, epsilon)
        jax.block_until_ready(dummy_theta)
        jax.block_until_ready(dummy_accepted)
        print("Compiled")
        # We'll store times for each iteration in a list
        iteration_times = []
        final_thetas = None
        final_accepts = None

        # ------------------------------------------------
        # 6) Run single-chain MCMC multiple times for timing
        # ------------------------------------------------
        for _ in range(args.n_time_iterations):
            t0 = time.time()
            # thetas, accepts = run_sampler_scan_list(
            thetas, accepts = run_sampler_scan_no_save_thetas(
                sampler_step_jitted,
                state,       # initial "theta"
                rng_key,     # RNG
                epsilon,
                num_samples
            )
            jax.block_until_ready(thetas)  # ensure last sample is computed
            t1 = time.time()

            iteration_times.append(t1 - t0)
        #     final_thetas = thetas
        #     final_accepts = accepts

        # acceptance_rate = np.mean(np.array(final_accepts))
        print(f"\nSampler: {sampler_name}")
        print(f"Times for each of {args.n_time_iterations} runs: {iteration_times}")

        # ------------------------------------------------
        # 7) Compute negative log-likelihood on final chain
        # ------------------------------------------------
        results_dict = {}
        results_dict['times'] = iteration_times  # store the time of each iteration

        # ------------------------------------------------
        # 8) Save results to disk in a dictionary-of-dicts
        # ------------------------------------------------
        if not os.path.exists(args.save_path):
            # create empty dictionary-of-dicts
            big_dict = {}
        else:
            # load existing
            big_dict = np.load(args.save_path, allow_pickle=True).item()

        if (sampler_name in big_dict) and (not args.force):
            print(f"Results for sampler={sampler_name} already exist in {args.save_path}.")
            print("Use --force to overwrite.")
        else:
            big_dict[sampler_name] = results_dict
            big_dict["args"] = args
            np.save(args.save_path, big_dict, allow_pickle=True)
            print(f"Results for sampler='{sampler_name}' saved to {args.save_path}.")

if __name__ == "__main__":
    main()
