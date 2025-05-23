import jax.scipy.stats as stats
from fojax.sampler import *
import matplotlib.pyplot as plt
import jax

from tqdm.notebook import tqdm
import numpy as np
import math
import os

import argparse
import arviz as az

def kl_normal(p_mean, p_std, q_mean, q_std):
    """Compute KL divergence D_KL(p || q) for two univariate normals."""
    return np.log(q_std / p_std) + (p_std**2 + (p_mean - q_mean)**2) / (2 * q_std**2) - 0.5
    
def ess_arviz(samples_batched):
    # Suppose `chain` is (n_samples, n_chains, n_variables)
    chains = np.asarray(jnp.permute_dims(samples_batched,(1,0, 2)))
    idata = az.from_dict(posterior={"x": chains})
    ess = az.ess(idata, var_names=["x"])

    return ess.to_array().data

@jax.jit
@safe_log_prob
def funnel_ll(w):
    """
    w: 1D array where w[0] = 'v' and w[1:] = 'x' in the funnel distribution.
        - p(v) = Normal(0, 3)
        - p(x | v) = Normal(0, exp(-v/2))
    Returns the scalar log-likelihood.
    """
    v = w[0]
    x = w[1:]  # everything else

    # Log prob of v ~ Normal(0, 3)
    ll_v = stats.norm.logpdf(v, loc=0., scale=3.0)
    
    # For x ~ Normal(0, exp(-v/2)):
    scale_x = jnp.exp(-v * 0.5)
    ll_x = jnp.sum(stats.norm.logpdf(x, loc=0., scale=scale_x))

    # Combine
    return ll_v + ll_x

# @jax.jit
def run_sampler_scan(sampler_step, theta_init, rng_key_init, epsilon, n_steps):
    """
    Uses jax.lax.scan to run MALA for n_steps (single chain).
    Returns all thetas, all accepted flags, plus final carry states.
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
    return thetas, accepts#, final_carry

# @jax.jit @jax.jit(static_argnums=0)
def run_sampler_scan_batched(sampler_step_batched, theta_init, rng_keys_init, epsilon, n_steps):
    """
    run MALA for multiple chains in parallel,
    using jax.lax.scan to do n_steps iteration.
    """
    def one_step(carry, _):
        theta, rng_keys = carry
        # each chain key is split once
        subkeys = jax.vmap(lambda k: jax.random.split(k, 2))(rng_keys)
    
        rng_keys_out = subkeys[:, 0]
        keys_for_step = subkeys[:, 1]
    
        theta_next, accepted = sampler_step_batched(theta, keys_for_step, epsilon)
        new_carry = (theta_next, rng_keys_out)
        return new_carry, (theta_next, accepted)


    init_carry = (theta_init, rng_keys_init)
    final_carry, (thetas, accepts) = jax.lax.scan(
        f=one_step,
        init=init_carry,
        xs=None,
        length=n_steps
    )

    return thetas, accepts#, final_carry


def p_init(D, num_chains, rng_key):
    init_shape = jnp.zeros((num_chains, D+1))
    theta = tree_random_normal(rng_key, init_shape)   
    theta = theta * 0.1
    return theta
    

def run_mcmc_and_compute_kl(epsilon, sampler, num_samples=2000, D = 10, num_chains = 1, seed = 0):
    master_key = jax.random.PRNGKey(seed)
    chain_key, init_key = jax.random.split(master_key, 2)
    chain_keys = jax.random.split(chain_key, num_chains)
    theta0 = p_init(D, num_chains, init_key)

    step = eval(sampler)
    sampler_step = step(funnel_ll)
    sampler_step_batched = jax.vmap(sampler_step, in_axes=(0, 0, None))
    
    thetas_batched, accepts_batched = run_sampler_scan_batched(
        sampler_step_batched, theta0, chain_keys, epsilon, num_samples
    )
    ave_accept = accepts_batched.mean()
    ess = ess_arviz(thetas_batched).flatten()
    print(f"Ave. Acceptance: {ave_accept:.2f}")

    samples = np.asarray(thetas_batched.reshape(-1,D+1))
    q_mean = np.mean(samples[:, 0])
    q_std =  np.std(samples[:, 0], ddof=1)  # ddof=1 for the unbiased estimator
    # Compute the KL divergence between p and q:
    try:
        # Compute empirical Normal from samples[:,0]
        kl_val = kl_normal(0, 3, q_mean, q_std)
    except ValueError: # std is 0.0
        # Compute empirical Normal from samples[:,0]
        kl_val = kl_normal(0, 3, q_mean, q_std + 10e-5)
        
    return kl_val, ave_accept, ess


def objective_factory(sampler, num_samples, eps_min, eps_max, D, num_chains, seed):
    def objective(epsilon):
        kl_val, accept_rate, ess = run_mcmc_and_compute_kl(
            epsilon=epsilon,
            sampler=sampler,     
            num_samples=num_samples,
            D=D,
            num_chains = num_chains,
            seed = seed
        )
        return kl_val, accept_rate, ess
    return objective

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperopt over funnel', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_samples', type=int, default=5000)
    parser.add_argument('--num_chains', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_trials', type=int, default=100)
    parser.add_argument('--funnel_D', type=int, default=10)
    parser.add_argument('--sampler', type=str, choices=['make_mala_step', 'make_fmala_step', 'make_line_fmala_step', 'make_precon_fmala_step', 'make_precon_line_fmala_step'], default='make_mala_step')
    parser.add_argument('--eps_min', type=float, default=0.1) #0.01 HMC
    parser.add_argument('--eps_max', type=float, default=2.0)
    parser.add_argument('--bashscript', action='store_true')
    parser.add_argument('--log', action='store_true')
    args = parser.parse_args()

    cpu_device = jax.devices('cpu')[0]

    # # Use a with statement to set the default device
    with jax.default_device(cpu_device):
        print("Grid Search")
        # Define grid
        if args.log:
            grid = np.logspace(np.log10(args.eps_min),np.log10(args.eps_max), args.num_trials).tolist()
        else:
            grid = np.linspace(args.eps_min, args.eps_max, args.num_trials).tolist()
        objective_func = objective_factory(args.sampler, args.num_samples, args.eps_min, args.eps_max, args.funnel_D, args.num_chains, args.seed)
    
        # Define save path
        if args.bashscript:
            save_path = f"./bash_{args.funnel_D}D/{args.sampler}_chains{args.num_chains}_simple_grid_seed_{args.seed}"
        else:
            save_path = f"./{args.funnel_D}D/{args.sampler}_chains{args.num_chains}_simple_grid_seed_{args.seed}"
    
        # Initialize or load previous results
        if os.path.exists(save_path):
            results = np.load(save_path)
            print(f"Resuming from saved progress: {save_path}")
            kl_list = results.get('minimization_objective', [])
            eps_list = results.get('parameter', [])
            accept_rate_list = results.get('accept_rate', [])
            ess_list = results.get('ess', [])
        else:
            results = {'args': args}
            kl_list = []
            eps_list = []
            accept_rate_list = []
            ess_list = []
    
        # Determine where to resume
        start_index = len(kl_list)
    
        # Resume from the last completed point
        for i in range(start_index, len(grid)):
            eps = grid[i]
            kl, accept_rate, ess = objective_func(eps)
            kl_list.append(kl)
            eps_list.append(eps)
            accept_rate_list.append(accept_rate)
            ess_list.append(ess)
    
            # Update results dictionary
            results['minimization_objective'] = kl_list
            results['parameter'] = eps_list
            results['accept_rate'] = accept_rate_list
            results['ess'] = ess_list
    
            # Save after every iteration to ensure progress is stored
            np.save(save_path, results)
    
            best_index = np.nanargmin(kl_list)
            print(f"Best so far: {kl_list[best_index]:.3f}, with epsilon = {eps_list[best_index]:.3f}")
    
        print("Search completed.")

    
    # python funnel_grid.py --sampler make_fmala_step --num_trials 100 --funnel_D 10 --seed 0
