#!/usr/bin/env python

import argparse, time, json
from pathlib import Path
import os

import numpy as np
import jax, jax.numpy as jnp
from flax import nnx

# fojax / local imports --------------------------------------------------------
from fojax.nn import MLP
from fojax.sampler import * 

# -----------------------------------------------------------------------------
# ▶︎ Data helpers --------------------------------------------------------------
# -----------------------------------------------------------------------------

def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path, mode=0o777)

def load_agw_1d(base_dir = "./data/", get_feats=False):
    agw_dir = os.path.join(base_dir, 'agw_data')
    if not os.path.exists(agw_dir):
        mkdir(agw_dir)
        import urllib
        urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/wjmaddox/drbayes/master/experiments/synthetic_regression/ckpts/data.npy',
            filename=os.path.join(agw_dir, 'data.npy')
        )

    data_path = os.path.join(agw_dir, 'data.npy')
    data = np.load(data_path)  # shape: (N, 2)
    x, y = data[:, 0], data[:, 1]
    y = y[:, None]  # (N, 1)

    def features(x_):
        return np.hstack([x_[:, None] / 2.0, (x_[:, None] / 2.0)**2])

    f = features(x)

    # Simple standardization of X, Y, and feature F
    x_means, x_stds = x.mean(axis=0), x.std(axis=0)
    y_means, y_stds = y.mean(axis=0), y.std(axis=0)
    f_means, f_stds = f.mean(axis=0), f.std(axis=0)

    X = ((x - x_means) / x_stds).astype(np.float32)
    Y = ((y - y_means) / y_stds).astype(np.float32)
    F = ((f - f_means) / f_stds).astype(np.float32)

    if get_feats:
        return F, Y
    # Return shape (N,1) for X and Y
    return X[:, None], Y

# ----------------------------------------------------------------------------
# ▶︎ Log‑prob + RMSE helpers ---------------------------------------------------
# ----------------------------------------------------------------------------


def _count_params(pytree) -> int:
    """Total number of scalar parameters in a PyTree."""
    leaves = jax.tree_util.tree_leaves(pytree)
    return sum(l.size for l in leaves)


def build_fns(
    graphdef: nnx.GraphDef,
    state: nnx.State,
    x: jax.Array,
    y: jax.Array,
    prior_sigma: float = 0.1,
    lik_sigma: float = 0.025,
):
    """Return (log_prob_fn, rmse_fn, param_state).

    • log_prob_fn — mean log‑prob under Gaussian likelihood + prior
    • rmse_fn — root‑mean‑squared‑error on the provided dataset
    """

    model_full = nnx.merge(graphdef, state)
    _, param_state, mutable_state = nnx.split(model_full, nnx.Param, ...)

    N = y.shape[0]
    M = _count_params(param_state)

    log_lik_const = -0.5 * jnp.log(2.0 * jnp.pi * (lik_sigma ** 2))
    log_prior_const = -0.5 * M * jnp.log(2.0 * jnp.pi * (prior_sigma ** 2))

    @jax.jit
    def log_prob(theta):
        m = nnx.merge(graphdef, theta, mutable_state)
        preds = m(x)
        sq_diff = jnp.sum((preds - y) ** 2)
        sum_ll = N * log_lik_const - 0.5 * sq_diff / (lik_sigma ** 2)

        sq_param = jax.tree_util.tree_reduce(
            lambda a, b: a + b,
            jax.tree_map(lambda p: jnp.sum(p ** 2), theta),
            0.0,
        )
        prior_quad = -0.5 * sq_param / (prior_sigma ** 2)
        log_prior = log_prior_const + prior_quad
        return (sum_ll + log_prior) / N

    @jax.jit
    def rmse(theta):
        m = nnx.merge(graphdef, theta, mutable_state)
        preds = m(x)
        return jnp.sqrt(jnp.mean((preds - y) ** 2))

    return log_prob, rmse, param_state

# ----------------------------------------------------------------------------
# ▶︎ Posterior‑predictive diagnostics (last k thinned samples) ----------------
# ----------------------------------------------------------------------------


def posterior_stats_last_k(
    thetas,
    graphdef,
    state_init,
    x_eval,
    y_eval,
    lik_sigma: float = 0.025,
    k: int = 100,
):
    """Compute RMSE and predictive NLL from the last k samples."""

    _, _, mutable_state = nnx.split(nnx.merge(graphdef, state_init), nnx.Param, ...)

    k = min(k, len(thetas))
    sel = thetas[-k:]

    preds = []  # (k, N, 1)
    for θ in sel:
        θ_jax = jax.tree_util.tree_map(jnp.asarray, θ)
        outputs = nnx.merge(graphdef, θ_jax, mutable_state)(x_eval)
        preds.append(np.asarray(outputs))
    preds = np.stack(preds)  # k × N × 1

    mean_pred = preds.mean(0).squeeze()  # N
    y_np = np.asarray(y_eval).squeeze()
    rmse = float(np.sqrt(((mean_pred - y_np) ** 2).mean()))

    log_const = -0.5 * np.log(2 * np.pi * (lik_sigma ** 2))
    sq = (preds.squeeze() - y_np[None, :]) ** 2
    logp_samples = log_const - sq / (2 * (lik_sigma ** 2))
    log_pred_density = np.log(np.exp(logp_samples).mean(0))  # MC log‑prob
    nll = float(-log_pred_density.mean())

    print(
        f"Posterior predictive (last {k} samples): RMSE={rmse:.4f}  NLL={nll:.4f}"
    )

    return dict(posterior_rmse=rmse, posterior_nll=nll)

# ----------------------------------------------------------------------------
# ▶︎ MCMC loop with thinning + live diagnostics ------------------------------
# ----------------------------------------------------------------------------


def run_sampler_thinned(
    step_fn,
    theta0,
    rng0,
    eps,
    n_steps,
    thin,
    log_prob_fn,
    rmse_fn,
    burn,
):
    """Run MCMC in a Python loop, keep every `thin`‑th sample on CPU."""

    theta, rng = theta0, rng0
    thetas, accepts, nlls, rmses = [], [], [], []

    for it in range(n_steps):
        rng, sub = jax.random.split(rng)
        theta, acc = step_fn(theta, sub, eps)

        if it % thin == 0:
            theta_cpu = jax.device_get(theta)
            accepts.append(bool(acc))

            nll = float(-log_prob_fn(theta))
            rmse_val = float(rmse_fn(theta))
            nlls.append(nll)
            rmses.append(rmse_val)
            if it > burn * n_steps:
                thetas.append(theta_cpu)

            print(
                f"[{it:>6}] acc={bool(acc)}, "
                f"acc_rate={np.mean(accepts):.3f}, "
                f"NLL={nll:.3f}, "
                f"RMSE={rmse_val:.4f}"
            )

        # crude stuck‑chain check
        if it > 10000 and np.mean(accepts) < 0.02:
            break

    return thetas, np.array(accepts), np.array(nlls), np.array(rmses)

# ----------------------------------------------------------------------------
# ▶︎ Main ---------------------------------------------------------------------
# ----------------------------------------------------------------------------


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--sampler", default="make_mala_step", choices=[
        "make_mala_step",
        "make_fmala_step",
        "make_line_fmala_step",
        "make_precon_fmala_step",
        "make_precon_fmala_step_no_correction",
        "make_precon_line_fmala_step",
    ])
    P.add_argument("--epsilon", type=float, default=1e-3)
    P.add_argument("--num_samples", type=int, default=1000)
    P.add_argument("--subsample", type=int, default=10)
    P.add_argument("--burn", type=float, default=0.8)
    P.add_argument("--collect", type=int, default=100)
    P.add_argument("--gpu_index", type=int, default=0)
    P.add_argument("--prior_sigma", type=float, default=0.1)
    P.add_argument("--lik_sigma", type=float, default=0.025)
    P.add_argument("--save_path", default="./reg_mcmc_results.npy")
    args = P.parse_args()

    # ---------------------------------------------------------------------
    # Device
    # ---------------------------------------------------------------------
    devices = jax.devices("gpu")
    device = devices[args.gpu_index] if devices else jax.devices("cpu")[0]

    with jax.default_device(device):
        # -----------------------------------------------------------------
        # Data
        # -----------------------------------------------------------------
        x_train, y_train = load_agw_1d(get_feats=False)
        x_val, y_val = x_train, y_train  # TEMP: duplicate train as validation

        # -----------------------------------------------------------------
        # Model
        # -----------------------------------------------------------------
        rng = jax.random.PRNGKey(0)
        model = MLP(rngs=nnx.Rngs(0))  # MLP definition decides layer sizes
        gdef, state = nnx.split(model)

        # -----------------------------------------------------------------
        # Log‑prob + RMSE fns
        # -----------------------------------------------------------------
        logp, rmse_fn, theta_init = build_fns(
            gdef, state, x_train, y_train, args.prior_sigma, args.lik_sigma
        )

        # -----------------------------------------------------------------
        # Sampler step fn
        # -----------------------------------------------------------------
        if args.sampler == "make_mala_step":
            step_fn = eval(args.sampler)(logp)
        elif args.sampler == "make_precon_fmala_step_no_correction":
            step_fn = eval(args.sampler[:-14])(logp)
        else:
            step_fn = eval(args.sampler)(logp, theta_init)

        # -----------------------------------------------------------------
        # MCMC
        # -----------------------------------------------------------------
        thetas, accepts, nlls, rmses = run_sampler_thinned(
            step_fn,
            theta_init,
            rng,
            args.epsilon,
            args.num_samples,
            args.subsample,
            logp,
            rmse_fn,
            args.burn,
        )

        print(f"\nFinal acceptance‑rate: {accepts.mean():.3f}")

    # ---------------------------------------------------------------------
    # Posterior predictive on validation set
    # ---------------------------------------------------------------------
    post_stats = posterior_stats_last_k(
        thetas,
        gdef,
        state,
        x_val,
        y_val,
        args.lik_sigma,
        k=args.collect,
    )

    # ---------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------
    results = {
        **vars(args),
        "thetas": thetas,
        "accepts": accepts,
        "nlls": nlls,
        "rmses": rmses,
        "accept_rate": float(accepts.mean()),
        **post_stats,
    }

    np.save(args.save_path, results, allow_pickle=True)
    print(f"Saved {len(thetas)} thinned samples (+ diagnostics) to {args.save_path}")


if __name__ == "__main__":
    main()


# python sample_regression.py --epsilon 0.01 --num_samples 10000