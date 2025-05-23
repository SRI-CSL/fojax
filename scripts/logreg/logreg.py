#!/usr/bin/env python

import argparse, json
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
from flax import nnx
import tensorflow as tf, tensorflow_datasets as tfds

# fojax / local imports
from fojax.nn import LogReg
from fojax.data import get_mnist_dataloaders_no_tf
from fojax.sampler import *

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

# -----------------------------------------------------------------
# ▶︎ Posterior-predictive metrics on the last 100 thinned samples
#    (accuracy, NLL, ECE) -----------------------------------------
# -----------------------------------------------------------------
import torch
from torchmetrics import CalibrationError

def posterior_stats_last_k(
    thetas,               # list/array of PyTrees (CPU)
    graphdef,             
    state_init,           # the nnx.State you built the model with
    x_eval, y_eval,       # JAX arrays (full train or held-out test set)
    k: int = 100,         # how many thinned samples to use
):
    """
    Returns dict(posterior_acc, posterior_nll, posterior_ece)
    and prints the trio nicely.
    """
    # --- split once to recover the "mutable" buffers ----------------
    _, _, mutable_state = nnx.split(nnx.merge(graphdef, state_init),
                                    nnx.Param, ...)

    k = min(k, len(thetas))
    sel_thetas = thetas[-k:]                             # last k samples

    logps = []                                           # (k, N, C)
    for θ in sel_thetas:
        θ_jax = jax.tree_util.tree_map(jnp.asarray, θ)
        logits = nnx.merge(graphdef, θ_jax, mutable_state)(x_eval)
        logps.append(np.asarray(jax.nn.log_softmax(logits, -1)))

    logps = torch.tensor(np.stack(logps), dtype=torch.float32)
    targets = torch.tensor(np.asarray(y_eval), dtype=torch.long)

    # average predictive distribution
    prob = logps.exp().mean(0)                           # (N, C)

    acc = (prob.argmax(1) == targets).float().mean().item()
    nll = torch.nn.functional.nll_loss(prob.log(), targets,
                                       reduction="mean").item()

    ece_metric = CalibrationError(task="multiclass",
                                  n_bins=100,
                                  num_classes=prob.shape[1],
                                  norm="l1")
    ece = ece_metric(prob, targets).item()

    print(f"Posterior predictive (last {k} samples): "
          f"Acc={acc:.4f}  NLL={nll:.4f}  ECE={ece:.4f}")

    return dict(posterior_acc=acc,
                posterior_nll=nll,
                posterior_ece=ece)

    
def tf_to_jax(t):
    return jnp.array(np.array(t))

def get_full_dataset(dataloader, steps):
    xs, ys, it = [], [], iter(dataloader)
    for _ in range(steps):
        batch = next(it)
        x, y = (batch["image"], batch["label"]) if isinstance(batch, dict) else batch
        xs.append(tf_to_jax(x)); ys.append(tf_to_jax(y))
    return jnp.concatenate(xs,0), jnp.concatenate(ys,0)

# ---------------------------------------------------------------------
# Log-prob and accuracy
# ---------------------------------------------------------------------
def build_fns(graphdef, state, x, y, prior_sigma=1.0):
    model_full = nnx.merge(graphdef, state)
    _, param_state, mutable_state = nnx.split(model_full, nnx.Param, ...)

    @jax.jit
    def log_prob(theta):
        m = nnx.merge(graphdef, theta, mutable_state)
        logits = m(x)
        logsm = jax.nn.log_softmax(logits, -1)
        nll = -jnp.sum(logsm[jnp.arange(logits.shape[0]), y])
        l2 = jax.tree_util.tree_reduce(
            lambda a,b: a+b,
            jax.tree_map(lambda p: jnp.sum(p**2), theta), 0.0)
        return -nll - 0.5*l2/(prior_sigma**2)

    @jax.jit
    def accuracy(theta):
        m = nnx.merge(graphdef, theta, mutable_state)
        preds = jnp.argmax(m(x), -1)
        return jnp.mean(preds == y)

    return log_prob, accuracy, param_state

# ---------------------------------------------------------------------
# Sampling with on-the-fly diagnostics
# ---------------------------------------------------------------------
def run_sampler_thinned(step_fn, theta0, rng0, eps, n_steps, thin,
                        log_prob_fn, acc_fn):
    theta, rng = theta0, rng0
    thetas, accepts, nlls, accs = [], [], [], []
    for it in range(n_steps):
        rng, sub = jax.random.split(rng)
        theta, acc = step_fn(theta, sub, eps)

        if it % thin == 0:
            theta_cpu = jax.device_get(theta)
            thetas.append(theta_cpu)
            accepts.append(bool(acc))

            nll = float(-log_prob_fn(theta))   # negative log-prob
            acc_val = float(acc_fn(theta))
            nlls.append(nll); accs.append(acc_val)

            print(f"[{it:>6}] acc={bool(acc)}, "
                  f"acc_rate={np.mean(accepts):.3f}, "
                  f"NLL={nll:.2f}, "
                  f"train-acc={acc_val*100:.2f}%")
        if it > 10000 and np.mean(accepts) < 0.02:
            break

    return thetas, np.array(accepts), np.array(nlls), np.array(accs)

# ---------------------------------------------------------------------
def main():
    P = argparse.ArgumentParser()
    P.add_argument('--sampler', default='make_mala_step',
                   choices=['make_mala_step','make_fmala_step',
                            'make_line_fmala_step','make_precon_fmala_step', 'make_precon_fmala_step_no_correction',
                            'make_precon_line_fmala_step'])
    P.add_argument('--epsilon', type=float, default=1e-3)
    P.add_argument('--num_samples', type=int, default=1000)
    P.add_argument('--subsample', type=int, default=10)
    P.add_argument('--gpu_index', type=int, default=0)
    P.add_argument('--prior_sigma', type=float, default=1.0)
    P.add_argument('--save_path', default="/project/synthesis/fmode/logreg/mcmc_results.npy")
    args = P.parse_args()

    # -----------------------------------------------------------------
    gpus = jax.devices("gpu")
    device = gpus[args.gpu_index] if gpus else jax.devices("cpu")[0]

    with jax.default_device(device):
        train_ds, test_ds, steps_train, steps_test = get_mnist_dataloaders_no_tf(
            batch_size=1000, seed=0)
        x_train, y_train = get_full_dataset(train_ds, steps_train)
        x_test, y_test = get_full_dataset(test_ds, steps_test)
        x_train = x_train.reshape(-1, 28*28)
        x_test = x_test.reshape(-1, 28*28)

        rng = jax.random.PRNGKey(0)
        model = LogReg(28*28, 10, rngs=nnx.Rngs(0))
        gdef, state = nnx.split(model)

        logp, acc_fn, theta_init = build_fns(
            gdef, state, x_train, y_train, args.prior_sigma)

        if args.sampler == "make_mala_step":
            step_fn = eval(args.sampler)(logp)
        elif args.sampler == 'make_precon_fmala_step_no_correction':
            step_fn = eval(args.sampler[:-14])(logp)
        else:
            step_fn = eval(args.sampler)(logp, theta_init)

        thetas, accepts, nlls, accs = run_sampler_thinned(
            step_fn, theta_init, rng,
            args.epsilon, args.num_samples, args.subsample,
            logp, acc_fn)

        print(f"\nFinal acceptance-rate: {accepts.mean():.3f}")

    # -----------------------------------------------------------------

    post_stats = posterior_stats_last_k(
        thetas=thetas,
        graphdef=gdef,
        state_init=state,
        x_eval=x_test,          
        y_eval=y_test,
        k=100,
    )
    
    # -----------------------------------------------------------------
    results = {
        **vars(args),           
        "thetas": thetas,
        "accepts": accepts,
        "nlls": nlls,
        "train_accs": accs,
        "accept_rate": float(accepts.mean()),
        **post_stats,          
    }
    
    np.save(args.save_path, results, allow_pickle=True)
    print(f"Saved {len(thetas)} thinned samples (+ diagnostics) to {args.save_path}")

if __name__ == "__main__":
    main()


# python logreg.py --sampler make_mala_step --epsilon 0.001 --num_samples 100 --subsample 2 --gpu_index 0
