import jax
import jax.numpy as jnp
import functools

@jax.jit
def tree_random_normal(rng_key, example_tree):#, dtype=jnp.float32):
    """
    Sample a PyTree of random normals, matching the structure & shape of example_tree.
    Each leaf gets a distinct subkey.
    """
    leaves, treedef = jax.tree_util.tree_flatten(example_tree)
    num_leaves = len(leaves)
    subkeys = jax.random.split(rng_key, num_leaves)

    out_leaves = []
    for sk, leaf in zip(subkeys, leaves):
        out_leaves.append(jax.random.normal(sk, shape=leaf.shape))   #, dtype=jnp.float32
    return jax.tree_util.tree_unflatten(treedef, out_leaves)

@jax.jit
def tree_add_scaled(a, b, alpha, beta):
    """
    Return alpha * a + beta * b, broadcasting across each leaf of PyTrees a, b.
    Works for any PyTree structure (must match).
    """
    return jax.tree.map(lambda x, y: alpha * x + beta * y, a, b)

@jax.jit
def tree_vdot(t1, t2):
    """
    Compute the dot product of two PyTrees t1 and t2.
    Assumes they have the same structure.
    """
    leaves1, _ = jax.tree_util.tree_flatten(t1)
    leaves2, _ = jax.tree_util.tree_flatten(t2)
    return sum(jnp.vdot(a, b) for a, b in zip(leaves1, leaves2))

@jax.jit
def tree_l2_norm(v):
    """Compute the global L2 norm (Euclidean norm) of a PyTree."""
    leaves, _ = jax.tree_util.tree_flatten(v)
    sq_norms = sum(jnp.sum(leaf ** 2) for leaf in leaves)  # Sum of squares
    return jnp.sqrt(sq_norms)  # Final norm

@jax.jit
def tree_norm(v):
    """
    Norm vector in pytree
    """
    norm = tree_l2_norm(v) #+ 1e-8  # Add small epsilon to avoid division by zero
    return jax.tree.map(lambda x: x / norm, v)

@jax.jit
def tree_normal_log_prob(x, mean, scale):
    """
    Compute the sum of log-probs for x ~ Normal(mean, scale)
    across all leaves of the PyTree x and mean (same structure).
    scale can be a scalar or a PyTree of the same structure.
    """
    # Flatten
    x_leaves, xdef = jax.tree_util.tree_flatten(x)
    m_leaves, mdef = jax.tree_util.tree_flatten(mean)
    # If scale is a scalar, we just broadcast it in the loop below.
    # If scale is also a PyTree, flatten it similarly (omitted here for brevity).
    
    total = 0.0
    for xl, ml in zip(x_leaves, m_leaves):
        D = xl.size
        # We'll treat 'scale' as a scalar for simplicity:
        term = -0.5 * jnp.sum(((xl - ml) / scale)**2) \
               - D * jnp.log(scale * jnp.sqrt(2*jnp.pi))
        total += term
    return total

def safe_log_prob(original_log_prob_fn):
    """
    Decorator that wraps a log-prob function so that any NaN/∞ values
    become -∞ (causing an automatic 'reject' in MCMC).
    """
    @functools.wraps(original_log_prob_fn)
    def wrapped_fn(theta, *args, **kwargs):
        lp = original_log_prob_fn(theta, *args, **kwargs)
        return jnp.where(jnp.isfinite(lp), lp, -jnp.inf)
    return wrapped_fn

# # Example usage:
# @safe_log_prob
# def my_log_prob(theta):
#     # Suppose your log_prob can return NaN for invalid domain, etc.
#     return -0.5 * jnp.sum(theta**2)  # Just a simple Gaussian shape

def count_parameters(obj):
    """Return the total number of elements in `obj`, which can be:
       - a single JAX array
       - a nested PyTree of JAX arrays
       - or any structure of nested lists/dicts/tuples containing JAX arrays.
    """
    leaves, _ = jax.tree_util.tree_flatten(obj)
    return float(sum(jnp.asarray(leaf).size for leaf in leaves))

def make_mala_step(log_prob_fn):#, dtype=jnp.float32):
    """Create a JIT-compiled MALA step that uses `log_prob_fn` internally."""
    
    @jax.jit
    def mala_step(theta, rng_key, epsilon):
        rng_key, rng_accept = jax.random.split(rng_key, 2)
        
        # 1) Sample z ~ N(0,I)
        z = tree_random_normal(rng_key, theta)#, dtype=dtype)

        # 2) Gradient wrt the real part of log_prob (if complex)
        grad = jax.grad(lambda th: log_prob_fn(th))(theta)

        # 3) Propose theta_star
        theta_star = jax.tree.map(
            lambda t, g, zz: t + 0.5*(epsilon**2)*g + epsilon*zz,
            theta, grad, z
        )
        # theta_star = theta + 0.5 * (epsilon**2) * grad + epsilon * z

        # 4) Gradient at theta_star
        grad_star = jax.grad(lambda th: log_prob_fn(th))(theta_star)

        # 5) Drift terms mu_cur and mu_star
        mu_cur = jax.tree.map(
            lambda t, g: t + 0.5*(epsilon**2)*g,
            theta, grad
        )
        mu_star = jax.tree.map(
            lambda t, g: t + 0.5*(epsilon**2)*g,
            theta_star, grad_star
        )

        # 6) Log-prob of Normal(...) with mean=mu, scale=epsilon
        lp_q_star_given_cur = tree_normal_log_prob(theta_star, mu_cur, epsilon)
        lp_q_cur_given_star = tree_normal_log_prob(theta,      mu_star, epsilon)

        # 7) Evaluate log_prob at theta and theta_star
        lp_theta = log_prob_fn(theta)
        lp_theta_star = log_prob_fn(theta_star)

        # 8) Metropolis–Hastings acceptance ratio
        log_acceptance_ratio = (
            lp_theta_star + lp_q_cur_given_star
            - lp_theta    - lp_q_star_given_cur
        )

        # 9) Accept/reject
        u = jax.random.uniform(rng_accept, ())
        accept = (jnp.log(u) < log_acceptance_ratio)

        # 10) Return either theta_star or the old theta
        theta_next = jax.lax.cond(
            accept,
            lambda _: theta_star,
            lambda _: theta,
            operand=None
        )
        return theta_next, accept

    return mala_step


def make_fmala_step(log_prob_fn, theta_example = None):
    """Create a JIT-compiled FMALA step that uses `log_prob_fn` internally."""

    if theta_example is not None:
        D = count_parameters(theta_example)
    else:
        D = 1.0
        print("Not implmementing correction")
    
    @jax.jit
    def fmala_step(theta, rng_key, epsilon):
        
        # test:
        epsilon = epsilon * (D**0.5) 
        
        rng_key, rng_accept, rng_v_t, rng_v_star = jax.random.split(rng_key, 4)
        
        # 0) Sample z ~ N(0,I)
        z = tree_random_normal(rng_key, theta)

        # 1) Sample v_t ~ N(0,I)
        v_t = tree_random_normal(rng_v_t, theta)
        v_t = tree_norm(v_t)

        # 2) Gradient wrt the real part of log_prob (if complex)
        lp_theta, jvp_t = jax.jvp(lambda th: log_prob_fn(th), (theta,), (v_t,))

        # 3) Propose theta_star
        theta_star = jax.tree.map(
            lambda t, v, zz: t + 0.5*(epsilon**2)*jvp_t*v + epsilon*zz, #### Removed D
            theta, v_t, z
        )

        # 4) Gradient at theta_star
        v_star = tree_random_normal(rng_v_star, theta)
        v_star = tree_norm(v_star)
        lp_theta_star, jvp_star = jax.jvp(lambda th: log_prob_fn(th), (theta_star,), (v_star,))

        # 5) Drift terms mu_cur and mu_star
        mu_cur = jax.tree.map(
            lambda t, v: t +  0.5*(epsilon**2)*jvp_t*v, #### Removed D
            theta, v_t
        )
        mu_star = jax.tree.map(
            lambda t, v: t +  0.5*(epsilon**2)*jvp_star*v, #### Removed D
            theta_star, v_star
        )

        lp_q_star_given_cur = tree_normal_log_prob(theta_star, mu_cur, epsilon)
        lp_q_cur_given_star = tree_normal_log_prob(theta, mu_star, epsilon)

        # 8) Metropolis–Hastings acceptance ratio
        log_acceptance_ratio = (
            lp_theta_star + lp_q_cur_given_star
            - lp_theta    - lp_q_star_given_cur
        )

        # 9) Accept/reject
        u = jax.random.uniform(rng_accept, ())
        accept = (jnp.log(u) < log_acceptance_ratio)

        # 10) Return either theta_star or the old theta
        theta_next = jax.lax.cond(
            accept,
            lambda _: theta_star,
            lambda _: theta,
            operand=None
        )
        return theta_next, accept

    return fmala_step

def make_line_fmala_step(log_prob_fn, theta_example = None):
    """Create a JIT-compiled FMALA step that uses `log_prob_fn` internally."""

    if theta_example is not None:
        D = count_parameters(theta_example)
#         print("Using bias correction, but it is not needed for Line-FMALA.")
    else:
        D = 1.0
        print("Not implmementing correction")
    
    
    @jax.jit
    def fmala_line_step(theta, rng_key, epsilon):

        epsilon = epsilon * (D ** 0.5) # Unnecessary bias correction as just a step size scalar
        
        rng_key, rng_accept, rng_v_t = jax.random.split(rng_key, 3)
        
        # 0) Sample z ~ N(0,I)
        z = jax.random.normal(rng_key, shape=(1,))

        # 1) Sample v_t ~ N(0,I)
        v_t = tree_random_normal(rng_v_t, theta)
        v_t = tree_norm(v_t)

        # 2) Gradient wrt the real part of log_prob (if complex)
        lp_theta, jvp_t = jax.jvp(lambda th: log_prob_fn(th), (theta,), (v_t,))

        # 3) Propose theta_star
        theta_star = jax.tree.map(
            lambda t, v: t + (0.5*(epsilon**2)*jvp_t + epsilon*z) * v,
            theta, v_t
        )

        # 4) Gradient at theta_star
        lp_theta_star, jvp_star = jax.jvp(lambda th: log_prob_fn(th), (theta_star,), (v_t,))

        # 5) Drift terms mu_cur and mu_star
        alpha_t = tree_vdot(theta, v_t)
        alpha_star = tree_vdot(theta_star, v_t)
        mu_cur = alpha_t + 0.5 * (epsilon**2) * jvp_t
        mu_star = alpha_star + 0.5 * (epsilon**2) * jvp_star 

        # 6) Log-prob of Normal(...) with mean=mu, scale=epsilon
        def normal_log_prob(x, mean, scale):
            D = x.size
            return -0.5 * jnp.sum(((x - mean)/scale)**2) \
                   - D*jnp.log(scale * jnp.sqrt(2*jnp.pi))

        lp_q_star_given_cur = normal_log_prob(alpha_star, mu_cur, epsilon)
        lp_q_cur_given_star = normal_log_prob(alpha_t, mu_star, epsilon)

        # 8) Metropolis–Hastings acceptance ratio
        log_acceptance_ratio = (
            lp_theta_star + lp_q_cur_given_star
            - lp_theta    - lp_q_star_given_cur
        )

        # 9) Accept/reject
        u = jax.random.uniform(rng_accept, ())
        accept = (jnp.log(u) < log_acceptance_ratio)

        # 10) Return either theta_star or the old theta
        theta_next = jax.lax.cond(
            accept,
            lambda _: theta_star,
            lambda _: theta,
            operand=None
        )
        return theta_next, accept

    return fmala_line_step


def make_precon_fmala_step(log_prob_fn, theta_example = None):
    """Create a JIT-compiled precon FMALA step that uses `log_prob_fn` internally."""

    if theta_example is not None:
        D = count_parameters(theta_example)
    else:
        D = 1.0
        print("Not implmementing correction")
    
    @jax.jit
    def vhv(th, v1):
        """
        Nested forward-mode Hessian-like product:
        returns (fun_val, jvp_val, hqp_val).
        """
        # 1) First JVP in direction v1
        #    Here we define a lambda that calls loss_fn(params, x, y).
        def jvp_in_v1(w):
            return jax.jvp(lambda w_: log_prob_fn(w_),    # function of w_
                        (w,),                            # primal
                        (v1,))                           # tangent

        # 2) Then JVP of the result in direction v1
        (fun_val, jvp_val1), (jvp_val2, vHv_val) = jax.jvp(jvp_in_v1, (th,), (v1,))
        return fun_val, jvp_val1, vHv_val

    @jax.jit
    def precon_fmala_step(theta, rng_key, epsilon):
        rng_key, rng_accept, rng_v_t, rng_v_star = jax.random.split(rng_key, 4)
        
        # 0) Sample z ~ N(0,I)
        z = tree_random_normal(rng_key, theta)

        # 1) Sample v_t ~ N(0,I)
        v_t = tree_random_normal(rng_v_t, theta)
        v_t = tree_norm(v_t)

        # 2) Gradient wrt the log_prob
        lp_theta, jvp_t, vhv_t = vhv(theta, v_t)

        # 3) Propose theta_star
        theta_star = jax.tree.map(
            lambda t, v, zz: t + ((0.5 * (epsilon**2) * jvp_t) / jnp.abs(vhv_t)) *v + epsilon*zz/ ((D * jnp.abs(vhv_t))**0.5),
            theta, v_t, z
        )

        # 4) Gradient at theta_star
        v_star = tree_random_normal(rng_v_star, theta)
        v_star = tree_norm(v_star)
        lp_theta_star, jvp_star, vhv_star = vhv(theta_star, v_star)

        # 5) Drift terms mu_cur and mu_star
        mu_cur = jax.tree.map(
            lambda t, v: t + (( 0.5 * (epsilon**2) * jvp_t) / jnp.abs(vhv_t)) * v,
            theta, v_t
        )
        mu_star = jax.tree.map(
            lambda t, v: t + ((0.5 * (epsilon**2) * jvp_star) / jnp.abs(vhv_star)) *v,
            theta_star, v_star
        )

        # 6) Log-prob of Normal(...) with mean=mu, scale=epsilon
        lp_q_star_given_cur = tree_normal_log_prob(theta_star, mu_cur, epsilon / ((D * jnp.abs(vhv_t))**0.5))
        lp_q_cur_given_star = tree_normal_log_prob(theta, mu_star, epsilon / ((D * jnp.abs(vhv_star))**0.5))

        # 8) Metropolis–Hastings acceptance ratio
        log_acceptance_ratio = (
            lp_theta_star + lp_q_cur_given_star
            - lp_theta    - lp_q_star_given_cur
        )

        # 9) Accept/reject
        u = jax.random.uniform(rng_accept, ())
        accept = (jnp.log(u) < log_acceptance_ratio)

        # 10) Return either theta_star or the old theta
        theta_next = jax.lax.cond(
            accept,
            lambda _: theta_star,
            lambda _: theta,
            operand=None
        )
        return theta_next, accept

    return precon_fmala_step

def make_precon_line_fmala_step(log_prob_fn, theta_example = None):
    """Create a JIT-compiled precon FMALA step that uses `log_prob_fn` internally."""

    if theta_example is not None:
        D = count_parameters(theta_example)
#         print("Using bias correction, but it is not needed for PC-Line-FMALA.")
    else:
        D = 1.0
    
    @jax.jit
    def vhv(th, v1):
        """
        Nested forward-mode Hessian-like product:
        returns (fun_val, jvp_val, hqp_val).
        """
        # 1) First JVP in direction v1
        #    Here we define a lambda that calls loss_fn(params, x, y).
        def jvp_in_v1(w):
            return jax.jvp(lambda w_: log_prob_fn(w_),    # function of w_
                        (w,),                            # primal
                        (v1,))                           # tangent

        # 2) Then JVP of the result in direction v1
        (fun_val, jvp_val1), (jvp_val2, vHv_val) = jax.jvp(jvp_in_v1, (th,), (v1,))
        return fun_val, jvp_val1, vHv_val

    @jax.jit
    def precon_line_fmala_step(theta, rng_key, epsilon):
        rng_key, rng_accept, rng_v_t = jax.random.split(rng_key, 3)
        
        # 0) Sample z ~ N(0,I)
        z = jax.random.normal(rng_key, shape=(1,))

        # 1) Sample v_t ~ N(0,I)
        v_t = tree_random_normal(rng_v_t, theta)
        v_t = tree_norm(v_t)

        # 2) Gradient wrt the log_prob
        lp_theta, jvp_t, vhv_t = vhv(theta, v_t)

        # 3) Propose theta_star
        theta_star = jax.tree.map(
            lambda t, v: t + (((0.5 * (epsilon**2) * jvp_t) / jnp.abs(vhv_t)) + epsilon * z / (jnp.abs(vhv_t)**0.5)) * v,
            theta, v_t
        )

        # 4) Gradient at theta_star
        lp_theta_star, jvp_star, vhv_star = vhv(theta_star, v_t)

        # 5) Drift terms mu_cur and mu_star
        alpha_t = tree_vdot(theta, v_t)
        alpha_star = tree_vdot(theta_star, v_t)
        mu_cur = alpha_t + ( 0.5 * (epsilon**2) * jvp_t) / jnp.abs(vhv_t)
        mu_star = alpha_star + ( 0.5 * (epsilon**2) * jvp_star) / jnp.abs(vhv_star)

        # 6) Log-prob of Normal(...) with mean=mu, scale=epsilon
        def normal_log_prob(x, mean, scale):
            D = x.size
            return -0.5 * jnp.sum(((x - mean)/scale)**2) \
                   - D*jnp.log(scale * jnp.sqrt(2*jnp.pi))

        lp_q_star_given_cur = normal_log_prob(alpha_star, mu_cur, epsilon / (jnp.abs(vhv_t)**0.5))
        lp_q_cur_given_star = normal_log_prob(alpha_t, mu_star, epsilon / (jnp.abs(vhv_star)**0.5))

        # 8) Metropolis–Hastings acceptance ratio
        log_acceptance_ratio = (
            lp_theta_star + lp_q_cur_given_star
            - lp_theta    - lp_q_star_given_cur
        )

        # 9) Accept/reject
        u = jax.random.uniform(rng_accept, ())
        accept = (jnp.log(u) < log_acceptance_ratio)

        # 10) Return either theta_star or the old theta
        theta_next = jax.lax.cond(
            accept,
            lambda _: theta_star,
            lambda _: theta,
            operand=None
        )
        return theta_next, accept

    return precon_line_fmala_step