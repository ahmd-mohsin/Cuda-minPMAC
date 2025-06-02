import jax, jax.numpy as jnp
from jaxopt import OSQP

def solve_min_energy_mac(H, b_min, w):
    """
    Solve the approximate MAC power-minimization problem.
    H: array of shape (N, L_y, sum(Lx)), channel matrices per tone.
    b_min: array of shape (U,), required rates per user.
    w: array of shape (U,), weight per user in objective.
    """
    U = b_min.shape[0]
    N = H.shape[0]
    H_split = jnp.split(H, indices_or_sections=U, axis=2)  
    G = []
    for H_u in H_split:  # each H_u is (N, L_y, Lx_u)
        # compute Frobenius norm squared of H_u for each tone n
        G.append(jnp.sum(H_u**2, axis=(1,2)))  # shape (N,)
    G = (1/jnp.log(2)) * jnp.stack(G, axis=1)  # shape (N, U)
    # Decision variable vector x = [P(0,1..U), ..., P(N-1,1..U), R(0,1..U), ..., R(N-1,1..U)]
    # Objective coefficients
    c_P = jnp.tile(w, N)        # each tone has w for P variables
    c_R = jnp.zeros(N*U)        # no cost for R
    c = jnp.concatenate([c_P, c_R])  # linear objective
    # Equality constraints: sum_n R_{u,n} = b_min[u]
    A_eq = jnp.zeros((U, 2*N*U))
    for u in range(U):
        # indices of R for user u across all tones:
        idx = N*U + u + jnp.arange(N)*U
        A_eq = A_eq.at[u, idx].set(1)
    b_eq = b_min
    # Inequality constraints: for each tone n and subset S: sum_{u in S} R_{u,n} <= sum_{u in S} G[n,u] P_{u,n}
    # We enforce a subset of these or all for small U
    A_ineq_list = []
    for n in range(N):
        # to reduce number of constraints, we enforce constraints only for cumulative sets in sorted order of gains (optional)
        # Here, do all subsets for illustration:
        for mask in range(1, 1<<U):
            coeff = jnp.zeros(2*N*U)
            for u in range(U):
                if mask & (1<<u):
                    coeff = coeff.at[N*U + n*U + u].set(1.0)                     # +1 * R_{u,n}
                    coeff = coeff.at[n*U + u].set(coeff[n*U + u] - G[n,u])       # -G[n,u] * P_{u,n}
            A_ineq_list.append(coeff)
    A_ineq = jnp.stack(A_ineq_list, axis=0)
    l_ineq = -jnp.inf * jnp.ones(A_ineq.shape[0])
    u_ineq = jnp.zeros(A_ineq.shape[0])
    # Solve QP (with zero quadratic part, OSQP can handle that)
    osqp = OSQP()
    sol = osqp.run(None, params_obj=(jnp.zeros_like(c), c), 
                   params_eq=(A_eq, b_eq), params_ineq=(A_ineq, l_ineq, u_ineq))
    x_opt = sol.params
    P_opt = x_opt[:N*U].reshape((N, U))
    R_opt = x_opt[N*U:].reshape((N, U))
    return P_opt, R_opt

# --------------  fixed test data  --------------
U  = 3
Lx = [1,1,1]
Ly = 2
N  = 2
w  = jnp.array([1.0, 0.8, 1.2])
b_min = jnp.array([1.0, 0.7, 0.4])

H = jnp.zeros((N, Ly, sum(Lx)), dtype=jnp.complex64)
H = H.at[0,:,:].set(jnp.array([[ 0.64+0.22j, -0.31+0.57j,  0.48-0.36j],
                               [-0.11+0.72j,  0.95-0.13j, -0.27-0.65j]], dtype=jnp.complex64))
H = H.at[1,:,:].set(jnp.array([[-0.52+0.61j,  0.07-0.84j, -0.67-0.25j],
                               [ 0.29+0.41j,  0.39+0.12j, -0.18+0.59j]], dtype=jnp.complex64))

# ----- call your Python solver -----
P_opt, R_opt = solve_min_energy_mac(H, b_min, w)
print(P_opt)
print(R_opt)
