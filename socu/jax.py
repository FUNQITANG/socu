import warp as wp
from warp.jax_experimental.ffi import jax_callable
import jax.numpy as jnp

from socu.block_tridiag_solver import calculate_off_diag_storage_len, optimal_problem_settings, \
                                      create_cholesky_factor_launch, create_cholesky_solve_launch, \
                                      create_cholesky_factor_and_solve_launch


def to_wp_dtype(dtype: jnp.dtype):
    if dtype == jnp.float32:
        return wp.float32
    elif dtype == jnp.float64:
        return wp.float64
    else:
        raise RuntimeError('dtype must be float32 or float64')


def ensure_E_size(horizon: int, E: jnp.ndarray):
    E_elements = calculate_off_diag_storage_len(horizon)
    if E.shape[0] < E_elements:
        return jnp.pad(E,
                       pad_width=((0, E_elements - E.shape[0]), (0, 0), (0, 0)),
                       mode='constant',
                       constant_values=0.0)
    else:
        return E


def pad_factor_data(L: jnp.ndarray, E: jnp.ndarray, b: jnp.ndarray = None, factor: bool = True):
    n = L.shape[1]
    opt_settings = optimal_problem_settings(n, to_wp_dtype(L.dtype))
    if factor:
        pad_multiple = opt_settings['pad_multiple']['factor']
    else:
        pad_multiple = opt_settings['pad_multiple']['solve']
    n_padded = ((n + pad_multiple - 1) // pad_multiple) * pad_multiple # round up to next multiple of pad_multiple

    if n_padded != n:
        pad_width = n_padded - n

        L = jnp.pad(L,
                    pad_width=((0, 0), (0, pad_width), (0, pad_width)),
                    mode='constant',
                    constant_values=0.0)
        identity_pad = jnp.eye(n_padded, dtype=L.dtype)
        identity_pad = identity_pad.at[:n, :n].set(0)
        L = L + identity_pad[None, :, :]

        E = jnp.pad(E,
                    pad_width=((0, 0), (0, pad_width), (0, pad_width)),
                    mode='constant',
                    constant_values=0.0)
        
        if b is not None:
            b = jnp.pad(b,
                        pad_width=((0, 0), (0, pad_width), (0, 0)),
                        mode='constant',
                        constant_values=0.0)
    
    return L, E, b


def _cholesky_factor_and_solve(
    L: jnp.ndarray, # (N, n, n)
    E: jnp.ndarray, # (N - 1, n, n)
    b: jnp.ndarray = None, # (N, n, n_rhs)
    pad_problem: bool = True,
    factor: bool = True,
):
    if L.ndim != 3:
        raise ValueError(f"L must be 3-dimensional, got shape {L.shape}")
    if E.ndim != 3:
        raise ValueError(f"E must be 3-dimensional, got shape {E.shape}")
    
    N, n, n_check = L.shape
    if n != n_check:
        raise ValueError(f"L must be square in last two dimensions, got shape {L.shape}")
    
    _, E_n, E_n_check = E.shape
    if E_n != n or E_n_check != n:
        raise ValueError(f"E dimensions must match L's n={n}, got E shape {E.shape}")
    
    if b is not None:
        if b.ndim != 3:
            raise ValueError(f"b must be 3-dimensional, got shape {b.shape}")
        
        b_N, b_n, _ = b.shape
        if b_N != N:
            raise ValueError(f"b must have shape ({N}, {n}, n_rhs), got {b.shape}")
        if b_n != n:
            raise ValueError(f"b second dimension must be {n}, got {b.shape}")
        
        if L.dtype != b.dtype:
            raise ValueError(f"L and b must have same dtype, got L: {L.dtype}, b: {b.dtype}")
    
    if L.dtype != E.dtype:
        raise ValueError(f"L and E must have same dtype, got L: {L.dtype}, E: {E.dtype}")
    

    dtype = to_wp_dtype(L.dtype)
    horizon = L.shape[0]
    n = L.shape[1]

    E = ensure_E_size(horizon, E)

    if pad_problem:
        L, E, b = pad_factor_data(L, E, b, factor=factor)

    if factor == False:
        assert b is not None

        def func(L_wp: wp.array3d(dtype=dtype), E_wp: wp.array3d(dtype=dtype), x_wp: wp.array3d(dtype=dtype)): # type: ignore
            create_cholesky_solve_launch(L_wp, E_wp, x_wp, dtype=dtype)()

        jax_func = jax_callable(func, num_outputs=1, in_out_argnames=['x_wp'])
        x = jax_func(L, E, b)[0]

        return x[:, :n, :]
    
    if b is None:
        def func(L_wp: wp.array3d(dtype=dtype), E_wp: wp.array3d(dtype=dtype)): # type: ignore
            create_cholesky_factor_launch(L_wp, E_wp, dtype=dtype)()

        jax_func = jax_callable(func, num_outputs=2, in_out_argnames=['L_wp', 'E_wp'])
        L, E = jax_func(L, E)

        return L[:, :n, :n], E[:, :n, :n]
    
    else:
        def func(L_wp: wp.array3d(dtype=dtype), E_wp: wp.array3d(dtype=dtype), x_wp: wp.array3d(dtype=dtype)): # type: ignore
            create_cholesky_factor_and_solve_launch(L_wp, E_wp, x_wp, dtype=dtype)()

        jax_func = jax_callable(func, num_outputs=3, in_out_argnames=['L_wp', 'E_wp', 'x_wp'])
        L, E, x = jax_func(L, E, b)

        return L[:, :n, :n], E[:, :n, :n], x[:, :n, :]


def cholesky_factor(
    L: jnp.ndarray, # (N, n, n)
    E: jnp.ndarray, # (N - 1, n, n)
    pad_problem: bool = True,
):
    return _cholesky_factor_and_solve(L, E, pad_problem=pad_problem, factor=True)


def cholesky_solve(
    L: jnp.ndarray, # (N, n, n)
    E: jnp.ndarray, # (N - 1, n, n)
    b: jnp.ndarray, # (N, n, n_rhs)
    pad_problem: bool = True,
):
    return _cholesky_factor_and_solve(L, E, b, pad_problem=pad_problem, factor=False)


def cholesky_factor_and_solve(
    L: jnp.ndarray, # (N, n, n)
    E: jnp.ndarray, # (N - 1, n, n)
    b: jnp.ndarray, # (N, n, n_rhs)
    pad_problem: bool = True,
):
    return _cholesky_factor_and_solve(L, E, b, pad_problem=pad_problem, factor=True)
