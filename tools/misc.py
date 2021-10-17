import gpflow
import numpy as np
from .lmflow import BasisFunctionKernel


def jit_chol(K):
    jitter_level = 1e-6
    for _ in range(10):
        try:
            return np.linalg.cholesky(K + jitter_level * np.eye(len(K)))
        except np.linalg.LinAlgError:
            jitter_level *= 10


def sample_prior(model_or_kernel, pX, num_samples=10):
    k = model_or_kernel if isinstance(model_or_kernel, gpflow.kernels.Kernel) else model_or_kernel.kernel
    if not isinstance(k, BasisFunctionKernel):
        K = k(pX)
        L = jit_chol(K)
        samples = L @ np.random.randn(len(L), num_samples)
    else:
        Phi = k.Phi(pX)
        samples = Phi @ np.random.randn(Phi.shape[1], num_samples)
    return samples
