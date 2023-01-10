import scipy.io
import numpy as np
from gpflow.models import GPR
from gpflow.kernels import Periodic, SquaredExponential
from gpflow.optimizers import Scipy
import matplotlib.pyplot as plt

np.random.seed(0)


def plot_1d_model(m, data=None, plot_mean=True, plot_var="y", plot_samples=False, plot_samples_z=None, ax=None,
                  pX=None):
    if ax is None:
        ax = plt.gca()
    if data is None:
        if hasattr(m, 'data'):
            X, Y = [d.numpy() for d in m.data]
        else:
            X, Y = None, None
    else:
        X, Y = data[0], data[1]

    if pX is None:
        data_inducingpts = np.vstack((X, m.inducing_variable.Z.numpy())) if hasattr(m, "inducing_variable") else X
        range = np.max(data_inducingpts) - np.min(data_inducingpts)
        pX = np.linspace(np.min(data_inducingpts) - range / 3, np.max(data_inducingpts) + range / 3, 1800)[:, None]

    #
    # Predicting
    predict_stats = m.predict_y(pX) if plot_var == "y" else m.predict_f(pX)
    pY, pYv = [d.numpy() for d in predict_stats]

    #
    # Plotting
    if plot_mean:
        line, = ax.plot(pX, pY, lw=1.5, label="mean")
    if X is not None:
        ax.plot(X, Y, 'x', color='C1')
    if plot_var is not False:
        ax.fill_between(pX.flatten(), (pY - 2 * pYv ** 0.5).flatten(), (pY + 2 * pYv ** 0.5).flatten(), alpha=0.3,
                        label="2$\sigma$ func" if plot_var == "f" else "2$\sigma$ data")
    if plot_samples:
        if plot_samples_z is None:
            plot_samples_z = np.random.randn(len(pX), 10)
        mu, cov = m.predict_f(pX, full_cov=True)

        chol, jitter = None, 1e-8
        while chol is None:
            try:
                chol = np.linalg.cholesky(cov[0, :, :] + jitter * np.eye(cov.shape[1]))
            except np.linalg.LinAlgError:
                jitter *= 10
                # print(jitter)
        f_samples = mu + chol @ plot_samples_z
        # m.predict_f_samples(num_samples=10)
        ax.plot(pX.flatten(), f_samples, color='C0', alpha=0.4)

    # plt.plot(pX, pY + 2 * pYv ** 0.5, col, lw=1.5)
    # plt.plot(pX, pY - 2 * pYv ** 0.5, col, lw=1.5)
    if hasattr(m, 'inducing_variable'):
        # ax.plot(m.inducing_variable.Z.numpy(), np.zeros(m.inducing_variable.Z.numpy().shape), 'k|', mew=2)
        if hasattr(m, 'q_mu'):
            ax.plot(m.inducing_variable.Z.numpy(), m.q_mu.numpy(), 'k|', mew=2)
        else:
            q_mu, _ = m.predict_f(m.inducing_variable.Z.numpy())
            ax.plot(m.inducing_variable.Z.numpy(), q_mu, 'k|', mew=2)


d = scipy.io.loadmat("./cw1a.mat")
lX, lY = [d[k] for k in ["x", "y"]]
take_dataset_items = np.hstack((np.random.permutation(len(lX))[:25],
                                np.argwhere(np.logical_or(lX > 1.5, lX < -1.5).flatten()).flatten()))
take_dataset_items = np.unique(take_dataset_items)

# take_dataset_items = take_dataset_items[np.logical_and(take_dataset_items != 4, take_dataset_items != 6)]
lX = lX[take_dataset_items, :]
lY = lY[take_dataset_items, :]

m = GPR((lX, lY), Periodic(SquaredExponential()))
Scipy().minimize(m.training_loss_closure(compile=True), m.trainable_variables)

# X = lX + np.random.standard_cauchy(lX.shape) * 0.0
# K = m.kernel(X) + np.eye(len(X)) * m.likelihood.variance.numpy()
#
# Y = np.linalg.cholesky(K) @ np.random.randn(len(X), 1)
# Y = Y - Y.mean()
X = lX
Y = lY

m2 = GPR((X, Y), Periodic(SquaredExponential()))
Scipy().minimize(m2.training_loss_closure(compile=True), m2.trainable_variables)

tX = np.linspace(-3, 3, 200)[:, None]
tY = m2.predict_f_samples(tX).numpy() + np.random.randn(len(tX), 1) * m2.likelihood.variance.numpy() ** 0.5
# plt.plot(tX, tY, 'x')
plot_1d_model(m2)

scipy.io.savemat("periodic1d.mat", dict(X=X, Y=Y, tX=tX, tY=tY, description="Periodic 1D dataset.", name="periodic1d",
                                        url="https://github.com/markvdw/inference-plots"))

np.random.seed(1)

X, Y = d['x'], d['y']
take_dataset_items = np.hstack((np.random.permutation(len(X))[:25],
                                np.argwhere(np.logical_or(X > 1.5, X < -1.5).flatten()).flatten()))
# take_dataset_items = take_dataset_items[np.logical_and(take_dataset_items != 43, take_dataset_items != 6)]
# take_dataset_items = take_dataset_items[take_dataset_items != 43]
X = X[take_dataset_items, :]
Y = Y[take_dataset_items, :]

sort = np.argsort(X.flatten())
X = X[sort, :]
Y = Y[sort, :]

lim = 1.5
Y = Y[np.logical_and(X > -lim, X < lim)][:, None]
X = X[np.logical_and(X > -lim, X < lim)][:, None]

scipy.io.savemat("periodic1d-subset.mat", dict(X=X, Y=Y, tX=tX, tY=tY, description="Periodic 1D dataset (subset).",
                                               name="periodic1d-subset",
                                               url="https://github.com/markvdw/inference-plots"))
