import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def plot_1d_model(m, data=None, plot_mean=True, plot_var="y", plot_samples=False, plot_samples_z=None, ax=None, pX=None):
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


def model_training_gif(m, hist, lml_plot=True, plot_var=True, axes_setup=lambda ax: None, **kwargs):
    param_hist = hist.iloc[:, ['GP' in v for v in hist.columns]]
    lml_hist = hist['lml']

    if lml_plot:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7, 2))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7, 2))
    axes_setup(ax)

    data_inducingpts = np.vstack((m.X.value, m.feature.Z.value)) if hasattr(m, 'feature') else m.X.value
    pX = np.linspace(np.min(data_inducingpts) - 1.0, np.max(data_inducingpts) + 1.0, 2000)[:, None]
    pY, pYv = m.predict_y(pX)
    ax.plot(m.X.value, m.Y.value, 'x')
    pred_mean_line = ax.plot(pX, pY, lw=1.5)
    col = pred_mean_line[0].get_color()
    if plot_var:
        ax.fill_between(pX.flatten(), (pY - 2 * pYv ** 0.5).flatten(), (pY + 2 * pYv ** 0.5).flatten(), alpha=0.3)
    if hasattr(m, 'feature'):
        Z_line = ax.plot(m.feature.Z.value, np.zeros(m.feature.Z.value.shape), 'k|', mew=2)

    if lml_plot:
        text = ax2.text(0.99, 0.99, 'hello', horizontalalignment='right', verticalalignment='top',
                        transform=ax2.transAxes,
                        fontsize=8)
        ax2.plot(lml_hist)
        location_marker = ax2.plot([0], lml_hist[0], 'o')

    def init():
        lines = pred_mean_line
        if lml_plot:
            lines = lines + [text] + location_marker
        if hasattr(m, 'feature'):
            lines = lines + Z_line
        return lines

    def animate(i):
        params = param_hist.iloc[i % len(param_hist)]
        param_name_object_resolve = {p.pathname: p.constrained_tensor for p in m.parameters}
        feed_dict = {param_name_object_resolve[k]: v for k, v in params.items()}

        lml = (m.compute_log_likelihood(feed_dict=feed_dict) if not hasattr(m, 'q_T_sqrt') else
               m.compute_exact_log_likelihood(feed_dict=feed_dict))
        if lml_plot:
            text.set_text('ELBO:{0:.2f}'.format(lml))

        pY, pYv = m.predict_y(pX, feed_dict=feed_dict)
        pred_mean_line[0].set_data(pX, pY)
        # pred_topvar_line[0].set_data(pX, pY + 2 * pYv ** 0.5)
        # pred_botvar_line[0].set_data(pX, pY - 2 * pYv ** 0.5)
        if plot_var:
            ax.collections.clear()
            ax.fill_between(pX.flatten(), (pY - 2 * pYv ** 0.5).flatten(), (pY + 2 * pYv ** 0.5).flatten(), alpha=0.3,
                            color='C0')
        if hasattr(m, 'feature'):
            Z_line[0].set_xdata(params[list(params.keys())[0].split('/')[0] + '/feature/Z'])

        if lml_plot:
            location_marker[0].set_data([i], [lml_hist[i]])

        print(i, i % len(param_hist), lml, "                   ")
        sys.stdout.flush()
        lines = pred_mean_line
        if lml_plot:
            lines = lines + [text] + location_marker
        if hasattr(m, 'feature'):
            lines = lines + Z_line
        return lines

    animation_kwargs = {"repeat": False, "interval": 40}
    animation_kwargs.update(kwargs)
    ani = animation.FuncAnimation(fig, animate, init_func=init, blit=True, save_count=len(param_hist),
                                  frames=len(param_hist), **animation_kwargs)
    return ani
