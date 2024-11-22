# %%
#!%load_ext autoreload
#!%autoreload 2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ipywidgets

from mplnow import FigDrawing

matplotlib.use("module://ipympl.backend_nbagg")
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"


# %%
def interactive(w: float = 1.0):
    fig, ax = plt.subplots(1, 1, layout="constrained")
    ax.plot(np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100) * w))
    fig.show()


ipywidgets.interactive(interactive, w=(0.1, 10.0, 0.1))

# %%
figd = FigDrawing()
figd.fig.show()


def interactive_mplnow(w: float = 1.0):
    with figd.begin():
        (ax,) = figd.subplots(1, 1)
        ax.plot(np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100) * w))


ipywidgets.interactive(interactive_mplnow, w=(0.1, 10.0, 0.1))

# %%
