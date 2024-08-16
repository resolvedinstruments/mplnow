import time

import numpy as np
import matplotlib.pyplot as plt

from mplnow import FigDrawing

fig = FigDrawing()


def draw_plot(fig: FigDrawing, t: float):
    fig.begin()

    fig.set_layout_engine("constrained")
    axs = fig.subplots(2, 1, sharex=True)

    x = np.linspace(0, 5, 400)
    y = np.sin((x - 0.5 * t) * 3 * np.pi)
    y2 = np.cos((x + 0.1 * t) * 3 * np.pi)

    axs[0].curve(x, y, ls="-", color="C1")
    axs[0].axhline(1, ls="--", color="k")
    axs[0].axhline(-1, ls="--", color="k")

    axs[1].curve(x, y2, ls="-", color="C2")
    axs[1].axhline(0, ls="--", color="k")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_xlim(x[0], 5)

    fig.end()


while True:
    draw_plot(fig, time.time())
    plt.pause(0.1)
