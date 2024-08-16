from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence, override
import logging
from contextlib import ExitStack

import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

log = logging.getLogger(__name__)

type UpdateCallback[T] = Callable[[T, dict[str, Any]], None]
type RemoveCallback[T] = Callable[[T], None]
type CreateCallback[T] = Callable[[], T]


@dataclass(frozen=True)
class ArtifactState[T]:
    # store name for additional checks of state mismatch
    creator_name: str
    # If deps change, the artifact is removed and recreated
    deps: dict[str, Any]
    artifact: T
    remover: RemoveCallback[T]
    # Cache for update function use, e.g. for custom checks if data has changed
    update_cache: dict[str, Any] = field(default_factory=dict)


class Drawing:
    def __init__(self):
        self._state: dict[int, ArtifactState] = {}
        self._iter_index: int = 0
        self._state_mismatch: bool = False
        self._drawstack: ExitStack | None = None

    def begin(self):
        if self._drawstack:
            log.error(f"begin(), but drawing already started. ({self})")
        self._drawstack = ExitStack()

        self._iter_index = 0

        @self._drawstack.callback
        def on_end():
            if self._iter_index != len(self._state):
                keys_to_delete = [
                    key for key in self._state.keys() if key >= self._iter_index
                ]

                for key in keys_to_delete:
                    state = self._state[key]
                    state.remover(state.artifact)
                    del self._state[key]

                log.warning(
                    f"artifacts left over - {len(keys_to_delete)} removed ({self})"
                )

            self._drawstack = None

        return self._drawstack

    def end(self):
        if self._drawstack:
            self._drawstack.close()
        else:
            log.error(f"end() called but drawing not in progress. {self}")

    def _get_exitstack(self):
        if self._drawstack is None:
            raise RuntimeError("Render not started.")
        return self._drawstack

    def _use_artifact[
        T
    ](
        self,
        name: str,
        deps: dict[str, Any],
        create: CreateCallback[T],
        update: UpdateCallback[T] = lambda x, y: None,
        remove: RemoveCallback[T] = lambda x: None,
    ) -> T:
        key = self._iter_index
        self._iter_index += 1

        state = self._state.get(key, None)

        if state:
            if state.creator_name != name:
                log.warning(f"artifact creator mismatch - recreating ({self})")
                state.remover(state.artifact)
            if state.deps != deps:
                log.debug(f"recreating artifact ({self})")
                state.remover(state.artifact)
            else:
                update(state.artifact, state.update_cache)
                return state.artifact

        state = ArtifactState[T](
            creator_name=name,
            deps=deps,
            artifact=create(),
            remover=remove,
        )
        self._state[key] = state
        return state.artifact


class AxesDrawing(Drawing):
    def __init__(self, ax: Axes):
        super().__init__()
        self.ax = ax

    def curve(self, x: ArrayLike, y: ArrayLike, **kwargs) -> Line2D:
        def create() -> Line2D:
            (line,) = self.ax.plot(x, y, **kwargs)
            return line

        def update(line: Line2D, cache: dict[str, Any]) -> None:
            if cache.get("x", None) is not x:
                line.set_xdata(x)
                cache["x"] = x

            if cache.get("y", None) is not y:
                line.set_ydata(y)
                cache["y"] = y

        def remove(artifact: Line2D):
            artifact.remove()

        return self._use_artifact("curve", kwargs, create, update, remove)

    def axhline(
        self, y: float = 0, xmin: float = 0, xmax: float = 1, **kwargs
    ) -> Line2D:
        def create():
            artifact = self.ax.axhline(y, xmin, xmax, **kwargs)
            return artifact

        def update(artifact: Line2D, cache):
            if cache.get("y", None) is not y:
                artifact.set_ydata([y, y])
                cache["y"] = y

        def remove(artifact: Line2D):
            artifact.remove()

        deps = dict(y=y, xmin=xmin, xmax=xmax, **kwargs)
        return self._use_artifact("axhline", deps, create, update, remove)

    def set_xlabel(self, xlabel: str, **kwargs) -> None:
        def create():
            self.ax.set_xlabel(xlabel, **kwargs)

        deps = dict(xlabel=xlabel, **kwargs)
        self._use_artifact("set_xlabel", deps, create)

    def set_ylabel(self, ylabel: str, **kwargs) -> None:
        def create():
            self.ax.set_ylabel(ylabel, **kwargs)

        deps = dict(ylabel=ylabel, **kwargs)
        self._use_artifact("set_ylabel", deps, create)

    def set_xlim(
        self, left=None, right=None, *, emit=True, auto=False, xmin=None, xmax=None
    ):

        def create():
            self.ax.set_xlim(left, right, emit=emit, auto=auto, xmin=xmin, xmax=xmax)

        deps = dict(left=left, right=right, emit=emit, auto=auto, xmin=xmin, xmax=xmax)
        self._use_artifact("set_xlim", deps, create)


class FigDrawing(Drawing):
    def __init__(self, fig: Figure | None = None):
        super().__init__()

        if fig is None:
            fig = plt.figure()
            plt.show(block=False)

        self.fig = fig

    @override
    def begin(self):
        exitstack = super().begin()

        exitstack.enter_context(plt.ioff())

        @exitstack.callback
        def on_end():
            self.fig.canvas.draw()

        return exitstack

    def subplots(
        self,
        nrows: int = 1,
        ncols: int = 1,
        *,
        sharex: bool | Literal["none", "all", "row", "col"] = False,
        sharey: bool | Literal["none", "all", "row", "col"] = False,
        squeeze: bool = True,
        width_ratios: Sequence[float] | None = None,
        height_ratios: Sequence[float] | None = None,
        subplot_kw: dict[str, Any] | None = None,
        gridspec_kw: dict[str, Any] | None = None,
        **fig_kw,
    ) -> list[AxesDrawing]:
        deps = dict(
            nrows=nrows,
            ncols=ncols,
            sharex=sharex,
            sharey=sharey,
            squeeze=squeeze,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
            subplot_kw=subplot_kw,
            gridspec_kw=gridspec_kw,
            **fig_kw,
        )

        def begin_axes(axes: list[AxesDrawing]):
            exitstack = self._get_exitstack()

            for ax in axes:
                exitstack.enter_context(ax.begin())

        def create() -> list[AxesDrawing]:
            axes_collection = self.fig.subplots(
                nrows=nrows,
                ncols=ncols,
                sharex=sharex,
                sharey=sharey,
                squeeze=squeeze,
                subplot_kw=subplot_kw,
                gridspec_kw=gridspec_kw,
                **fig_kw,
            )

            axes_list: list[Axes] = []
            if not isinstance(axes_collection, np.ndarray):
                axes_list = [axes_collection]
            else:
                axes_list = axes_collection.flatten().tolist()

            axes_wrappers = [AxesDrawing(ax) for ax in axes_list]
            begin_axes(axes_wrappers)
            return axes_wrappers

        def update(axes: list[AxesDrawing], _cache):
            begin_axes(axes)

        def remove(axes: list[AxesDrawing]):
            for axw in axes:
                axw.ax.remove()

        return self._use_artifact("subplots", deps, create, update, remove)

    def set_layout_engine(self, layout=None, **kwargs):
        def create():
            self.fig.set_layout_engine(layout, **kwargs)

        deps = dict(layout=layout, **kwargs)
        self._use_artifact("set_layout_engine", deps, create)
