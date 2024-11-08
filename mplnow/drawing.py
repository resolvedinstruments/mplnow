from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence, override, Self, Iterable
import logging
from contextlib import ExitStack

import numpy as np
from numpy.typing import ArrayLike
from matplotlib import pyplot
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection, PathCollection
from matplotlib.artist import Artist
from matplotlib.legend import Legend

log = logging.getLogger(__name__)

type UpdateCallback[T] = Callable[[T, dict[str, Any]], None]
type RemoveCallback[T] = Callable[[T], None]
type CreateCallback[T] = Callable[[], T]


@dataclass(frozen=True)
class ArtifactState[T]:
    # store name for additional checks of state mismatch
    creator_name: str
    # If deps change, the artifact is removed and recreated
    deps: dict[str, Any] | list[Any]
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
            log.warning(f"begin(), but drawing already started. ({self})")
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
            log.warning(f"end() called but drawing not in progress. {self}")

    def _get_drawstack(self):
        if self._drawstack is None:
            raise RuntimeError("Render not started.")
        return self._drawstack

    def _use_artifact[
        T
    ](
        self,
        name: str,
        deps: dict[str, Any] | list[Any],
        create: CreateCallback[T],
        update: UpdateCallback[T] = lambda _x, _y: None,
        remove: RemoveCallback[T] = lambda _x: None,
    ) -> T:
        key = self._iter_index
        self._iter_index += 1

        state = self._state.get(key, None)

        if state:
            if state.creator_name != name:
                log.warning(f"artifact creator mismatch - recreating ({self})")
                state.remover(state.artifact)
            elif state.deps != deps:
                log.debug(f"recreating artifact ({self})")
                state.remover(state.artifact)
            else:
                update(state.artifact, state.update_cache)
                return state.artifact

        try:
            artifact = create()
        except Exception as e:
            # remove artifact from state so remover always knows create() finished successfully
            if key in self._state:
                del self._state[key]
            raise e

        self._state[key] = ArtifactState[T](name, deps, artifact, remove)
        return artifact

    def _remove_all(self):
        for key, state in self._state.items():
            state.remover(state.artifact)

        self._state.clear()


class AxesDrawing(Drawing):
    def __init__(self, ax: Axes):
        super().__init__()
        self.ax = ax

        # Keep track of legend artists to trigger legend updates
        self._legend_deps: list[Artist] = []

    @override
    def begin(self):
        cm = super().begin()

        # reset color cycle
        self.ax.set_prop_cycle(None)  # type: ignore

        return cm

    def plot(self, x: ArrayLike, y: ArrayLike, fmt: str = "", **kwargs) -> list[Line2D]:
        def create() -> list[Line2D]:
            if fmt:
                lines = self.ax.plot(x, y, fmt, **kwargs)
            else:
                lines = self.ax.plot(x, y, **kwargs)

            self._legend_deps += lines
            return lines

        def update(lines: list[Line2D], cache: dict[str, Any]) -> None:
            fmt_colors = "bgrcmykw"
            if not (
                "color" in kwargs or "c" in kwargs or any(c in fmt_colors for c in fmt)
            ):
                # cycle color for lines
                self.ax._get_lines.get_next_color()  # type: ignore

            if cache.get("x", None) is not x:
                lines[0].set_xdata(x)
                cache["x"] = x

            if cache.get("y", None) is not y:
                lines[0].set_ydata(y)
                cache["y"] = y

        def remove(lines: list[Line2D]):
            for line in lines:
                self._legend_deps.remove(line)
                line.remove()

        deps = [fmt, kwargs]
        return self._use_artifact("plot", deps, create, update, remove)

    def scatter(
        self,
        x: ArrayLike,
        y: ArrayLike,
        s: ArrayLike | None = None,
        c: ArrayLike | None = None,
        **kwargs,
    ) -> PathCollection:
        """
        A scatter plot of *y* vs. *x* with varying marker size and/or color.
        """

        def create() -> Any:
            collection = self.ax.scatter(x, y, **kwargs)
            self._legend_deps.append(collection)
            return collection

        def update(collection: Any, cache: dict[str, Any]) -> None:
            offsets = np.column_stack([x, y])
            if cache.get("offsets", None) is not offsets:
                collection.set_offsets(offsets)
                cache["offsets"] = offsets

        def remove(collection: Any) -> None:
            self._legend_deps.remove(collection)
            collection.remove()

        deps = kwargs
        return self._use_artifact("scatter", deps, create, update, remove)

    def axhline(
        self,
        y: float = 0,
        xmin: float = 0,
        xmax: float = 1,
        **kwargs,
    ) -> Line2D:
        def create():
            artifact = self.ax.axhline(y, xmin, xmax, **kwargs)
            self._legend_deps.append(artifact)
            return artifact

        def update(artifact: Line2D, cache: dict[str, Any]):
            if cache.get("y", None) != y:
                artifact.set_ydata([y, y])
                cache["y"] = y

        def remove(artifact: Line2D):
            self._legend_deps.remove(artifact)
            artifact.remove()

        deps = dict(xmin=xmin, xmax=xmax, **kwargs)
        return self._use_artifact("axhline", deps, create, update, remove)

    def axvline(
        self,
        x: float = 0,
        ymin: float = 0,
        ymax: float = 1,
        **kwargs,
    ) -> Line2D:
        def create():
            artifact = self.ax.axvline(x, ymin, ymax, **kwargs)
            self._legend_deps.append(artifact)
            return artifact

        def update(artifact: Line2D, cache: dict[str, Any]):
            if cache.get("x", None) != x:
                artifact.set_xdata([x, x])
                cache["x"] = x

        def remove(artifact: Line2D):
            self._legend_deps.remove(artifact)
            artifact.remove()

        deps = dict(ymin=ymin, ymax=ymax, **kwargs)
        return self._use_artifact("axvline", deps, create, update, remove)

    def fill_between(
        self,
        x,
        y1,
        y2=0,
        where=None,
        interpolate=False,
        step=None,
        **kwargs,
    ):
        def create():
            artifact = self.ax.fill_between(
                x, y1, y2=y2, where=where, interpolate=interpolate, step=step, **kwargs
            )
            self._legend_deps.append(artifact)
            return artifact

        def update(artifact: PolyCollection, cache):
            pass

        def remove(artifact: PolyCollection):
            self._legend_deps.remove(artifact)
            artifact.remove()

        deps = dict(
            x=x, y1=y1, y2=y2, where=where, interpolate=interpolate, step=step, **kwargs
        )
        return self._use_artifact("fill_between", deps, create, update, remove)

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

    def tick_params(self, axis: Literal["both", "x", "y"] = "both", **kwargs):
        def create():
            self.ax.tick_params(axis=axis, **kwargs)

        deps = dict(axis=axis, **kwargs)
        self._use_artifact("tick_params", deps, create)

    def set_xlim(self, left: float, right: float, *, emit=True, auto=False):
        def create():
            self.ax.set_xlim(left, right, emit=emit, auto=auto)

        deps = dict(left=left, right=right, emit=emit, auto=auto)
        self._use_artifact("set_xlim", deps, create)

    def set_ylim(self, bottom: float, top: float, *, emit=True, auto=False):
        def create():
            self.ax.set_ylim(bottom, top, emit=emit, auto=auto)

        deps = dict(bottom=bottom, top=top, emit=emit, auto=auto)
        self._use_artifact("set_ylim", deps, create)

    def twinx(self) -> Self:
        drawstack = self._get_drawstack()

        def create() -> Self:
            ax_ = self.ax.twinx()
            ax = AxesDrawing(ax_)
            drawstack.enter_context(ax.begin())
            return ax

        def update(ax: Self, _cache):
            drawstack.enter_context(ax.begin())

        def remove(ax: Self):
            ax.end()
            ax.ax.remove()

        return self._use_artifact("twinx", {}, create, update, remove)

    def set_title(
        self,
        label: str,
        fontdict: dict | None = None,
        loc: Literal["center", "left", "right"] | None = None,
        pad: float | None = None,
        *,
        y: float | None = None,
        **kwargs,
    ):
        def create():
            self.ax.set_title(label, fontdict=fontdict, loc=loc, pad=pad, y=y, **kwargs)

        deps = dict(label=label, fontdict=fontdict, loc=loc, pad=pad, y=y, **kwargs)
        self._use_artifact("set_title", deps, create)

    def autoscale(
        self,
        enable: bool = True,
        axis: Literal["both", "x", "y"] = "both",
        tight: bool | None = None,
    ):
        def create():
            self.ax.autoscale(enable=enable, axis=axis, tight=tight)

        def update(_artifact, _cache):
            if enable:
                self.ax.relim()
                self.ax.autoscale(enable=enable, axis=axis, tight=tight)

        deps = [enable, axis, tight]
        self._use_artifact("autoscale", deps, create, update)

    def legend(
        self,
        handles: Iterable[Artist | tuple[Artist, ...]] | None = None,
        labels: Iterable[str] | None = None,
        **kwargs,
    ):
        def create():
            args = [x for x in [handles, labels] if x is not None]
            legend = self.ax.legend(*args, **kwargs)
            return legend

        def remove(legend: Legend):
            legend.remove()

        deps = [handles, labels, kwargs, *self._legend_deps]
        return self._use_artifact("legend", deps, create, remove=remove)


class FigDrawing(Drawing):
    def __init__(self, fig: Figure | None = None, **kwargs):
        super().__init__()

        if fig is None:
            with pyplot.ioff():
                fig = pyplot.figure(**kwargs)

        self.fig = fig

    @override
    def begin(self, draw_on_end=True):
        drawstack = super().begin()

        drawstack.enter_context(pyplot.ioff())

        if draw_on_end:

            @drawstack.callback
            def on_end():
                self.draw()

        return drawstack

    def draw(self):
        self.fig.canvas.draw()

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
            drawstack = self._get_drawstack()

            for ax in axes:
                drawstack.enter_context(ax.begin())

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
                axw.end()
                axw.ax.remove()

        return self._use_artifact("subplots", deps, create, update, remove)

    def set_layout_engine(self, layout=None, **kwargs):
        def create():
            self.fig.set_layout_engine(layout, **kwargs)

        deps = dict(layout=layout, **kwargs)
        self._use_artifact("set_layout_engine", deps, create)

    def legend(
        self,
        handles: Iterable[Artist | tuple[Artist, ...]] | None = None,
        labels: Iterable[str] | None = None,
        **kwargs,
    ):
        def create():
            args = [x for x in [handles, labels] if x is not None]
            legend = self.fig.legend(*args, **kwargs)
            return legend

        def remove(legend: Legend):
            legend.remove()

        deps = [handles, labels, kwargs]
        return self._use_artifact("legend", deps, create, remove=remove)

    def set_size_inches(
        self, w: float | tuple[float, float], h: float | None = None, forward=True
    ):
        def create():
            self.fig.set_size_inches(w, h, forward)

        deps = [w, h, forward]
        self._use_artifact("set_size_inches", deps, create)

    def clear(self, draw=True):
        self._remove_all()
        self.fig.clear()

        if draw:
            self.draw()
