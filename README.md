# mplnow

Matplotlib now!

An immediate mode style API wrapper for matplotlib plotting functions.

### How it works

E.g. `axvline` in `AxesDrawing`

```Python
class AxesDrawing(Drawing):
    ...

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
```

The `Drawing` class provides `self._use_artifact` to maintain a persistent state between re-draws.

- `create`: callback to draw the artifact for the first time
- `update`: callback to update the artifact without completely redrawing
- `remove`: callback to remove the artifact from the drawing
- `deps`: dict[str, Any] or list[Any] of dependancies - if any of these change between drawings, the artifact is completely `remove()` and `create()` again. Otherwise if the dependancies stay the same, only `update()` is called.

For `axvline`, all arguments except `x` are supplied as dependancies. If `x` is changed between calls, the simpler `line2d.set_xdata` is used to update the line, instead of creating a new line.
