import time
from typing import Callable
from dataclasses import dataclass
import logging
import asyncio
from asyncio import AbstractEventLoop

log = logging.getLogger(__name__)


class AsyncioTickWatch:
    def __init__(
        self, loop: AbstractEventLoop | None = None, tick_length: float | None = None
    ):
        self.ticks = 0
        self._task: asyncio.Task | None = None
        self.tick_length = tick_length or 0.01

        self._loop = loop or asyncio.get_event_loop()

    async def _main(self):
        try:
            while True:
                self.ticks += 1
                await asyncio.sleep(self.tick_length)
        except asyncio.CancelledError:
            log.debug(f"tick watch cancelled ({self.ticks})")

    def start(self):
        if self._task and not self._task.done():
            self._task.cancel()

        self.ticks = 0
        self._task = self._loop.create_task(self._main())

    def stop(self):
        if self._task and not self._task.done():
            self._task.cancel()


@dataclass
class TimePoint:
    name: str
    time_perf: int
    time_process: int
    asyncio_ticks: int


class Stopwatch:

    def __init__(
        self,
        loop: AbstractEventLoop | None = None,
        tick_length: float | None = None,
        print_fn: Callable[[str], None] | None = print,
    ):
        self.time_points: list[TimePoint] = []
        self._print_fn = print_fn

        if loop:
            self._ticker = AsyncioTickWatch(loop=loop, tick_length=tick_length)
        else:
            self._ticker = None

    def begin(self):
        self.time_points = []
        if self._ticker:
            self._ticker.start()
        self.mark("begin")

    def mark(self, name: str):
        ticks = 0 if not self._ticker else self._ticker.ticks
        point = TimePoint(
            name,
            time.perf_counter_ns(),
            time.process_time_ns(),
            ticks,
        )
        self.time_points.append(point)

    def end(self, name="end", print=True):
        self.mark(name)
        if self._ticker:
            self._ticker.stop()

        if print:
            self.print()

    def print(self):
        if not self._print_fn:
            return

        t0 = self.time_points[0]
        tlast = t0

        for tp in self.time_points[1:]:
            tperf = tp.time_perf - tlast.time_perf
            tproc = tp.time_process - tlast.time_process
            records = [
                f"{tperf * 1e-6 :6.2f} ms",
                f"{tproc * 1e-6 :6.2f} ms",
            ]
            if self._ticker:
                ticks = tp.asyncio_ticks - tlast.asyncio_ticks
                records += [
                    f"{ticks * self._ticker.tick_length * 1e3 :3.0f} ms",
                ]

            max_name_length = max(len(tp.name) for tp in self.time_points)
            self._print_fn(f"{tp.name:>{max_name_length}s}: {', '.join(records)}")

            tlast = tp
