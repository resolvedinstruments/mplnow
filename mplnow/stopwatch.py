import time
from dataclasses import dataclass


@dataclass
class TimePoint:
    name: str
    time_perf: int
    time_process: int


class Stopwatch:

    def __init__(self):
        self.time_points: list[TimePoint] = []

    def begin(self):
        self.time_points = []
        self.mark("begin")

    def mark(self, name: str):
        time_perf = time.perf_counter_ns()
        time_process = time.process_time_ns()
        self.time_points.append(TimePoint(name, time_perf, time_process))

    def end(self, name="end", print=True):
        self.mark(name)

        if print:
            self.print()

    def print(self):
        t0 = self.time_points[0]
        tlast = t0

        for tp in self.time_points[1:]:
            tperf = tp.time_perf - tlast.time_perf
            tproc = tp.time_process - tlast.time_process
            print(f"{tp.name}: {tperf * 1e-6 :.2f} ms, {tproc * 1e-6 :.2f} ns")

            tlast = tp
