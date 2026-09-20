#!/usr/bin/env python3
"""Reserve VRAM and create feedback-controlled fluctuating GPU utilization.

Install NVML bindings if needed:
    pip install nvidia-ml-py

Semantics:
  --util-mean X   => X% mean target on EACH selected GPU.
  --util-total X  => X% total mean target divided evenly across selected GPUs.

For --util-mode sine-balanced, individual GPU targets fluctuate with phase
shifts. With equal means/jitters and no clipping, the sum of target utilization
stays approximately constant while each GPU moves up/down.
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import random
import signal
import time
from dataclasses import dataclass
from typing import List, Optional

import torch

try:
    from pynvml import (
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetName,
        nvmlDeviceGetUtilizationRates,
        nvmlInit,
        nvmlShutdown,
    )
except Exception as exc:
    raise RuntimeError(
        "Install NVML bindings first: pip install nvidia-ml-py\n"
        f"Original import error: {exc}"
    ) from exc

GIB = 1024 ** 3
MIB = 1024 ** 2


def parse_list(text: str, n: int, name: str) -> List[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if len(vals) == 1:
        vals *= n
    if len(vals) != n:
        raise ValueError(f"{name}: expected 1 value or {n} values, got {vals}")
    return vals


def torch_index_for_physical_gpu(physical_gpu: int) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return physical_gpu
    tokens = [x.strip() for x in visible.split(",") if x.strip()]
    if all(x.isdigit() for x in tokens):
        p = str(physical_gpu)
        if p not in tokens:
            raise RuntimeError(
                f"GPU {physical_gpu} is hidden by CUDA_VISIBLE_DEVICES={visible!r}"
            )
        return tokens.index(p)
    raise RuntimeError(
        "Please unset CUDA_VISIBLE_DEVICES, or use a numeric list such as 0,1,2."
    )


def reserve_vram(device: torch.device, target_bytes: int, chunk_mb: int):
    refs = []
    free0, _ = torch.cuda.mem_get_info(device)
    target_bytes = min(int(target_bytes), int(free0))
    chunk = max(1, int(chunk_mb * MIB))
    held = 0

    while held < target_bytes:
        n = min(chunk, target_bytes - held)
        try:
            x = torch.empty(n, dtype=torch.uint8, device=device)
            x.zero_()  # commit memory
            refs.append(x)
            held += x.numel()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break

    torch.cuda.synchronize(device)
    free1, _ = torch.cuda.mem_get_info(device)
    print(
        f"[{device}] VRAM held={held/GIB:.2f} GiB, free={free1/GIB:.2f} GiB",
        flush=True,
    )
    return refs


@dataclass
class TargetGenerator:
    mode: str
    mean: float
    jitter: float
    rank: int
    world: int
    wave_period: float
    random_hold: float
    rw_alpha: float
    rw_sigma: float
    rng: random.Random

    def __post_init__(self):
        self.current = float(self.mean)
        self.last_update = time.monotonic()

    def value(self, now: float, start: float) -> float:
        m = float(self.mean)
        j = max(0.0, float(self.jitter))
        if self.mode == "none" or j == 0:
            return max(0.0, min(100.0, m))

        if self.mode in ("sine", "sine-balanced"):
            phase = 2 * math.pi * (now - start) / max(self.wave_period, 1.0)
            if self.mode == "sine-balanced":
                phase += 2 * math.pi * self.rank / max(1, self.world)
            return max(0.0, min(100.0, m + j * math.sin(phase)))

        if now - self.last_update >= max(self.random_hold, 0.2):
            if self.mode == "uniform":
                self.current = self.rng.uniform(m - j, m + j)
            elif self.mode == "random-walk":
                self.current += self.rw_alpha * (m - self.current)
                self.current += self.rng.gauss(0.0, self.rw_sigma)
                self.current = max(m - j, min(m + j, self.current))
            else:
                raise ValueError(self.mode)
            self.last_update = now
        return max(0.0, min(100.0, self.current))


def make_buffers(device: torch.device, n: int):
    while n >= 256:
        try:
            a = torch.randn((n, n), dtype=torch.float16, device=device)
            b = torch.randn((n, n), dtype=torch.float16, device=device)
            c = torch.empty((n, n), dtype=torch.float16, device=device)
            torch.cuda.synchronize(device)
            return n, a, b, c
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            n //= 2
    raise RuntimeError(f"{device}: not enough VRAM for compute buffers")


def busy_for(seconds, device, a, b, c):
    if seconds <= 0:
        return
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        torch.mm(a, b, out=c)
        torch.cuda.synchronize(device)


def worker(
    physical_gpu: int,
    rank: int,
    world: int,
    vram_gb: Optional[float],
    free_fraction: Optional[float],
    util_mean: float,
    util_jitter: float,
    util_mode: str,
    wave_period: float,
    random_hold: float,
    rw_alpha: float,
    rw_sigma: float,
    leave_free_gb: float,
    chunk_mb: int,
    matrix_size: int,
    slice_ms: float,
    feedback_s: float,
    kp: float,
    ki: float,
    ema_alpha: float,
    print_s: float,
    seed: int,
):
    logical_gpu = torch_index_for_physical_gpu(physical_gpu)
    torch.cuda.set_device(logical_gpu)
    device = torch.device(f"cuda:{logical_gpu}")

    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(physical_gpu)
    try:
        name = nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode(errors="replace")

        free0, total = torch.cuda.mem_get_info(device)
        leave = int(leave_free_gb * GIB)
        if vram_gb is not None:
            requested = int(vram_gb * GIB)
        elif free_fraction is not None:
            requested = int(max(0, free0 - leave) * free_fraction)
        else:
            requested = 0
        requested = min(requested, max(0, free0 - leave))

        print(
            f"[GPU {physical_gpu}] {name} total={total/GIB:.2f}GiB "
            f"free={free0/GIB:.2f}GiB mean={util_mean:.1f}% "
            f"jitter=±{util_jitter:.1f}% mode={util_mode}",
            flush=True,
        )

        reservations = reserve_vram(device, requested, chunk_mb)

        stop = False
        def _stop(*_):
            nonlocal stop
            stop = True
        signal.signal(signal.SIGTERM, _stop)
        signal.signal(signal.SIGINT, _stop)

        compute_enabled = util_mean > 0 or util_jitter > 0
        if compute_enabled:
            n, a, b, c = make_buffers(device, matrix_size)
        else:
            n, a, b, c = 0, None, None, None

        gen = TargetGenerator(
            util_mode, util_mean, util_jitter, rank, world,
            wave_period, random_hold, rw_alpha, rw_sigma,
            random.Random(seed + physical_gpu * 100003),
        )

        slice_s = max(0.01, slice_ms / 1000.0)
        feedback_s = max(slice_s, feedback_s)
        print_s = max(feedback_s, print_s)
        start = last_fb = last_print = time.monotonic()

        duty = max(0.0, min(1.0, util_mean / 100.0))
        ema = None
        integral = 0.0
        observed = 0.0
        avg_sum = 0.0
        avg_n = 0

        print(
            f"[GPU {physical_gpu}] controller slice={slice_s*1000:.0f}ms "
            f"feedback={feedback_s:.2f}s matrix={n} kp={kp} ki={ki}",
            flush=True,
        )

        while not stop:
            loop_start = time.perf_counter()
            now = time.monotonic()
            target = gen.value(now, start)

            if compute_enabled and duty > 0:
                busy_for(slice_s * duty, device, a, b, c)

            remain = slice_s - (time.perf_counter() - loop_start)
            if remain > 0:
                time.sleep(remain)

            now = time.monotonic()
            if now - last_fb >= feedback_s:
                observed = float(nvmlDeviceGetUtilizationRates(handle).gpu)
                ema = observed if ema is None else ema_alpha * observed + (1-ema_alpha) * ema
                dt = max(now - last_fb, 1e-3)
                error = target - ema
                integral = max(-500.0, min(500.0, integral + error * dt))
                duty += kp * error / 100.0 + ki * integral / 100.0
                duty = max(0.0, min(1.0, duty))
                avg_sum += observed
                avg_n += 1
                last_fb = now

            if now - last_print >= print_s:
                avg = avg_sum / avg_n if avg_n else observed
                free_now, _ = torch.cuda.mem_get_info(device)
                print(
                    f"[GPU {physical_gpu}] target={target:5.1f}% "
                    f"obs={observed:5.1f}% ema={(ema or 0):5.1f}% "
                    f"avg={avg:5.1f}% duty={100*duty:5.1f}% "
                    f"free={free_now/GIB:5.2f}GiB",
                    flush=True,
                )
                avg_sum = 0.0
                avg_n = 0
                last_print = now

        if compute_enabled:
            del a, b, c
        del reservations
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
    finally:
        try:
            nvmlShutdown()
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--gpus", required=True, help="Physical IDs, e.g. 0,1,2,3")

    mg = p.add_mutually_exclusive_group()
    mg.add_argument("--vram-gb", default=None, help="One value or one per GPU")
    mg.add_argument("--free-vram-fraction", type=float, default=None)

    ug = p.add_mutually_exclusive_group()
    ug.add_argument("--util-mean", default=None, help="PER-GPU mean target")
    ug.add_argument("--util-total", type=float, default=None,
                    help="TOTAL target, divided evenly among selected GPUs")

    p.add_argument("--util-jitter", default="0",
                   help="Fluctuation amplitude, one value or one per GPU")
    p.add_argument("--util-mode",
                   choices=("none", "sine", "sine-balanced", "uniform", "random-walk"),
                   default="sine-balanced")
    p.add_argument("--wave-period", type=float, default=24.0)
    p.add_argument("--random-hold", type=float, default=3.0)
    p.add_argument("--random-walk-alpha", type=float, default=0.20)
    p.add_argument("--random-walk-sigma", type=float, default=2.0)

    p.add_argument("--leave-free-gb", type=float, default=1.5)
    p.add_argument("--chunk-mb", type=int, default=256)
    p.add_argument("--matrix-size", type=int, default=2048)
    p.add_argument("--slice-ms", type=float, default=50.0)
    p.add_argument("--feedback-s", type=float, default=0.5)
    p.add_argument("--kp", type=float, default=0.35)
    p.add_argument("--ki", type=float, default=0.015)
    p.add_argument("--ema-alpha", type=float, default=0.35)
    p.add_argument("--print-s", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        raise ValueError("No GPUs specified")
    n = len(gpus)

    vrams = parse_list(args.vram_gb, n, "--vram-gb") if args.vram_gb else [None] * n

    if args.util_total is not None:
        if args.util_total < 0:
            raise ValueError("--util-total must be >= 0")
        means = [args.util_total / n] * n
    elif args.util_mean is not None:
        means = parse_list(args.util_mean, n, "--util-mean")
    else:
        means = [0.0] * n

    jitters = parse_list(args.util_jitter, n, "--util-jitter")

    for gpu, mean, jitter in zip(gpus, means, jitters):
        if not 0 <= mean <= 100:
            raise ValueError(f"GPU {gpu}: mean {mean} outside [0,100]")
        if jitter < 0:
            raise ValueError(f"GPU {gpu}: jitter must be >= 0")

    if args.free_vram_fraction is not None and not (0 < args.free_vram_fraction <= 1):
        raise ValueError("--free-vram-fraction must be in (0,1]")

    print("Targets:", flush=True)
    for i, (gpu, mean, jitter) in enumerate(zip(gpus, means, jitters)):
        print(f"  GPU {gpu}: mean={mean:.2f}% jitter=±{jitter:.2f}% phase={i}/{n}", flush=True)
    print(f"  sum(mean targets)={sum(means):.2f}%", flush=True)

    if args.util_mode == "sine-balanced":
        if max(means) - min(means) > 1e-9 or max(jitters) - min(jitters) > 1e-9:
            print("[WARN] constant-sum sine is cleanest with equal means/jitters.", flush=True)
        if any(m-j < 0 or m+j > 100 for m, j in zip(means, jitters)):
            print("[WARN] sine clipping at 0/100 will break exact zero-sum targets.", flush=True)

    ctx = mp.get_context("spawn")
    procs = []
    for rank, (gpu, vram, mean, jitter) in enumerate(zip(gpus, vrams, means, jitters)):
        pr = ctx.Process(
            target=worker,
            args=(
                gpu, rank, n, vram, args.free_vram_fraction,
                mean, jitter, args.util_mode, args.wave_period,
                args.random_hold, args.random_walk_alpha, args.random_walk_sigma,
                args.leave_free_gb, args.chunk_mb, args.matrix_size,
                args.slice_ms, args.feedback_s, args.kp, args.ki,
                args.ema_alpha, args.print_s, args.seed,
            ),
        )
        pr.start()
        procs.append(pr)

    try:
        for pr in procs:
            pr.join()
    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
        for pr in procs:
            if pr.is_alive():
                pr.terminate()
        for pr in procs:
            pr.join()


if __name__ == "__main__":
    main()
