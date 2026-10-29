"""Overlay policy rollout logs (infer_drill_side.py stdout) on the demo trajectories.

Usage: python compare_rollouts.py --dataset-root DIR --png out.png run1.log run2.log ...
"""
import argparse
import glob
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dataset-root", required=True)
ap.add_argument("--png", required=True)
ap.add_argument("--duration", type=float, default=54.5)
ap.add_argument("logs", nargs="+")
args = ap.parse_args()

# demo: observation.state per episode, time = frame / 20 fps
demos = []
for f in sorted(glob.glob(f"{args.dataset_root}/data/*/*.parquet")):
    d = pd.read_parquet(f)
    demos.append(np.stack(d["observation.state"].to_numpy())[:, :5])
grid = np.linspace(0, args.duration, 400)


def resample(t, x):
    return np.stack([np.interp(grid, t, x[:, j]) for j in range(4)], axis=1)


demo_r = np.stack([resample(np.arange(len(x)) / 20.0, x) for x in demos])
demo_mean, demo_std = demo_r.mean(0), demo_r.std(0)

runs = []
for p in args.logs:
    ph, st = [], []
    for line in open(p):
        if line.startswith("phase"):
            ph.append(float(line.split(":")[1]))
        elif line.startswith("state"):
            st.append([float(v) for v in re.findall(r"-?\d+\.\d+|-?\d+", line.split(":", 1)[1])][:5])
    n = min(len(ph), len(st))
    t = np.array(ph[:n]) * args.duration
    runs.append(resample(t, np.array(st[:n])))

names = ["joint1", "joint2", "joint3", "joint4"]
fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
for j, ax in enumerate(axs):
    ax.fill_between(grid, demo_mean[:, j] - demo_std[:, j], demo_mean[:, j] + demo_std[:, j],
                    color="gray", alpha=0.3, label="demo mean±std")
    ax.plot(grid, demo_mean[:, j], "k", lw=1)
    for i, r in enumerate(runs):
        ax.plot(grid, r[:, j], lw=1, label=f"run{i + 1}")
    ax.set_ylabel(names[j])
axs[0].legend(ncol=4)
axs[-1].set_xlabel("time [s]")
fig.tight_layout()
fig.savefig(args.png, dpi=110)

print("RMSE vs demo mean [rad] (per run, per joint):")
for i, r in enumerate(runs):
    print(f"run{i + 1}", np.sqrt(((r - demo_mean) ** 2).mean(0)).round(3))
print("demo episode-to-episode std (mean over time):", demo_std.mean(0).round(3))
if len(runs) > 1:
    r = np.stack(runs)
    print("run-to-run std (mean over time):", r.std(0).mean(0).round(3))
