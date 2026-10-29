"""Run pai_training/scripts/debug_policy.py headlessly, saving the plot as PNG.

Usage (same arguments as debug_policy.py, plus --save-png):
  python run_debug_policy.py --save-png out.png --dataset-root ... --repo-id ... \
      --policy-path ... --episode-index 0
"""
import runpy
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

args = sys.argv[1:]
png = "debug_policy.png"
if "--save-png" in args:
    i = args.index("--save-png")
    png = args[i + 1]
    del args[i : i + 2]

plt.show = lambda *a, **k: plt.savefig(png, dpi=120)
sys.argv = ["debug_policy.py"] + args
sys.path.insert(0, "/home/ubuntu/pai_training/scripts")
runpy.run_path("/home/ubuntu/pai_training/scripts/debug_policy.py", run_name="__main__")
print(f"plot saved to {png}")
