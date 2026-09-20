"""Stack per-tag sample PNGs into one labelled contact sheet.

usage: overview.py OUT.png TITLE TAG [TAG ...] [--suffix SUFFIX]
reads <dir of OUT>/<TAG><SUFFIX>.png for each tag (default suffix "")
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as im  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

args = sys.argv[1:]
suffix = ""
if "--suffix" in args:
    i = args.index("--suffix")
    suffix = args[i + 1]
    del args[i : i + 2]
out, title, tags = Path(args[0]), args[1], args[2:]

fig, ax = plt.subplots(len(tags), 1, figsize=(16, 2.1 * len(tags)), squeeze=False)
for axis, tag in zip(ax[:, 0], tags):
    axis.imshow(im.imread(out.parent / f"{tag}{suffix}.png"))
    axis.set_ylabel(tag, rotation=0, ha="right", va="center", fontsize=12)
    axis.set_xticks([])
    axis.set_yticks([])
fig.suptitle(title)
fig.tight_layout()
fig.savefig(out, dpi=80)
