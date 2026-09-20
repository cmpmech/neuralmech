---
name: sweep-dump
description: Hyperparameter / data / epoch sweeps of a NeuralMech driver whose outcome the user judges by eye. Trains sed-patched copies of the driver in parallel, runs the matching sample/plot script per model, and dumps per-config PNGs + a labelled overview contact sheet + a metrics table into results/<sweep>/ for the user to pick from. Invoke when the user asks to "sweep", "try a few settings", "give me a dump/overview I can review", or wants generative samples compared across configs. Never picks the winner; applies the user's pick afterwards.
---

Sweep a driver over a few constants, dump images the user can judge, apply their pick.
The repo stays untouched until the user chooses. Claude reports numbers, the user judges
the pictures — do not call a config "best" on visual grounds.

## 1. Diagnose before sweeping
Load the existing model and print the numbers the driver already reports (IoU/recon,
KL, active units, sample stats). One evaluation tells you which axis matters (e.g. KL
in the hundreds → the KL weight is inert; sweep it in ×4 steps, not ×2).

## 2. Sweep = sed-patched copies, never edited drivers
Per config `TAG`, copy the driver into the scratchpad and patch only constants:

```bash
sed -e "s|^BETA = .*|BETA = $BETA|" \
    -e "s|^EPOCHS = .*|EPOCHS = $EPOCHS|" \
    -e "s|^DATA_DIR = .*|DATA_DIR = Path('$REPO/data')|" \       # BASE_DIR-relative paths break in the copy
    -e "s|^MODEL_DIR = .*|MODEL_DIR = Path('$S/$TAG/models')|" \  # driver's save name may collide across tags
    $REPO/projects/X/driver_train.py > $S/$TAG/train.py
cd $S/$TAG && OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$REPO MPLBACKEND=Agg \
    python train.py > $OUT/$TAG.log 2>&1 && cp $S/$TAG/models/*.pt2 $OUT/models/$TAG.pt2
```

- `MPLBACKEND=Agg` turns the driver's `plt.show()` into no-ops; `grep -q` the patched
  line to be sure the sed hit.
- `OMP_NUM_THREADS=4` per process — 8 parallel torch processes with default threads
  oversubscribe a shared box 10×. Spread runs over GPUs with `CUDA_VISIBLE_DEVICES`.
- If per-batch CPU work (augmentation, standardization) throttles, patch
  `standardize(augment(x)).to(device)` → `augment(standardize(x).to(device))` in the copy.
- Tag = the varied constants, e.g. `b16_c4`, `n1000`, `e800`. Always include the current
  repo config as `baseline` (reuse the saved model, don't retrain).
- Confounds get a control run: more data at fixed epochs = more steps, so add
  `n500_e3200` next to `n2000_e800`.
- Wait with `timeout 590 bash -c 'until [ $(ls $OUT/models | grep -c pt2) -ge N ]; do sleep 30; done'`;
  trust the files on disk, not task notifications.

## 3. Dump = the driver's own sample script, savefig instead of show
Copy `driver_sample.py` per tag, patch `MODEL_DIR`/`MODEL`, `SAMPLES = 8`, any knob the
user is testing (temperature, threshold), and `plt.show()` → `fig.savefig("$OUT/${TAG}_<knob>.png", dpi=100)`.
Same seed in every copy so samples are comparable across rows.

## 4. Deliverables in `results/<sweep>/` (gitignored)
- `<tag>_<knob>.png` — raw sample-script output per config
- `overview*.png` — one contact sheet per knob value, rows = tags:
  `python .claude/skills/sweep-dump/scripts/overview.py $OUT/overview.png "title" tag1 tag2 ...`
  (expects `$OUT/<tag>_<knob>.png` next to it; pass `--suffix _T1.0`)
- `metrics.txt` — one row per tag from the log's summary prints; put the data row last
- `<tag>.log`, `models/<tag>.pt2`
Report: folder path, tag legend, the metrics table, one or two objective observations,
then stop and ask for the pick.

## 5. Apply the pick
Copy `models/<tag>.pt2` → `models/<driver's own save name>`, set the winning constants in
the driver and the model name in the sample script, delete knobs the pick made obsolete.
If the pick changes the dataset, regenerate it with the repo generator (keep its seed so
the old set is a prefix of the new one), back the old file up, and say which other
drivers read that file.
