---
name: mlhp-visualize
description: >-
  See and verify mlhp finite-element results by rendering them to PNG/MP4
  without ParaView. Use to confirms a simulation actually worked — that a 
  setup, field, or deformation looks right — by rendering an image and 
  reading it back, the fastest check that a run is correct. Also produces 
  presentable figures and animations for users. Bundled headless-vtk
  scripts render VTU/PVTU from C++ or Python runs; a reference catalogs the
  mlhp output API and the in-memory 2D matplotlib path. Use whenever a result
  or setup should be visually checked, or a figure or animation is wanted.
---

# mlhp-visualize

The first use of this skill is **verification**: after running a simulation,
render the result and look at it to confirm it did what it should — the field
is sensible, the geometry/boundary conditions are right, the deformation goes
the right way, nothing is empty or diverged. Rendering an image and reading it
back is often the fastest and most honest check that a run worked. The same
tools also make **presentable** figures and animations for users. No ParaView:
the bundled scripts render headlessly with vtk and encode with ffmpeg.

Scripts (in `scripts/`; need `vtk` + `numpy`, video also needs `ffmpeg`):
- **`render_vtu.py`** — one VTU/PVTU → PNG. The verification workhorse and stills.
- **`make_video.py`** — a glob of timestep files → MP4 (one global color range).
- `mlhp_visualize.py` — shared core, imported by the two CLIs (don't run directly).

Run them with any Python that has `vtk` — it need not be the interpreter mlhp
is built against (rendering reads the files, it doesn't import mlhp). `--help`
on each lists every option.

## Assume the first render is not right

Whatever you produce — a verification image or a figure for a user — assume the
framing and annotations are **not right yet**. Read the PNG back and actively
criticize it before doing anything else with it:

- **Content**: does it actually show what the simulation should produce — the
  right field, geometry, magnitudes, orientation? Is anything empty, clipped,
  off-center, or obviously wrong?
- **Composition**: are the colorbar, title, labels and legend clean — no
  oversized fonts, no tick labels overlapping the bar or each other, no wasted
  whitespace, no panel floating over the data? **Crop the colorbar/label region
  and enlarge it ~3× before judging** — these defects are invisible at
  full-frame scale (the scripts give a decent starting layout, not a perfect
  one).

Name the concrete flaws, fix them (a script flag, or adjust the render), and
render again. Do at least one look–criticize–improve pass and repeat until the
specific flaws are gone. This self-review is the point of the skill: it is what
lets a good image come out without the user having to point the problems out.

## Which path

- **Files → render (general, 2D and 3D, C++ or Python).** A run writes
  `.vtu`/`.pvtu` with the mlhp output API; render them with the scripts. The
  output API is catalogued in `reference/output_api.md`.
- **Python, 2D, no files wanted.** Postprocess in memory (`DataAccumulator`)
  and plot with matplotlib — fastest for a 2D self-check, writes only your PNG.
  See `reference/output_api.md` (§ in-memory). Here a single Python program
  both imports `mlhp` and plots, so that one interpreter needs `matplotlib` +
  `numpy` too — whereas the file → render path is two separate programs (the
  mlhp run writes files; `render_vtu.py` reads them), which may use different
  interpreters, so the renderer only needs `vtk`.

## Verify (agent self-check)

Render, then **Read the PNG back and judge it** (per the section above); do not
declare a result correct without looking.

```
python scripts/render_vtu.py result.vtu --field VonMisesStress --out vm.png
python scripts/render_vtu.py flow.pvtu  --field mag:Solution    --out u.png
```

`--field` takes `NAME`, `mag:NAME` (vector magnitude), or `comp:NAME:i`.

## Present (figures / animations)

```
# 3D deformation: warp by the displacement field, color by stress (Pa->MPa),
# overlay the deformed mesh edges, isometric view:
python scripts/render_vtu.py solid_9.pvtu --field VonMisesStress --scale 1e-6 \
    --warp Displacement --view iso --edges edges_9.pvtu --bar-title "vM [MPa]"

# animation over a timestep series (files sorted by their trailing integer):
python scripts/make_video.py "outputs/solid_*.pvtu" --field VonMisesStress \
    --scale 1e-6 --warp Displacement --view iso \
    --edges-pattern "outputs/edges_*.pvtu" --fps 4 --out deform.mp4
```

**Verifying a video.** You cannot read an MP4 — to check an animation, inspect
representative still frames as PNGs (first / middle / last): render a few
timestep files with `render_vtu.py`, or pass `--keep-frames` to `make_video.py`
and read those. And get the **single-frame layout right before batch-encoding**:
run the look–criticize–improve loop on one frame first, then `make_video.py` —
don't discover an oversized colorbar after rendering a hundred frames.

## Traps the scripts already encode (only matters if you go off-script)

- **2D framing.** With parallel projection, `ResetCamera` sets the scale from
  the bounding-box *radius*, so a long thin domain fills a fraction of the
  frame. The fill is from the extent + viewport aspect instead (`--view 2d`).
- **Colorbar.** `vtkScalarBarActor` autoscales its fonts to the bar size — the
  title balloons — unless `UnconstrainedFontSizeOn()`. A title with math-like
  characters (`"|u|"`) triggers vtk math-text and floods warnings; use a plain
  title (`"u"`).
- **Video.** Use one global color range across all frames or colors/colorbar
  flicker; x264 rejects odd pixel dimensions. `make_video.py` handles both.
- **FCM / cut cells.** Stresses can spike arbitrarily on tiny cut cells — use
  `--range p99` and restrict output to the physical domain (`domainCellMesh`,
  see output_api.md), or one sliver washes out the scale.
- **Headless Linux** has no OpenGL context by default: `pip install vtk-osmesa`
  or run under `xvfb-run`. Fine on Windows/macOS/Linux-with-display.
