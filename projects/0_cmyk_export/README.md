# CMYK figure export

Springer print compliance requires every book figure to be CMYK or grayscale (no
RGB). matplotlib cannot write CMYK and PNG has no CMYK colorspace, so conversion
happens after the drivers run.

Drivers write their book outputs into fixed sibling folders under `results/` (assumed
to exist -- they are committed with a `.gitkeep`, so drivers never `mkdir` them):

- `results/rgb_pdf/` -- every matplotlib book figure (`savefig(RGB_PDF_DIR / ...)`)
- `results/rgb_png/` -- every pvpython 3D render (the `0_pvpython` drivers `save_png` here)
- `results/data/`    -- every plot data file (`save_csv`/`savetxt` to `CSV_DIR`)

`figures_to_cmyk.py` reads the rgb originals (`rgb_pdf/*.pdf` and any raster render in
`rgb_png/*.png`) and fills the remaining two folders so every figure exists in each:

- `results/cmyk_pdf/` -- the print set the book imports (`\graphicspath{{cmyk_pdf/}}`)
- `results/rgb_pdf/`  -- the rgb reference (matplotlib writes here directly)
- `results/rgb_png/`  -- rgb raster for slides (transparency preserved)

Run it after generating figures with `--book`:

```
python projects/0_cmyk_export/figures_to_cmyk.py            # convert
python projects/0_cmyk_export/figures_to_cmyk.py --export   # also copy into the book repo
python projects/0_cmyk_export/figures_to_cmyk.py --report   # also list cmyk pdfs > 1 mb
python projects/0_cmyk_export/figures_to_cmyk.py --force    # rebuild everything
```

`--export` copies `cmyk_pdf/*.pdf` into `book/plots/` and `data/*` into
`book/plottingdata/`, overwriting same-named files but leaving anything else in those
folders untouched.

It is idempotent (skips outputs newer than their source). A `rgb_pdf/*.pdf` source
(matplotlib) is colour-converted to cmyk with ghostscript, staying vector, and
rasterized to an rgb png; a `rgb_png/*.png` source (a pvpython render written there by a
`0_pvpython` driver) becomes a cmyk pdf with its alpha kept as a soft mask plus a wrapped
rgb pdf. When a
stem exists as both, the pdf wins. Colour goes through `FOGRA39L_coated.icc` (European
coated, Springer's target).

`--report` flags `cmyk_pdf` files over 1 mb and labels each `vector` or `raster`: a
large *vector* field (a finely resolved `tricontourf`) shrinks if you add
`set_rasterized(True)` at the source; a large *raster* is already an image and will
not.

## Non-obvious technicalities (authored by Claude)

Drivers export book figures as `.pdf`. Vector plots stay vector; raster fields
(`imshow`, `pcolormesh`, finely-meshed `tricontourf`) carry `set_rasterized(True)` so
the PDF embeds an image instead of millions of vector primitives. `imshow` needs no
such flag -- it is always an embedded image. Rasterization only controls file size; it
is independent of the cmyk conversion (ghostscript converts vector fields to cmyk
vector at any size).

Requires `gs`, `pdftocairo` (poppler), and ImageMagick `convert` on the path.
