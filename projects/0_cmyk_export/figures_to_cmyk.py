import argparse
import shutil
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PNG_DIR = (RESULTS_DIR / "rgb_png").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
CMYK_PDF_DIR = (RESULTS_DIR / "cmyk_pdf").resolve()
DATA_DIR = (RESULTS_DIR / "data").resolve()

# the book imports the cmyk print set from plots/ and the plot data from plottingdata/
BOOK_PLOTS_DIR = (BASE_DIR / "../../../book/plots").resolve()
BOOK_DATA_DIR = (BASE_DIR / "../../../book/plottingdata").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true", help="rebuild outputs even if up to date")
parser.add_argument("--report", action="store_true", help="list cmyk pdfs above the size limit")
parser.add_argument("--export", action="store_true",
                    help="copy cmyk_pdf/ into book/plots/ and data/ into book/plottingdata/")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# tag the source as srgb, then convert to fogra39 coated (springer european print target)
ICC_SRGB = Path("/usr/share/texlive/texmf-dist/tex/generic/colorprofiles/sRGB.icc")
ICC_CMYK = Path("/usr/share/texlive/texmf-dist/tex/generic/colorprofiles/FOGRA39L_coated.icc")

DPI = 400  # rasterization resolution for an rgb png derived from a vector pdf
SIZE_LIMIT = 1_000_000  # bytes; cmyk pdfs above this are flagged by --report

# sources are the rgb originals the drivers write: matplotlib pdfs in rgb_pdf/,
# raster renders (e.g. pvpython) in rgb_png/. cmyk_pdf/ is always derived.

# --------------------------------------- helper --------------------------------------
def run(cmd):
    """run a command (list form), raising on a non-zero exit."""
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def up_to_date(source, target):
    """true if target exists and is no older than source, unless --force is set."""
    return not args.force and target.exists() and target.stat().st_mtime >= source.stat().st_mtime


def pdf_to_cmyk(source, target):
    """convert a pdf to CMYK via ghostscript; vector content stays vector."""
    run([
        "gs", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=pdfwrite",
        "-dProcessColorModel=/DeviceCMYK", "-sColorConversionStrategy=CMYK",
        "-dOverrideICC=true", f"-sOutputICCProfile={ICC_CMYK}",
        "-o", str(target), str(source),
    ])


def pdf_to_png(source, target):
    """rasterize a pdf to a transparent RGB png at DPI (pdftocairo appends .png)."""
    run(["pdftocairo", "-png", "-transp", "-r", str(DPI), "-singlefile",
         str(source), str(target.with_suffix(""))])


def raster_to_cmyk(source, target):
    """convert a raster image to a CMYK pdf; alpha is kept as a soft mask."""
    run(["convert", str(source), "-profile", str(ICC_SRGB),
         "-profile", str(ICC_CMYK), str(target)])


def raster_to_rgb_pdf(source, target):
    """wrap a raster image into an RGB pdf; alpha is kept as a soft mask."""
    run(["convert", str(source), "-profile", str(ICC_SRGB), str(target)])


# --------------------------------------- export --------------------------------------
CMYK_PDF_DIR.mkdir(parents=True, exist_ok=True)

# rgb_pdf/ and rgb_png/ hold the source originals; a pdf wins over a raster of the
# same stem (the matplotlib vector is preferred over a derived png)
by_stem = {}
for p in RGB_PNG_DIR.glob("*.png"):
    by_stem[p.stem] = p
for p in RGB_PDF_DIR.glob("*.pdf"):
    by_stem[p.stem] = p
sources = sorted(by_stem.values())

for source in sources:
    stem = source.stem
    rgb_png = RGB_PNG_DIR / f"{stem}.png"
    rgb_pdf = RGB_PDF_DIR / f"{stem}.pdf"
    cmyk_pdf = CMYK_PDF_DIR / f"{stem}.pdf"

    if source.suffix == ".pdf":  # matplotlib pdf: it is already the rgb_pdf original
        if not up_to_date(source, cmyk_pdf):
            pdf_to_cmyk(source, cmyk_pdf)
        if not up_to_date(source, rgb_png):
            pdf_to_png(source, rgb_png)
    else:  # raster render: it is already the rgb_png original
        if not up_to_date(source, cmyk_pdf):
            raster_to_cmyk(source, cmyk_pdf)
        if not up_to_date(source, rgb_pdf):
            raster_to_rgb_pdf(source, rgb_pdf)

print(f"converted {len(sources)} figures into rgb_png/, rgb_pdf/, cmyk_pdf/")

# --------------------------------------- export --------------------------------------
# copy into the book repo, overwriting same-named files but leaving everything else in
# place (the book may hold figures/data this pipeline does not generate)
if args.export:
    cmyk_pdfs = sorted(CMYK_PDF_DIR.glob("*.pdf"))
    for p in cmyk_pdfs:
        shutil.copy2(p, BOOK_PLOTS_DIR / p.name)
    data_files = sorted(f for f in DATA_DIR.iterdir() if f.is_file())
    for f in data_files:
        shutil.copy2(f, BOOK_DATA_DIR / f.name)
    print(f"exported {len(cmyk_pdfs)} pdfs to book/plots/ and {len(data_files)} "
          f"data files to book/plottingdata/")

# --------------------------------------- report --------------------------------------
if args.report:
    big = sorted((p for p in CMYK_PDF_DIR.glob("*.pdf") if p.stat().st_size > SIZE_LIMIT),
                 key=lambda p: p.stat().st_size, reverse=True)
    print(f"{len(big)} cmyk pdfs over {SIZE_LIMIT // 1000} kb (rasterize the vector ones):")
    for p in big:
        # vector pdfs carry no image xobject; only those shrink when rasterized at the source
        listing = subprocess.run(["pdfimages", "-list", str(p)],
                                 capture_output=True, text=True).stdout.splitlines()
        kind = "raster" if len(listing) > 2 else "vector"
        print(f"  {p.stat().st_size // 1000:>6} kb  {kind}  {p.name}")
