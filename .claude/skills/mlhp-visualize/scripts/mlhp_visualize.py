"""Shared rendering core for the mlhp-visualize skill.

Turns an mlhp VTU/PVTU file into a PNG with a headless vtk pipeline (no
ParaView). Used by render_vtu.py (one image) and make_video.py (a series).
Platform independent: native OpenGL offscreen works on Windows/macOS/Linux
with a display; on a headless Linux box install the `vtk-osmesa` wheel or run
under `xvfb-run` (see check_offscreen).

Requires: vtk, numpy.
"""
import os
import sys

try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
except ImportError:                                            # pragma: no cover
    sys.exit("This script needs the 'vtk' module: pip install vtk numpy")
import numpy as np


# --------------------------------------------------------------------------- #
# Color maps (control points in [0,1] -> RGB). turbo is the default.
# --------------------------------------------------------------------------- #
_CMAPS = {
    "turbo":    [(0.0,(0.19,0.07,0.23)),(0.25,(0.10,0.50,0.85)),(0.5,(0.10,0.80,0.45)),
                 (0.7,(0.85,0.90,0.10)),(0.85,(0.95,0.45,0.10)),(1.0,(0.65,0.0,0.05))],
    "viridis":  [(0.0,(0.27,0.00,0.33)),(0.33,(0.23,0.42,0.55)),(0.66,(0.13,0.66,0.52)),
                 (1.0,(0.99,0.91,0.14))],
    "coolwarm": [(0.0,(0.23,0.30,0.75)),(0.5,(0.87,0.87,0.87)),(1.0,(0.71,0.02,0.15))],
    "gray":     [(0.0,(0.0,0.0,0.0)),(1.0,(1.0,1.0,1.0))],
}


def color_transfer(vmin, vmax, cmap="turbo"):
    ctf = vtk.vtkColorTransferFunction()
    pts = _CMAPS.get(cmap, _CMAPS["turbo"])
    span = (vmax - vmin) or 1.0
    for t, (r, g, b) in pts:
        ctf.AddRGBPoint(vmin + t * span, r, g, b)
    return ctf


def check_offscreen():
    """Return True if offscreen rendering produces a non-empty image."""
    rw = vtk.vtkRenderWindow(); rw.SetOffScreenRendering(1); rw.SetSize(16, 16)
    rw.AddRenderer(vtk.vtkRenderer()); rw.Render()
    return True   # if Render() didn't crash, the GL context works


# --------------------------------------------------------------------------- #
# IO + fields
# --------------------------------------------------------------------------- #
def read_dataset(path):
    """Read .vtu/.pvtu/.vtp (the generic reader auto-detects the format)."""
    r = vtk.vtkXMLGenericDataObjectReader(); r.SetFileName(path); r.Update()
    out = r.GetOutput()
    if out is None or out.GetNumberOfPoints() == 0:
        # .pvtu is sometimes better served by the parallel reader
        r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(path); r.Update()
        out = r.GetOutput()
    return out


def resolve_field(dataset, spec, scale=1.0):
    """Make a scalar point array for coloring and return its name.

    spec forms:
      "Name"            scalar array (or first component)
      "mag:Name"        Euclidean magnitude of a vector array
      "comp:Name:i"     component i of an array
    `scale` multiplies the values (e.g. 1e-6 for Pa -> MPa).
    """
    pd = dataset.GetPointData()
    if spec.startswith("mag:"):
        name = spec[4:]
        a = vtk_to_numpy(pd.GetArray(name))
        vals = np.linalg.norm(a.reshape(a.shape[0], -1), axis=1)
        out = "%s_mag" % name
    elif spec.startswith("comp:"):
        _, name, i = spec.split(":"); i = int(i)
        a = vtk_to_numpy(pd.GetArray(name))
        vals = a[:, i] if a.ndim > 1 else a
        out = "%s_%d" % (name, i)
    else:
        name = spec
        a = vtk_to_numpy(pd.GetArray(name))
        vals = a[:, 0] if a.ndim > 1 else a
        out = name
    vals = np.ascontiguousarray(vals, dtype=float) * scale
    arr = vtk.vtkDoubleArray(); arr.SetName(out)
    arr.SetNumberOfValues(vals.size)
    for i, v in enumerate(vals):
        arr.SetValue(i, float(v))
    dataset.GetPointData().AddArray(arr)
    return out, vals


def warp(dataset, vecname, scale=1.0):
    dataset.GetPointData().SetActiveVectors(vecname)
    w = vtk.vtkWarpVector(); w.SetInputData(dataset); w.SetScaleFactor(scale); w.Update()
    return w.GetOutput()


def value_range(vals, mode):
    """mode: 'auto' -> (min,max); 'pNN' -> (0, NN-th pct); (lo,hi) tuple -> as-is."""
    if isinstance(mode, (tuple, list)):
        return float(mode[0]), float(mode[1])
    if mode and mode.startswith("p"):
        return 0.0, float(np.percentile(vals, float(mode[1:])))
    return float(np.min(vals)), float(np.max(vals))


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #
def fill_camera_2d(ren, bounds, W, fieldH):
    """Parallel projection that FILLS the field viewport with a 2D dataset.

    Do not use ResetCamera with parallel projection: it sets the scale from the
    bounding-box radius (half the diagonal), leaving large margins for a thin
    domain. Set the parallel scale from the extent and the viewport aspect.
    """
    cx, cy = 0.5*(bounds[0]+bounds[1]), 0.5*(bounds[2]+bounds[3])
    wx, wy = bounds[1]-bounds[0], bounds[3]-bounds[2]
    aspect = W / fieldH
    pscale = max(wy/2.0, (wx/2.0)/aspect)
    cam = ren.GetActiveCamera(); cam.ParallelProjectionOn()
    cam.SetFocalPoint(cx, cy, 0.0); cam.SetPosition(cx, cy, 1.0); cam.SetViewUp(0, 1, 0)
    cam.SetParallelScale(pscale)


_DIRS = {"+x":(1,0,0),"-x":(-1,0,0),"+y":(0,1,0),"-y":(0,-1,0),"+z":(0,0,1),"-z":(0,0,-1),
         "iso":(1,-1,1)}

def view_camera_3d(ren, view, zoom=1.3):
    """Reset to fit, then orient. Perspective is fine in 3D (the margin issue is
    specific to 2D parallel projection)."""
    ren.ResetCamera()
    cam = ren.GetActiveCamera()
    if view == "iso":
        cam.Azimuth(35); cam.Elevation(25); cam.OrthogonalizeViewUp()
    elif view in _DIRS:
        fp = cam.GetFocalPoint(); d = cam.GetDistance(); v = _DIRS[view]
        cam.SetPosition(fp[0]+v[0]*d, fp[1]+v[1]*d, fp[2]+v[2]*d)
        cam.SetViewUp(0,0,1) if view in ("+y","-y") else cam.SetViewUp(0,1,0)
    ren.ResetCameraClippingRange()
    cam.Zoom(zoom)


# --------------------------------------------------------------------------- #
# Colorbar + label  (UnconstrainedFontSizeOn avoids the giant-title autoscale)
# --------------------------------------------------------------------------- #
def scalar_bar(ctf, title, horizontal, label_fmt="%.2f"):
    bar = vtk.vtkScalarBarActor(); bar.SetLookupTable(ctf)
    bar.SetTitle(title); bar.SetNumberOfLabels(5); bar.SetLabelFormat(label_fmt)
    bar.UnconstrainedFontSizeOn()
    if horizontal:
        bar.SetOrientationToHorizontal()
        bar.SetTextPositionToPrecedeScalarBar()       # ticks below the bar
        bar.GetPositionCoordinate().SetValue(0.34, 0.40); bar.SetWidth(0.34); bar.SetHeight(0.40)
    else:
        bar.GetPositionCoordinate().SetValue(0.12, 0.16); bar.SetWidth(0.5); bar.SetHeight(0.66)
    for tp in (bar.GetTitleTextProperty(), bar.GetLabelTextProperty()):
        tp.SetColor(0, 0, 0); tp.SetFontFamilyToArial(); tp.BoldOff(); tp.ItalicOff(); tp.ShadowOff()
    bar.GetTitleTextProperty().SetFontSize(20); bar.GetLabelTextProperty().SetFontSize(15)
    return bar


def text_label(text, x, y):
    t = vtk.vtkTextActor(); t.SetInput(text)
    tp = t.GetTextProperty()
    tp.SetColor(0,0,0); tp.SetFontFamilyToArial(); tp.SetFontSize(22)
    tp.BoldOff(); tp.ItalicOff(); tp.ShadowOff()
    t.SetDisplayPosition(x, y)
    return t


# --------------------------------------------------------------------------- #
# Main render
# --------------------------------------------------------------------------- #
def render(path, out, field, view="2d", warp_field=None, warp_scale=1.0,
           rng="auto", scale=1.0, cmap="turbo", title=None, bar_title=None,
           label=None, edges=None, show_edges=False, size=None, label_fmt="%.2f"):
    """Render one dataset to a PNG. Returns the (vmin, vmax) used."""
    d = read_dataset(path)
    if warp_field:
        d = warp(d, warp_field, warp_scale)
    cname, vals = resolve_field(d, field, scale)
    vmin, vmax = value_range(vals, rng)
    ctf = color_transfer(vmin, vmax, cmap)

    surf = vtk.vtkDataSetSurfaceFilter(); surf.SetInputData(d); surf.Update()
    mapper = vtk.vtkPolyDataMapper(); mapper.SetInputConnection(surf.GetOutputPort())
    mapper.SetLookupTable(ctf); mapper.SetScalarRange(vmin, vmax)
    mapper.SetScalarModeToUsePointFieldData(); mapper.SelectColorArray(cname)
    mapper.InterpolateScalarsBeforeMappingOn()
    actor = vtk.vtkActor(); actor.SetMapper(mapper)

    is2d = (view == "2d")
    W, H = size or ((1600, 590) if is2d else (1200, 900))
    strip = 90 if is2d else 0
    fieldH = (H - strip) if is2d else H

    renF = vtk.vtkRenderer(); renF.SetBackground(1,1,1); renF.AddActor(actor)
    renF.SetViewport(0.0, (strip/H if is2d else 0.0), (1.0 if is2d else 0.86), 1.0)

    # mesh edges: a separate file overlaid, or feature edges of this dataset
    if edges:
        e = read_dataset(edges)
        if warp_field:
            e = warp(e, warp_field, warp_scale)
        em = vtk.vtkDataSetMapper(); em.SetInputData(e); em.ScalarVisibilityOff()
        ea = vtk.vtkActor(); ea.SetMapper(em)
        ea.GetProperty().SetColor(0.1,0.1,0.1); ea.GetProperty().SetLineWidth(1.0)
        renF.AddActor(ea)
    elif show_edges:
        ed = vtk.vtkExtractEdges(); ed.SetInputConnection(surf.GetOutputPort())
        em = vtk.vtkPolyDataMapper(); em.SetInputConnection(ed.GetOutputPort()); em.ScalarVisibilityOff()
        ea = vtk.vtkActor(); ea.SetMapper(em)
        ea.GetProperty().SetColor(0.15,0.15,0.15); ea.GetProperty().SetLineWidth(1.0)
        renF.AddActor(ea)

    if is2d:
        fill_camera_2d(renF, d.GetBounds(), W, fieldH)
    else:
        view_camera_3d(renF, view)

    if label:
        renF.AddViewProp(text_label(label, 20, H-40))

    rw = vtk.vtkRenderWindow(); rw.SetOffScreenRendering(1)
    rw.AddRenderer(renF); rw.SetSize(W, H)

    # colorbar: bottom strip (2D) or right strip (3D), on a clean white band
    renB = vtk.vtkRenderer(); renB.SetBackground(1,1,1)
    renB.SetViewport(0.0, 0.0, 1.0, strip/H) if is2d else renB.SetViewport(0.86, 0.0, 1.0, 1.0)
    renB.AddViewProp(scalar_bar(ctf, bar_title or cname, horizontal=is2d, label_fmt=label_fmt))
    if is2d and label:
        renB.AddViewProp(text_label(label, 20, 30))
    rw.AddRenderer(renB)

    if title:
        tt = text_label(title, 20, H-40)
        tt.GetTextProperty().SetFontSize(24)
        renF.AddViewProp(tt)

    rw.Render()
    w2i = vtk.vtkWindowToImageFilter(); w2i.SetInput(rw); w2i.Update()
    pw = vtk.vtkPNGWriter(); pw.SetFileName(out)
    pw.SetInputConnection(w2i.GetOutputPort()); pw.Write()
    if not (os.path.exists(out) and os.path.getsize(out) > 0):
        raise RuntimeError("render produced no image; on headless Linux install "
                           "vtk-osmesa or run under xvfb-run")
    return vmin, vmax
