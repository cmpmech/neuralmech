#!/usr/bin/env python
"""Render one mlhp VTU/PVTU file to a PNG (headless vtk, no ParaView).

Examples
--------
  # 2D field, colorbar in a bottom strip, frame filled (no whitespace):
  python render_vtu.py result.vtu --field VonMisesStress --out vm.png

  # vector magnitude of a 2D solution:
  python render_vtu.py flow.pvtu --field mag:Solution --out umag.png

  # 3D deformation: warp by Displacement, colour by von Mises (Pa->MPa),
  # overlay the deformed mesh edges, isometric view:
  python render_vtu.py solid_5.pvtu --field VonMisesStress --scale 1e-6 \
      --warp Displacement --view iso --edges edges_5.pvtu \
      --bar-title "vM [MPa]" --out step5.png

Then open/Read the PNG. For a quick self-check, that's all you need.
"""
import argparse
import mlhp_visualize as mlhpviz


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="VTU/PVTU/VTP file")
    p.add_argument("--out", default="render.png", help="output PNG (default render.png)")
    p.add_argument("--field", required=True,
                   help="array to colour by: NAME | mag:NAME | comp:NAME:i")
    p.add_argument("--view", default="2d",
                   choices=["2d", "iso", "+x", "-x", "+y", "-y", "+z", "-z"],
                   help="2d = filled parallel projection (2D data); others are 3D cameras")
    p.add_argument("--warp", default=None, metavar="VECNAME",
                   help="warp geometry by this vector field (e.g. Displacement)")
    p.add_argument("--warp-scale", type=float, default=1.0)
    p.add_argument("--range", default="auto",
                   help="'auto' | 'pNN' (0..NNth percentile, caps cut-cell spikes) | 'LO,HI'")
    p.add_argument("--scale", type=float, default=1.0, help="multiply field values (e.g. 1e-6 Pa->MPa)")
    p.add_argument("--cmap", default="turbo", choices=list(mlhpviz._CMAPS))
    p.add_argument("--edges", default=None, help="overlay this edge/mesh file as a black wireframe")
    p.add_argument("--show-edges", action="store_true", help="draw feature edges of the input")
    p.add_argument("--title", default=None)
    p.add_argument("--bar-title", default=None, help="colorbar title (defaults to field name)")
    p.add_argument("--label", default=None, help="corner label, e.g. a time/step stamp")
    p.add_argument("--size", default=None, help="WxH, e.g. 1600x900")
    p.add_argument("--label-fmt", default="%.2f", help="colorbar tick format (default %%.2f)")
    a = p.parse_args()

    rng = a.range
    if "," in rng:
        lo, hi = rng.split(","); rng = (float(lo), float(hi))
    size = tuple(int(x) for x in a.size.lower().split("x")) if a.size else None

    vmin, vmax = mlhpviz.render(
        a.input, a.out, a.field, view=a.view, warp_field=a.warp, warp_scale=a.warp_scale,
        rng=rng, scale=a.scale, cmap=a.cmap, title=a.title, bar_title=a.bar_title,
        label=a.label, edges=a.edges, show_edges=a.show_edges, size=size, label_fmt=a.label_fmt)
    print(f"wrote {a.out}  (range [{vmin:.4g}, {vmax:.4g}])")


if __name__ == "__main__":
    main()
