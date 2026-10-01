# mlhp output API

How to make mlhp emit visualizable data — the public API at a glance, so the
common cases are one lookup away. The `examples/` and headers stay
authoritative for anything not covered here. Python names first, C++ in §C++.

## Entry points (Python)

- `basisOutput(basis, cellmesh, output, processors)` — fields that need the
  basis + dof vector (solution, stress, …).
- `meshOutput(mesh, cellmesh, output, processors)` — mesh-only / cell data.

`output` is a target, `processors` is a list (see below), `cellmesh` controls
subdivision and which entities are written.

## Output targets

- `VtuOutput(filename="out.vtu", writemode=...)` — one serial `.vtu`.
- `PVtuOutput(filename="out", ...)` — parallel: `out.pvtu` master + `out/out_0.vtu`,
  `out_1.vtu`, … one piece per partition. Render the `.pvtu`.
- `DataAccumulator()` — **no file**, keeps the result in memory (see in-memory §).
- `writemode` ∈ `Ascii | Base64Inline | Base64Appended | RawBinary |
  RawBinaryCompressed` (default compressed; needs zlib — the "ZLIB not found"
  configure warning just disables compression, harmless).

## cellmesh — subdivision and entities

- `gridCellMesh(resolution, topologies=default)` — subdivide each element
  `resolution` times per axis.
- `degreeOffsetResolution(basis, offset=2, exceptLinear=True)` — resolution
  follows polynomial degree. **One output cell per element under-resolves p>1
  fields** — use this (or a higher `gridCellMesh` resolution) to actually see
  high-order detail.
- `domainCellMesh(fn, resolution, ...)` / `boundaryCellMesh(fn, resolution, ...)`
  — marching-cubes interior / boundary surface of an implicit domain.
  **`domainCellMesh` keeps only the physical material**, so an immersed hole /
  fictitious region appears as a gap in the output — this is how you restrict
  FCM output to the real domain (and avoid plotting meaningless cut-cell values).
- `PostprocessTopologies`: `Corners | Edges | Faces | Volumes` (OR-combine).
  Use `Edges` alone to output a mesh wireframe (overlay it on a solid with
  `render_vtu.py --edges`). **Trap:** a topology above the dimension (e.g.
  `Volumes` in 2D) silently writes 0 cells / 0 points.

## processors — what field gets written

The list order is the **array order in the file** and the index in
`DataAccumulator.data()`. Name each processor; the name becomes the VTK array
name you pass to `--field`.

- `solutionProcessor(D, dofs, name="Solution")` — writes one **multi-component**
  array (e.g. velocity components then pressure, or the displacement vector).
  Split components when rendering (`--field comp:Solution:0` / `mag:Solution`),
  or warp by it (`--warp Displacement`).
- `vonMisesProcessor(dofs, kinematics, constitutive, name="VonMisesStress")`
- `stressProcessor` / `strainProcessor` / `strainEnergyProcessor(…)`
- `functionProcessor(field_or_implicit, name)` — an implicit domain writes 0/1.
- `cellDataProcessor(D, data, name)`, `cellIndexProcessor(D, …)`,
  `basisFunctionProcessor(D, indices, …)`, `kdTreeInfoProcessor(D)`.

Stresses are in the model's units (Pa if E is in Pa) — pass `--scale 1e-6` to
the renderer for MPa and set `--bar-title "vM [MPa]"`.

## In-memory path (Python, 2D, no files)

Fastest 2D self-check; writes only your matplotlib PNG.

```python
result = mlhp.DataAccumulator()
mlhp.basisOutput(basis, mlhp.domainCellMesh(domain, [degree+2]*2), result, processors)
tri = result.triangulation()        # matplotlib.tri.Triangulation (quads -> tri pairs)
vm  = result.data()[1]              # k-th array == k-th processor; here VonMisesStress
ax.tricontourf(tri, vm, levels=24, cmap="turbo")
```

`result.data()[k]` is the k-th processor's values; `triangulation()` is 2D only.
For a stable color scale (esp. with cut cells) cap with a percentile, e.g.
`levels=np.linspace(0, np.percentile(vm, 99), 24)`.

## C++

Same concepts in `include/mlhp/core/postprocessing.hpp` + `cellmesh.hpp`:

```cpp
writeOutput( basis /*or mesh*/, cellmesh::grid( resolution ),
             std::tuple{ makeSolutionProcessor<D>( dofs ),
                         makeVonMisesProcessor<D>( dofs, kinematics, constitutive ) },
             VtuOutput{ "out" } /*or PVtuOutput{ "out" }*/ );
```

- Processor factories: `makeSolutionProcessor`, `makeVonMisesProcessor`,
  `makeStressProcessor`, `makeStrainProcessor`, `makeFunctionProcessor`,
  `makeCellDataProcessor`, `mergeProcessors`. Also `writeVtu(mesh, "mesh.vtu")`,
  `writeStl(triangulation, …)`.
- cellmesh: `cellmesh::grid`, `cellmesh::domain`, `cellmesh::boundary`,
  `degreeOffsetResolution`. Same `PostprocessTopologies`.
- There is no C++ renderer and none is needed: render the `.vtu`/`.pvtu` the
  C++ run produced with the bundled Python `render_vtu.py` / `make_video.py`.
