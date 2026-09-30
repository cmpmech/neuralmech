# 2D Benchmark

2D heterogeneous microstructure benchmark for **Chapter 1 (Computational Mechanics
Meets Artificial Intelligence)**: real CT (and other) images of rocks and engineered
materials in `geometry/`, the physics solved on them in `forward/`, and topology optimization
under the same setups in `optimization/`.

## Geometry (`geometry/`)

Each source has its own script writing to `data/2D_benchmark/geometries/<source>/`:
one `uint8` tensor `<TYPE>_<RESOLUTION>.pt` of shape `(N, RESOLUTION, RESOLUTION)`
per material type, where sample `<TYPE>_<RESOLUTION>_<i>` is row `i`, plus an
`index.csv` recording the volume, slice axis, slice index and crop corner of every
sample. Raw volumes are downloaded to `external_data/2D_benchmark/<source>/` and
deleted after extraction unless `KEEP_RAW = True`. Licenses and DOIs are in `sources.csv`.

- `convert_ctscans.py` -> `external_data/ctscans/<name>.npy`
  own CT scans from the original 16-bit tif stacks (`external_data/TriaxCTScans`, a link to
  the project drive) listed in `external_data/ct_scans.csv`; bins the 2000^2 scans 3x and maps
  the core gray range to uint8
- `preprocess_ctscans.py` -> `geometries/ctscans/` _needs `external_data/ctscans/`_
  39 scans, 12 types by name prefix; horizontal and vertical crops inside the cylindrical cores
- `preprocess_sandstones11.py` -> `geometries/sandstones11/`
  11 Kocurek sandstones (DPMP 317), grayscale 1000^3, 128 to 512
- `preprocess_ibm.py` -> `geometries/ibm/`
  18 sandstone and carbonate plugs (Figshare+), first 2500^3 grayscale region, 128 to 1024
- `preprocess_presalt.py` -> `geometries/presalt/`
  16 Brazilian pre-salt carbonates (DPMP 503), high-resolution grayscale, 128 to 1024
- `preprocess_mrccm.py` -> `geometries/mrccm/`
  Indiana limestone and Middle Eastern carbonate (DPMP 362), high-resolution volumes, 128 to 1024
- `preprocess_deeprocksr.py` -> `geometries/deeprocksr/`
  DeepRock-SR 2D slices of sandstone, carbonate and coal (DPMP 215), 128 and 256
- `preprocess_cfrp_twill.py` -> `geometries/cfrp_twill/`
  3 carbon fibre / epoxy twill-weave plates (Zenodo 11200931), crops inside the tilted plates
- `preprocess_alsi10mg.py` -> `geometries/alsi10mg/`
  3 laser powder bed fused AlSi10Mg lattices (Zenodo 17602110, DICOM), crops inside the lattice box
- `preprocess_cement.py` -> `geometries/cement/`
  Portland, Portland-calcite and Portland-fly-ash pastes (Zenodo 2533863, ptychographic, RAR)
- `preprocess_kfoam.py` -> `geometries/kfoam/`
  open-cell graphite foam block (Zenodo 3532935), reconstructed volume from the zip
- `preprocess_syntactic_foam.py` -> `geometries/syntactic_foam/`
  resin with hollow glass spheres, 0.13 g/cm^3 at 10 vol% (Zenodo 19699628)
- `preprocess_gfrp_ud.py` -> `geometries/gfrp_ud/`
  unidirectional glass fibre composite (Zenodo 1195879), fibre cross-sections from three lab-CT
  resolutions and one synchrotron scan of the same material, one type each
- `preprocess_cfrp_t700.py` -> `geometries/cfrp_t700/`
  T700 carbon fibre / epoxy (Zenodo 7632124), fibre cross-sections from the four slow 0.4 um
  synchrotron scans; gray values inverted so fibres (denser) are bright as in the other sets

- `postprocess_geometries.py` _needs `geometries/`_
  shows sample 0 of every type of one source at one resolution; `--book` exports one picked
  slice per source for the Chapter 1 overview figure

## Setups (`bcs/`)

Boundary conditions and sources, shared by `forward/` and `optimization/`, and `pixelmesh.py`,
which maps them onto the dofs of one bilinear element per pixel.

- `presets.py` -> `data/2D_benchmark/setups/<name>_<RESOLUTION>.pt`
  the representative setups: `tension`, `conduction`, `stretch`, `opening`, `burst`, `channel`,
  and for optimization `cantilever`, `heat_sink`
- `generator.py` -> `setups/random<nfields>_<seed>_<i>_<RESOLUTION>.pt`
  random admissible boundary conditions and sources (`--count --nfields --seed`)

## Forward problems (`forward/`)

Every driver solves one geometry (`--source --type --sample --resolution`, grayscale, or
two-phase with `--threshold`) under one setup (`--setup`, boundary conditions and sources)
with material properties interpolated between `--<property>-min` (gray value 0) and
`--<property>-max` (gray value 1). `--save` writes the fields to
`data/2D_benchmark/solutions/<physics>/`, `--book` the figure of the Chapter 1 overview
(run with `--threshold 0.5`, the defaults otherwise).

- `elasticity.py` _needs a setup_
  plane-strain linear elasticity, E from the gray value (`tension`)
- `poisson.py` _needs a setup_
  steady heat conduction, conductivity from the gray value (`conduction`)
- `plasticity.py` _needs a setup_
  plane-strain J2 plasticity with linear hardening, load-stepped Newton; E and yield stress
  from the gray value (`stretch`)
- `fracture.py` _needs a setup_
  AT2 phase-field fracture, staggered, volumetric-deviatoric split; E and Gc from the gray
  value (`opening`)
- `wave.py` _needs a setup and a GPU_
  two-phase acoustic wave equation on cuwave, density and bulk modulus from the gray value
  (`burst`)
- `fluid.py` _needs a setup and a GPU_
  incompressible Navier-Stokes by lattice Boltzmann on cufluid, the weak phase solid and the
  matrix fluid (`channel`, `--reynolds`)

## Optimization problems (`optimization/`)

Every driver runs density-based topology optimization (SIMP, density filter, optimality
criteria) on the full rectangle under one setup (`--setup --resolution --aspect`), with the
settings of `[optimization]` in `settings.toml`. `--save` writes all physical (filtered)
iterates as gray values, like the geometries, with their compliance to
`data/2D_benchmark/optimization/<problem>/<setup>_<RESOLUTION>.pt`; `--book` the iterates
1, 2, 4, ..., 64 for the Chapter 1 figure.

- `compliance.py` _needs a setup_
  minimum compliance, plane-strain linear elasticity (`cantilever`)
- `heat_conduction.py` _needs a setup_
  minimum thermal compliance, steady heat conduction with a uniform source (`heat_sink`)

## Non-obvious technicalities (authored by Claude)

**Physics: one element per pixel.** The drivers mesh the unit square with $R^2$ bilinear
elements, so the dofs sit on the $(R+1)^2$ pixel nodes where the boundary conditions act,
and the per-pixel material (`mlhp.scalarFieldFromVoxelData`) is constant in each element.
`PixelMesh` finds the node of every dof by projecting $x$ and $y$ onto the basis, which
bilinear shape functions reproduce exactly. Dirichlet values enter as mlhp dirichlet dofs,
edge tractions as nodal loads (trapezoidal rule), both read from the setup files.

**Shared setups.** `bcs/` sits beside `forward/` and `optimization/`, and a project directory
starting with a digit cannot be imported as a package, so both helpers append `bcs/` to
`sys.path` before importing `boundary` and `pixelmesh` from it.

**Optimization.** The designs are stored with gray value 1 the strong phase, as the
geometries, so every iterate is a valid geometry of the forward problems; the plots invert
this to the usual solid black of topology optimization. The
setups need homogeneous Dirichlet conditions (the objective $\mathbf{f}^\top\mathbf{u}$ is
the compliance only then); the heat source is `uniform`, independent of the design. The void
bounds, `E = 1e-9` and `kappa = 1e-3`, follow the usual topology optimization choices rather
than the contrast of the forward problems; the filter radius is in pixels, so designs at 128
and 256 differ in their finest members.

**Displacement control.** A prescribed increment is never written into the iterate
directly; the first Newton iteration of a step lifts it (the increment as nonhomogeneous
dirichlet values at the converged state), later iterations are homogeneous. Writing it
into the iterate puts the whole jump into the last element column, and the J2 return
mapping then oscillates instead of converging.

**Phase-field fracture.** Set up as the Chapter 8 reference `8_physics_drivers/fracture2D.py`:
the degraded material, the history $H = \max \psi^+$ and the damage equation are the C
subroutines of `solvers/material_subroutines/phasefield.py`, the history is projected once
per load step, and every staggered iteration does one Newton step of the displacement
followed by one damage solve (both MKL pardiso, matrices allocated once). The run stops
once the reaction falls below 10 % of its peak; ligaments of the heterogeneous body keep
a small residual force after the crack has crossed.

**Wave.** cuwave's `AcousticWave` interpolates $1/\rho$ and $1/\kappa$ between the phases
(indicator 0 = the `min` values, which may exceed the `max` ones), so a grayscale geometry
mixes harmonically, not linearly as the static drivers do. The defaults make the pores
sound-hard holes ($\rho, \kappa$ large, so no flux enters them), the scalar analogue of a
traction-free void. Sound-soft holes ($\rho, \kappa$ small, $p \approx 0$) turn the matrix
channels between the pores into waveguides with a cutoff near $c/(2w)$ for channel width
$w$; at the default frequency the wave then stays trapped near the source. The indicator
lives on the nodes as the maximum of the adjacent pixels, and sources are scaled by it,
so nothing is injected into a hole. Gaussian sources are point sources on the nodes
within four widths.

**Fluid.** cufluid needs a binary solid (grayscale is thresholded at 0.5) and one condition
per face: a fully prescribed edge becomes a (moving) wall, an edge without conditions an
outflow. Fluid pockets without a path to an outflow only accumulate the inflowing mass and
are made solid (8-connected, as D2Q9 streams diagonally). The pore resistance needs a large
pressure drop, which lattice Boltzmann carries as density; at a fixed Reynolds number the
density variation scales with the lattice velocity squared, so the inflow is 0.002 (density
within 3 %) instead of the usual 0.05 (density up to 16 times the rest value).

- The scans are cylindrical cores in air from two scanners: 976 x 976 x (1018-1274) voxels
  at 0.0956 mm and 666 x 666 x (1288-1616) voxels at about 0.090 mm. The core axis and radius are detected per
  scan by thresholding halfway between the air and rock gray values.
- The end caps pass that threshold but are dimmer and noisier (cone-beam edge). The
  usable z range is the contiguous run around mid-height whose median core brightness
  above air reaches `END_BRIGHTNESS` (80%) of the typical slice, walking in from both
  ends so darker layers inside the core are kept, minus `MARGIN`. The core threshold sits at
  25% between air and rock, since the conversion clips air to 0.
- Crops are taken at native voxel size, so the field of view grows with the
  resolution. A square of side $R$ fits in a horizontal slice only if
  $R \le \sqrt{2}\,r$, which limits the ladder to 128 and 256; larger resolutions
  raise an error instead of padding with air.
- Gray values are clipped per scan to the 0.1/99.9 percentiles of the core (a few
  outlier voxels would otherwise dominate a min-max scaling) and mapped to 0-255. A range
  including the air left the rock with only ~40 of the 256 levels.
- Several scans are stitched from parts acquired separately, visible as brightness jumps
  along the core axis. Parts are split where the mean core gray value of neighbouring
  5-slice windows differs by more than `STEP`; vertical crops never cross a split. The 666^2
  (vg-data) scans are equal sub-scans of about 420 binned slices whose jumps (2-10 gray levels)
  are no larger than real rock layering, so they are split at those known positions instead.
  Cupping (darker core centre) and ring artifacts near the rotation axis are left as scanned.
- Samples come from seeded uniform positions and may overlap. The public volumes are
  blocks cut from inside the rock, so crops are drawn along all three axes; a resolution
  that does not fit a volume is skipped for it.
- `helper.py` downloads public DPMP files without login: the portal's download
  endpoint returns a one-time Tapis link. `ILS4.mat` of MRCCM is truncated on the
  server and skipped.
- Specimens surrounded by air (CFRP plates, cement pillars) use `helper.material_mask`: the
  air/material midpoint threshold with enclosed pores filled slice by slice; only crops lying
  entirely inside it are kept. Open structures (AlSi10Mg lattice, KFoam) use `helper.box_mask`,
  the bounding box of the specimen, since their pores reach the surrounding air.
- Fibre composites are cut only normal to the fibres (`extract(..., axes=(0,))`), so every crop
  is a cross-section of packed discs. The glass fibre specimen mask comes from local texture
  (smoothed Laplacian magnitude) inside the field of view, intersected over three depths since
  the fibres are slightly inclined; resin-rich gaps between tows fall outside it.
- `preprocess_cement.py` reads the RAR archives with `libarchive-c` (installed in the venv, not
  in `requirements.txt`); the installed 7z lacks the RAR codec.
- The 11 Sandstones (DPMP 317) and the IBM plugs come from the same group and both
  contain Kocurek sandstones; treat them as one family when splitting train and test.

## Data sources and licensing (notes)

Goal: a dataset of heterogeneous microstructures (rocks, porous or engineered
materials) at 128, 256, 512 and 1024, combining our own CT scans with public
datasets. Only real images are used (CT, but also e.g. electron microscopy), no generated or
synthetic data. The result will
be reshared in modified form, together with simulation results, so every source
license must allow this.

### License summary

- Digital Porous Media Portal (DPMP, formerly Digital Rocks Portal, https://digitalporousmedia.org)
  - Default license is ODC-By 1.0: share, modify and use freely with attribution; derivatives under the same terms.
  - Uploaders can request other licenses, so check each project page.
  - Cite the portal DOI (10.17612/FGMN-D889) plus each dataset's DOI.
  - Public datasets are free to download.
  - User agreement: https://digital-porous-media.github.io/dpm_docs/user_agreement/
- Imperial College pore-scale website downloads: no license is stated, so do NOT
  redistribute them. Use the Figshare copies instead, which carry an explicit license.
- Figshare items: the license is shown per item and is usually CC BY. Verify each one.
- Recommended license for the release: ODC-By for own scans and derived simulation
  fields. Include a per-sample attribution table with source DOIs and each file's license.

### Candidate datasets

Licenses checked 2026-09-23 via the DPMP project API and the Figshare API; the
machine-readable manifest is `sources.csv`. DPMP files download without login via
`/api/datafiles/tapis/download/public/<system>/<path>/`, which returns a one-time link.

| # | Dataset | Material | Size | Grayscale? | Source / DOI | License |
|---|---------|----------|------|-----------|--------------|---------|
| 1 | Neumann et al., 11 Sandstones | 11 sandstones (Berea variants, Bandera, Bentheimer, Kirby, Leopard, Parker, ...) | ~1000^3 | raw + filtered + segmented | DPMP project 317, doi:10.17612/f4h1-w124 | ODC-BY 1.0 (verified) |
| 2 | IBM full-plug set (18 samples) | sandstone + carbonate | 2500x2500x7500 standard volume; 3 x 2500^3 cubes per sample (gray, filtered, binary) | yes | Figshare+ doi:10.25452/figshare.plus.21375565 ; paper: Sci. Data 2023, doi:10.1038/s41597-023-02259-z ; code: github.com/IBM/microCT-Dataset | CDLA-Sharing-1.0 (verified via Figshare API, 2026-09-23): share-alike for the data itself, so crops of it stay CDLA-Sharing; computed results are not restricted |
| 3 | Brazilian pre-salt carbonates (16 samples) | carbonates | low (48-64 um) + high (6-8 um) resolution pairs | yes + segmented | DPMP project 503, doi:10.17612/xr50-s717 ; paper: Sci. Data 2024, doi:10.1038/s41597-024-04198-9 | ODC-BY 1.0 (verified) |
| 4 | MRCCM, multi-resolution complex carbonates | carbonates | multi-resolution | yes | DPMP project 362, doi:10.17612/3T36-Q704 | ODC-BY 1.0 (verified) |
| 5 | DeepRock-SR | sandstone, carbonate, coal | 500^2 slices (2D), 2.7-25 um | yes | DPMP project 215, doi:10.17612/S3M9-E024 | ODC-BY 1.0 (verified) |
| 6 | DRP-372 | 125 real DRP samples across 50+ lithologies (rocks, soils, catalyst layers, meteorites, biofilms, stalagmites, ...) | 256^3 and 480^3 crops | segmented only | DPMP project 372, doi:10.17612/93pd-y471 ; paper: Sci. Data 2022, doi:10.1038/s41597-022-01664-0 | ODC-BY 1.0 (verified) |

Fit for the resolution ladder:
- 1024: datasets 1, 2, possibly 3 and 4.
- 512: datasets 1-4, and 6 (480^3 crops, slightly under 512).
- 128 and 256: all of them.

### Candidates beyond rock (checked 2026-09-23, not yet downloaded)

Suggested in chapter 1 (TomoBank, NIST AM-Bench, Zenodo composites and foams). CC-BY 4.0,
the NIST terms and the Argonne license all allow modification and redistribution with
attribution; CC-BY-SA is share-alike like the IBM set.

| Dataset | Material | Size | Format | License |
|---|---|---|---|---|
| Zenodo 11200931, CFRP twill weave (3 scans) | carbon fibre / epoxy, woven | 3.6 GB | raw volumes | CC-BY 4.0 |
| Zenodo 154714, wind-blade GFRP fatigue | glass fibre / polyester, UD | 17 GB | 8-bit tiff stacks | CC-BY 4.0 |
| Zenodo 7632124, CFRP fast/slow scans | carbon fibre / epoxy | 25 GB | zip of volumes | CC-BY 4.0 |
| Zenodo 18364215 (+ B, C), pultruded CFRP | carbon fibre, synchrotron | 42 GB each | h5 | CC-BY 4.0 |
| Zenodo 4587827, GF-PA66 | short glass fibre / polyamide | 7 GB | h5 | CC-BY-SA 4.0 |
| Zenodo 3532935, KFoam | graphite foam | 16.5 GB | radiographs + reconstruction | CC-BY 4.0 |
| Zenodo 19699628 (+ 2 more), syntactic foam | hollow particles / matrix, 10-40 vol% | 68-80 GB each | reconstructed slices | CC-BY 4.0 |
| Zenodo 14993414, d30 open-pore foam | open-cell foam | 3-32 GB | projections + reconstruction | CC-BY 4.0 |
| Zenodo 17602110, AlSi10Mg L-PBF | additively manufactured aluminium, porosity | 2.8 GB | image stacks | CC-BY 4.0 |
| NIST AM-Bench 2022 IN718 XCT | additively manufactured nickel alloy | large | image stacks | NIST terms (public domain in US) |
| Zenodo 2533863, Portland cement pastes | cement paste (ptychographic) | 13.5 GB | rar | CC-BY 4.0 |
| TomoBank (APS) | various | varies | mostly raw projections | Argonne license (attribution) |

Added first (small): CFRP twill weave, AlSi10Mg, KFoam, cement pastes, syntactic foam
0.13 g/cm^3 at 10 vol% only.

Added: Zenodo 1195879 (multimodal UD glass fibre, 4.3 GB) and Zenodo 7632124 (T700 carbon fibre / epoxy, the four 0.4 um synchrotron scans, 25 GB). Prefer
cross-sections normal to the fibres for these.

Deferred, to discuss (large downloads):
- syntactic foam 0.13 g/cm^3 at 20 and 40 vol% (Zenodo 19699628, 20 + 28 GB) and the
  0.23 / 0.31 g/cm^3 densities (Zenodo 19699671, 19700670, 71 + 80 GB)
- pultruded CFRP samples A, B, C (Zenodo 18364215, 18370051, 18364338, 42 GB each)
- NIST AM-Bench 2022 IN718 XCT (size to check)
- wind-blade GFRP fatigue (Zenodo 154714, 17 GB)
- d30 open-pore foam (Zenodo 14993414, 3-32 GB) and GF-PA66 (Zenodo 4587827, CC-BY-SA)
- TomoBank entries that ship reconstructed volumes
- more fibre-resolved sets: CFRP compact tension (Zenodo 11551777, 13 GB, only the unloaded scan),
  UD fibre beds (Zenodo 4441253, CC0, ~5 GB per scan), UD glass (Zenodo 5483719), pultruded UD
  carbon fibre with constituents (Zenodo 18435237, 9.7 GB)

Not usable: CC-BY-NC(-SA) sets (non-commercial conflicts with ODC-By), the synthetic foam
set (Zenodo 16735632), and TomoBank entries that only ship projections unless we reconstruct.

### Excluded

- Generated or synthetic: MicroLib (SliceGAN), MICRO2D, Santos et al. synthetic binary set.
- DRP-372 (dataset 6 above, removed 2026-09-27): segmented only, and 92 of its 133 volumes come from
  generated or reconstructed projects (374, 204, 57, 65, 69, 276); 11 more duplicate the 11 Sandstones.
- License unclear: Imperial College website images (Berea, C1/C2, S1-S9, Bentheimer,
  Estaillades, Ketton, ...); coal XCT set (Swin-T paper).

### Helper tooling

- github.com/LukasMosser/digital_rocks_data: download and load scripts for the Eleven Sandstones and Imperial sets.
- github.com/digital-porous-media/dpm_tools: DPMP Python tools.

### TODO

1. ~~Confirm each license and record it in a manifest~~ (done, `sources.csv`).
2. Download the data (DPMP web interface or Globus; Figshare+ API).
3. Decide on the resolution strategy for the public data: a) fixed voxel size with a
   growing field of view (crops), or b) a fixed field of view that is downsampled.
   Datasets 3 and 4 provide real multi-resolution pairs.
4. Extract 2D slices and/or 3D crops at 128, 256, 512 and 1024. Keep grayscale and
   segmented versions separate.
5. Run simulations. Store results alongside the inputs with the same sample IDs.
6. Release under ODC-By with a README containing the attribution table plus
   citations of DPMP and each dataset DOI.
