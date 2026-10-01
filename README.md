# MuonsAndMatter: muon shield simulation for SHiP

Simulates muons going through the SHiP muon shield (magnetised iron blocks in the TCC8/ECN3 caverns) up to
one or more sensitive planes. There are two simulation engines for the same geometry:

- **Geant4** (`muons_and_matter/`): full simulation, multi-core on CPUs.
- **cuda_muons** (`cuda_muons/`, a git submodule): fast GPU propagation, where energy loss and multiple
  scattering in each step are sampled from histograms made with Geant4. See [cuda_muons/README.md](cuda_muons/README.md).

The magnetic field can be a uniform field per iron block, a field map simulated with the FEM code
[snoopy](https://gitlab.cern.ch/meliebsc/snoopy/), or a field map read from a file (see [Magnetic fields](#magnetic-fields)).

## Repository layout

| Path | Content |
|---|---|
| `muons_and_matter/muons_and_matter.py` | Main Geant4 simulation (multi-core) |
| `muons_and_matter/cuda_muons_ship.py` | GPU simulation with the same detector as the Geant4 one |
| `muons_and_matter/lib/ship_muon_shield.py` | Detector construction (magnets, cavern, target, sensitive planes) and field-map loading |
| `muons_and_matter/lib/magnet_simulations.py` | FEM field simulation with snoopy |
| `muons_and_matter/lib/reference_designs/params.py` | Named magnet parametrisations (`tokanut_v5`, `try_opt`, ...) |
| `muons_and_matter/bin/` | Analysis and helper scripts (plotting, field-map tools, ...) |
| `cpp/` | Geant4 C++ code, built as the Python module `muon_slabs` |
| `cuda_muons/` | GPU simulation (submodule) |
| `data/muons/` | Input muon samples |
| `data/materials/` | Material properties for the FEM simulation |
| `data/outputs/` | Field maps and simulation outputs |
| `plots/` | Plotting scripts and notebooks |

## Setup

### Container (UZH physik cluster)

Everything (Geant4, snoopy, CUDA, Python packages) is in an Apptainer container. Download
`snoopy_geant_slurm.sif` from
[Containers](https://uzh-my.sharepoint.com/:f:/g/personal/luis_felipe_cattelan_physik_uzh_ch/EjWSU34WfZRLiJQ98M3XD58B5BOe7T9fRzW2ffz93Bi9nQ?e=dfgTXF)
and, from the repository root, open a shell in it with

```bash
bash shell_container.sh
```

`shell_container.sh` sets `PROJECTS_DIR` (the parent folder of this repository, used to find
`MuonsAndMatter/data/materials`) and starts a shell that sources `set_env.sh`, which adds
`muons_and_matter/`, `cpp/build/` and `cuda_muons/` to `PYTHONPATH`. On other clusters, adapt the
container path and the `-B` bind mounts (every directory you need must be bound).
Outside the container, run `source set_env.sh` from the repository root and set `PROJECTS_DIR` yourself.

### Builds (once, inside the container)

```bash
git submodule update --init --recursive
bash build_cpp.sh        # Geant4 module (cpp/build/muon_slabs)
```

The GPU extension is installed separately, see [cuda_muons/README.md](cuda_muons/README.md#installation).

On a laptop, the Geant4 binary release works, and the Python packages can be installed with pip
(`requirements.txt`; snoopy only if you need FEM fields).

## Inputs

### Muons

Arrays with columns `[px, py, pz, x, y, z, pdg_id, weight]` (GeV/c and m; `weight` optional; `pdg_id`
±13, or the charge ±1), as `.npy`/`.pkl`, or `.h5` files with datasets `px, py, pz, x, y, z, pdg, weight`.
The samples from the SHiP collaboration are in `data/muons/` (`full_sample.h5`, `full_sample_after_target.h5`).

### Magnet parameters

`-params` takes a name from `muons_and_matter/lib/reference_designs/params.py` or a text file with one number
per line. Each magnet has 15 parameters (cm, except `NI`):

```
zgap, dZ, dXIn, dXOut, dYIn, dYOut, gapIn, gapOut, x_yokeIn, x_yokeOut, dY_yokeIn, dY_yokeOut, midGapIn, midGapOut, NI
```

`zgap` is the gap before the magnet and `dZ` its half-length. In the uniform-field mode, `NI` is the field in
the core (T). For the FEM simulation, `-use_B_goal` (Geant4) / `-NI_from_B` (CUDA) derive the current from
that target field.

## Running

### Geant4

```bash
python3 muons_and_matter/muons_and_matter.py -params tokanut_v5 --f data/muons/full_sample_after_target.h5 --n 100000 --c 45
```

Muons are split over `--c` CPU processes (default 45: adapt to your machine). Main options
(`-h` lists all of them, with defaults):

| Option | Meaning |
|---|---|
| `-sens_plane 82 91` | z positions (m) of the sensitive planes |
| `-field_mode {uniform,read_file,simulate}` | Magnetic field, see [Magnetic fields](#magnetic-fields) (default `simulate`) |
| `-field_file` | Field map file (read or written, default `data/outputs/fields_mm.h5`) |
| `-SC_mag`, `-use_B_goal`, `-diluted_iron` | Hybrid magnets, current from the target field (FEM), diluted iron |
| `-remove_cavern`, `-remove_target`, `-SND`, `-decay_vessel` | Geometry options |
| `-save_data` | Save the hits to `data/outputs/output_<tag>.pkl` |
| `-plot_magnet` | Plot the shield and the muon tracks |

### GPU (cuda_muons) with the same detector

```bash
python3 muons_and_matter/cuda_muons_ship.py -params tokanut_v5 -field_mode uniform --save_dir data/outputs/outputs_cuda.pkl
```

| Option | Meaning |
|---|---|
| `-muons` / `-n_muons` | Input file (default `data/muons/full_sample_after_target.h5`) / maximum number of muons (0 = all) |
| `-sens_plane 82 91` | z positions (m) of the sensitive planes |
| `-field_mode`, `-field_file` | As for Geant4 (default `simulate`; with `simulate`, the map is saved only if `-field_file` is given) |
| `-field_spectrometer` | Extra field map further downstream (e.g. a spectrometer), see below |
| `-NI_from_B`, `-diluted_iron`, `-SND`, `-remove_cavern`, `-expanded_sens_plane` | Same meaning as for Geant4 |
| `-seed` | Random seed (plane `i` uses `seed + i`; `-1` = random) |
| `--n_steps` | Maximum number of steps per plane (default 5000) |
| `--save_dir` | Save the output (pickle) to this path; nothing is saved by default |
| `--gpu` | GPU index |

From Python: `cuda_muons_ship.run_from_params(params, muons, sensitive_plane=..., field_mode=..., ...)` returns a
dict with `px, py, pz, x, y, z, pdg_id` (and `weight`). With `return_all=True`, every input muon is returned in
input order, and the ones that missed a plane have zero momentum.

## Magnetic fields

### Field modes

| `-field_mode` | Field |
|---|---|
| `uniform` | One constant field per iron block, from `NI` (and the yoke dilution); zero in air |
| `read_file` | Field map read from `-field_file`; never simulates or overwrites it |
| `simulate` | Field map simulated with snoopy (slow, needs snoopy); saved to `-field_file`. In Geant4, the main process simulates it once and the workers read the file |

### Field map files

HDF5 files with two datasets:

- `B`: shape `(N, 3)`, field `[Bx, By, Bz]` in T on a regular grid (float16 or float32).
- `d_space`: shape `(3, 3)`, one row `[min, max, step]` (cm) per axis x, y, z.

Conventions (shared by Geant4 and cuda_muons):

- **Only the quadrant x ≥ 0, y ≥ 0 is stored.** Other points are mirrored: the lookup uses `(|x|, |y|, z)`,
  `Bx` changes sign where `x·y < 0`, `Bz` where `y < 0`, `By` never (the symmetry of the shield dipoles).
- **Point order:** y slowest, then x, then z fastest (the order of `np.meshgrid(X, Y, Z)` raveled).
- `max - min` must be a whole number of steps on each axis (checked when reading); the grid has
  `round((max - min)/step) + 1` points per axis.
- The field is taken from the nearest grid point, and is zero outside the grid.

A simulated map covers the magnets in x and y plus 50 cm, and z from −0.5 m to about 2 m after the last
magnet, on a 2 × 2 × 5 cm grid (`RESOL_DEF`).

To check the field-map path against the uniform mode, `muons_and_matter/bin/make_uniform_field_map.py` writes
the uniform per-block fields of a design as a field map (`-field_mode read_file` with it should agree with
`-field_mode uniform`).

### Spectrometer field (`-field_spectrometer`, GPU only)

A second field map further downstream, without having to make one map covering everything. Muons are first
propagated through the magnets (with their field) to a plane 20 cm before the start of the spectrometer map,
then through the sensitive planes with only the spectrometer field. All sensitive planes must be after that
plane (otherwise an error is raised).

The spectrometer map can be a quadrant map (x, y ≥ 0, mirrored as above) or a **full map** covering negative
x and y (e.g. `data/MainSpectrometerField.h5`, a horizontal-field dipole without the shield symmetry). A full map
is used as it is, without mirroring (a warning is printed); the spectrometer stage has no magnet blocks, so the
geometry does not need the symmetry either.

## References

- SHiP software, for the muon shield construction: [FairShip](https://github.com/ShipSoft/FairShip)
- FEM magnetic field simulation: [snoopy](https://gitlab.cern.ch/meliebsc/snoopy/)
