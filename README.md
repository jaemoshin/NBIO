# Neural Boundary Integral Operators (NBIO)

Research code for **Neural Boundary Integral Operators: Solver-Consistent Learning Across Geometry, Resolution, and Kernel Mismatch**, accepted with minor revision at TMLR.

**Author:** Jae Mo Shin, Seoul National University  
**Contact:** jaemoshin@gmail.com  
**Paper and reviews:** https://openreview.net/forum?id=K8fA2QOlV8

NBIO retains the analytic singular kernel of a boundary integral equation, solves for the boundary density, and evaluates the potential at query points. The zero-correction limit is the classical Nyström solver. The screened-family experiments use learning to represent missing kernel physics, with matched generic, radial, and spline controls.

## Installation

Python 3.10 or newer is recommended. From the repository directory:

```bash
git clone https://github.com/jaemoshin/NBIO.git
cd NBIO
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The checks below run on CPU. CUDA is useful for training and larger evaluations; install a PyTorch build suitable for your hardware.

## Quick checks

```bash
PYTHONPATH=. python scripts/smoke_nbio_paths.py --device cpu
PYTHONPATH=. python scripts/test_reviewer3_spline.py
PYTHONPATH=. python scripts/test_radial_surface_law.py
PYTHONPATH=. python scripts/aggregate_tmlr_final.py
```

The first three commands check small manufactured solver/model paths and algebra. The last command recomputes the seed-replication summary from included result files and writes the aggregate tables; it does not train a model.

## Contents

| Directory | Contents |
|---|---|
| `src/` | Solver layers, kernel corrections, data generation, training, and evaluation utilities |
| `configs/` | Model and experiment configurations |
| `scripts/` | Experiment entry points, structural checks, and result aggregation |
| `experiments/surface3d/` | 3D surface solver and radial-law implementation |
| `results/` | Compact JSON/CSV evidence, including negative controls |
| `checkpoints/` | Selected radial and spline checkpoints, with checksum manifests |

The implementation, results, and selected checkpoints are taken from the reviewed supplementary artifact. Publication preparation did not change their numerical content.

## Data and experiments

The basic manufactured datasets can be regenerated with:

```bash
PYTHONPATH=. python scripts/gen_data.py --help
PYTHONPATH=. python scripts/train.py --help
```

Inspect the corresponding script/configuration before running a full experiment. The screened-fixture generator `scripts/gen_yukawa.py` uses settings defined in the script rather than a command-line parser; inspect those settings before running it. Entry points for the learned screened family include `train_screening_confirmatory.py`, `screening_interval_holdout_v1.py`, `radial_data_efficiency_v1.py`, and `train_reviewer3_spline.py`.

Selected model checkpoints are in `checkpoints/radial_paperization_v1/` and `checkpoints/reviewer3_evidence_v1/`. Check their integrity with:

```bash
(cd checkpoints/radial_paperization_v1 && sha256sum -c SHA256SUMS)
(cd checkpoints/reviewer3_evidence_v1 && sha256sum -c SHA256SUMS)
```

Large generated tensors, many non-core checkpoints, and external CAD meshes are omitted. ABC/Fusion 360 geometry must be obtained from its original providers. Some full experiment entry points therefore require regenerated data or an explicitly supplied checkpoint/data directory; not every full benchmark can be run from the compact artifact alone. The included JSON/CSV files allow inspection and reaggregation of the reported compact results.

## Scope

Known-kernel accuracy mainly reflects solver structure. The learned-law experiments address the validated screened family; they do not establish arbitrary kernel discovery, universal extrapolation, or lower total cost than an optimized classical solver. The released numerical results retain the negative and mixed controls.

## Citation

Please cite the paper linked above when using this implementation. `CITATION.cff` contains the author and paper metadata. The final published citation can be updated after camera-ready verification.

## License

MIT license. See [LICENSE](LICENSE).
