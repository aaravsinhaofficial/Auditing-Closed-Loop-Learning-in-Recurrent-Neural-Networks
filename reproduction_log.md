# Original Artifact Reproduction Log

Artifact path: `external/original_artifact`

## Static Inventory
- `code/figures.ipynb`: present
- `code/train_non_linear.ipynb`: present
- `code/theory.ipynb`: present
- `code/tracking_task.ipynb`: present
- `code/utils.py`: present
- `requirements.txt`: present

## Dependency Notes
numpy==2.2.6
scipy==1.15.3
torch==2.7.0

## Saved Data Load Check
- `adam_all_loss.pkl`: loaded as `ndarray`
- `closed_loop_tracking_data.pkl`: loaded as `dict`
- `data_closed.pkl`: loaded as `tuple`
- `eig_all_dict.pkl`: loaded as `dict`
- `non_linear_all_k_eff.pkl`: loaded as `ndarray`
- `non_linear_closed.pkl`: loaded as `tuple`
- `non_linear_open.pkl`: loaded as `tuple`
- `saved_arrays.pkl`: loaded as `dict`

## Figure Status
| Figure source | Status | Notes |
|---|---:|---|
| `figures.ipynb` | partial | Uses precomputed pickle files; scriptable refactor provided in this repo. |
| `train_non_linear.ipynb` | not run by default | Writes checkpoints under relative `models/`; no explicit seeds in original notebook. |
| `theory.ipynb` | not run by default | Contains low-rank/effective-model analysis converted into package tests and metrics. |
| `tracking_task.ipynb` | not run by default | Long checkpoint-heavy tracking run; config-driven replacement included. |

Use `jupyter nbconvert --execute` manually for exact notebook execution if desired.
