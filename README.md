# Deep Learning in Computational Mechanics <br> <small>a comprehensive reference</small>

## about

## submodules
[mlhp](https://gitlab.com/hpfem/code/mlhp) is included as git submodule. To clone recursively use
```
git clone --recurse-submodules https://github.com/Leon-Herrmann/neuralmech
```
### installation of mlhp
- TODO change branch in mlhp

## main structure
- `data/` - generated data (small, but gitignored if large)
- `external_data/` - data generation tools with data in `data` (large, excluded from main repo)
- `models/` - trained networks
- `results/` - results for post-processing
- `projects/` - main drivers
- `templates/` - elements with repeated use (e.g., `training_loop.py`)
- `DL.py` - deep learning utilities
- `NN.py` - network architectures
- `solvers/` - classical physics solvers

## projects
### chapter 2: ml introduction (`2_intro_ml`)
- `linear_regression.py`[.gitignore](.gitignore)
- `logistic_regression.py`
