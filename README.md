# Deep Learning in Computational Mechanics <br> <small>an honest reference</small>

> [!NOTE]
> This work is incomplete and under active development.

## About

**neuralmech** is a collection of ml-enhanced physics solvers & optimizers answering

<p align="center"><b><i>When and where is deep learning useful in numerical simulation?</i></b></p>
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset=".assets/images/logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset=".assets/images/logo_light.png">
    <img src=".assets/images/logo_light.png" alt="NeuralMech" width="500">
  </picture>
</p>

**Features**

- simplistic & extendable implementations
- reproducible results
- associated to the succesor of [**deep learning in computational mechanics**](https://link.springer.com/book/10.1007/978-3-031-89529-6): a completely new book growing out of the earlier edititions:

<table><tr>
  <td><img src=".assets/images/book2.png" alt="deep learning in computational mechanics book 2" width="150"></td>
  <td><img src=".assets/images/book1.png" alt="deep learning in computational mechanics book 1" width="150"></td>
</tr></table>

As this project is ongoing, feedback is highly welcome. Finished chapters are available on request under a **personal-use, non-redistribution license** — feel free to reach out via [email](#contact). By requesting a copy, you agree to the license terms included in the document.

<details>
<summary><b>Chapters available on request</b></summary>

- **Computational Mechanics Meets Artificial Intelligence** (chapter 1)
- **Fundamental Machine Learning** (chapter 2)
- **Artificial Neural Networks** (chapter 3)

<img src=".assets/images/expressivity.jpg" alt="neural network expressivity" width="300">

- **Neural Network Architectures** (chapter 4)
- **Probabilistic Machine Learning** (chapter 5)
- **Governing Equations** (chapter 8)

</details>

<details>
<summary><b>Chapters in progress</b></summary>

- **Machine Learning Algorithms** (chapter 6)
- **Practical Machine Learning** (chapter 7)
- **Numerical Methods** (chapter 9)
- **Machine Learning in Computational Mechanics** (chapter 10)
- **Neural Surrogates** (chapter 11)
- **Neural Solvers** (chapter 12)
- **Physics-Informed Neural Networks** (chapter 13)
- **Constitutive Modeling with Neural Networks** (chapter 14)
- **Generative Artificial Intelligence** (chapter 15)
- **Neural Optimization** (chapter 16)

<img src=".assets/images/fwi.png" alt="full waveform inversion" width="300">

- **Large Language Models** (chapter 17)
- **Simulation Acceleration via GPUs** (chapter 18)

<img src=".assets/images/strains.png" alt="matrix-free finite element method on GPU" width="500">

- **Deep Reinforcement Learning** (chapter 19)
- **Computational Mechanics After Artificial Intelligence** (chapter 20)

</details>

## Requirements

- install via requirements

```
pip install -r requirements.txt
```

> [!NOTE]
> The requirements are currently broader than necessary and will be trimmed down later.

### mlhp

[mlhp](https://gitlab.com/hpfem/code/mlhp) is included as a git submodule. To clone recursively use

```
git clone --recurse-submodules https://github.com/cmpmech/neuralmech
```

- for now, use `pip install mlhp` ([for more advanced physics](solvers/README.md#building-mlhp-from-source) C++ compilation will be needed)

### cuwave

- the GPU finite difference wave solver behind the wave and transient topology optimization drivers
- covered by the requirements, or install it on its own with

```
pip install cuwave
```

- needs a [cupy](https://cupy.dev) matching the installed CUDA toolkit

## Structure

|                                          |                                                                            |
| ---------------------------------------- | -------------------------------------------------------------------------- |
| [`data/`](data/)                         | generated data (small, but gitignored if large)                            |
| [`external_data/`](external_data/)       | data generation tools with data in `data` (large, excluded from main repo) |
| [`models/`](models/)                     | trained networks                                                           |
| [`results/`](results/)                   | results for postprocessing                                                 |
| [`projects/`](projects/)                 | main drivers; see [projects](projects/README.md)                           |
| [`DL.py`](DL.py)                         | deep learning utilities                                                    |
| [`NN.py`](NN.py)                         | network architectures                                                      |
| [`ML.py`](ML.py)                         | classical machine learning models                                          |
| [`postprocessing.py`](postprocessing.py) | postprocessing helpers                                                     |
| [`solvers/`](solvers/)                   | classical physics solvers                                                  |

## Contact

[neuralmech@pm.me](mailto:neuralmech@pm.me)

## License

MIT; see [LICENSE](LICENSE).
