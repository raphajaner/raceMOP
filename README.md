# RaceMOP

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

This repository is the official implementation of the [paper](https://scholar.google.com/citations?view_op=view_citation&hl=de&user=2ttMbLQAAAAJ&citation_for_view=2ttMbLQAAAAJ:WF5omc3nYNoC):

> **RaceMOP: Mapless Online Path Planning for Multi-Agent Autonomous Racing using Residual Policy Learning**
>
> [Trumpp, Raphael](https://scholar.google.com/citations?user=2ttMbLQAAAAJ&hl=en), 
> [Ehsan Javanmardi](https://scholar.google.com/citations?user=7kusI58AAAAJ&hl=en&oi=ao),
> [Jin Nakazato](https://scholar.google.com/citations?user=7Y1CFxMAAAAJ&hl=en),
> [Manabu Tsukada](https://scholar.google.com/citations?user=GSfYxPIAAAAJ&hl=en),
> and [Marco Caccamo](https://scholar.google.com/citations?user=Jbo1MqwAAAAJ&hl=en&oi=ao).

Presented at: 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS).

<p align="center">
  <img src="docs/anim_1.gif" alt="Replicated real-world racetracks." width="200" />
  <img src="docs/anim_2.gif" [main.py](../icra_2023/main.py)alt="Replicated real-world racetracks." width="200" />
  <img src="docs/anim_3.gif" alt="Replicated real-world racetracks." width="200" />
</p>

## Table of contents
- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Reference](#reference)
- [License](#license)

## Background
The interactive decision-making in multi-agent autonomous racing offers insights valuable beyond the domain of self-driving cars.
Mapless online path planning is particularly of practical appeal but poses a challenge for safely overtaking opponents due to the limited planning horizon.
Accordingly, this paper introduces RaceMOP, a novel method for mapless online path planning designed for multi-agent racing of F1TENTH cars.
Unlike classical planners that depend on predefined racing lines, RaceMOP operates without a map, relying solely on local observations to overtake other race cars at high speed.
Our approach combines an artificial potential field method as a base policy with residual policy learning to introduce long-horizon planning capabilities.
We advance the field by introducing a novel approach for policy fusion with the residual policy directly in probability space.
Our experiments for twelve simulated racetracks validate that RaceMOP is capable of long-horizon decision-making with robust collision avoidance during overtaking maneuvers.
RaceMOP demonstrates superior handling over existing mapless planners while generalizing to unknown racetracks, paving the way for further use of our method in robotics.

<p align="center">
  <img src="docs/cover.png" alt="Architecture of the residual controller." width="400" />
</p>

## Install
- Results in the paper where obtained using Python 3.8.10; 3.10 seems to work too.
- We recommend to use a virtual environment for the installation:
  ```bash
  python -m venv racemop_env
  source racemop_env/bin/activate
  ```
- Activate the environment and install the following packages:
    ```bash
  pip install torch==2.0.1
  pip install tensordict==0.1.2
  pip install torchrl==0.1.1
  pip install numpy==1.24.4
  pip install torchinfo
  pip install matplotlib
  pip install gymnasium==0.29.1
  pip install hydra-core
  pip install tqdm
  pip install flatdict
  pip install wandb
  pip install numba
  pip install cvxpy
  pip install pyclothoids
  pip install --upgrade "jax[cpu]"
  pip install tensorboard
  pip install scikit-learn
  pip install pandas
  pip install termcolor
  pip install pyglet==1.5
    ```
- The simulator should be installed as a module:
    ```bash
  pip install -e simulator/
    ```
## Usage
### Inference
You can start evaluating the provided agent by running the following command:
```bash
python main.py
```
The use of your GPU can be avoided by running:
```bash
python main.py device=cpu
```
### Rendering
Rendering can be enabled by setting running:
```bash
python main.py render=True
```
Since this will launch windows for all 12 maps, we recommend selecting a specific map for evaluation:
```bash
python main.py mode=inference render=True maps.maps_train=[Catalunya] maps.maps_test=[]
```

### Others
The baseline controller can be evaluated by running:
```bash
python main.py mode=baseline
```

### Docstrings
Most of the code is documented with *automatically* generated docstrings, please use them with caution.

## Reference
If you find our work useful, please consider citing our paper:

```bibtex 
@inproceedings{trumpp2024racemop,
  title={RaceMOP: Mapless online path planning for multi-agent autonomous racing using residual policy learning},
  author={Trumpp, Raphael and Javanmardi, Ehsan and Nakazato, Jin and Tsukada, Manabu and Caccamo, Marco},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  year={2024},
  organization={IEEE}
}

```

## License
[GNU General Public License v3.0 only" (GPL-3.0)](LICENSE.txt) © [raphajaner](https://github.com/raphajaner)
