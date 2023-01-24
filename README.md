Thesis pilot: Adaptive Flight Controller using Reinforcement Learning
==============================

This repository contains the code and documentation for a thesis research project that aims to advance the state-of-the-art in adaptive flight control through Reinforcement Learning.

Aircraft control systems are crucial for safety and must always keep the aircraft controllable, regardless of any environment or system disruptions. One way to improve flight safety and stability is through adaptive flight controllers, which can adjust their behaviour in response to changing circumstances. Reinforcement learning is a promising technique for designing these controllers.

## Research Aim 🛩️🤖
The research aims to develop a cascaded flight controller that uses a hybrid offline and online learning approach to provide robustness and adaptiveness to aircraft control. The goal is to create a controller that can still function effectively under failure and adverse conditions.
Preliminary work for my control systems with RL MSc thesis

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/iamlucassantos/thesis_pilot/HEAD)


## Setup Instructions 🛠️
To run the code in this repository, you will need to create a Python environment as follows:
1. Create a conda environment \
```conda env create -f environmnet.yml -n thesis-pilot``` 
2. Activate the environment\
```conda activate thesis-pilot```
3. Add the conda environment to Jupiter kernel \
```python3 -m ipykernel install --user --name tesis-pilot --display-name "Python 3.9 (thesis_pilot)```


## Repository contents 📚
The repository contains the following:

- Code for the cascaded flight controller and reinforcement learning algorithms used in the research
- Data and results from simulations and experiments
- Thesis document (in PDF format)
- Presentation slides used for thesis defense

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

## Contact 📞 
If you have any questions or need help with the code or research, please contact me at [lucas6eng@gmail.com](mailto:lucas6eng@gmail.com).

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
