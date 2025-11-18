# Embedding Models Task

## Overview
This repository contains a short embedding-models exercise. Complete your answers in the provided notebook and submit the completed `response.ipynb`.

Repository files:
- [response.ipynb](response.ipynb) — Notebook where you should write your solutions and explanations. You may add additional markdown and Python cells to this file if you feel that this would aid your submission.
- [data.txt](data.txt) — Dataset of sentences used in the task.
- [requirements.txt](requirements.txt) — Python packages used by the task.
- [environment.yml](environment.yml) — Conda environment recipe (name: `embeds`) that installs the requirements.
- [.gitignore](.gitignore) — Files ignored by git.
- [README.md](README.md) — This file.

You may also edit the requirements.txt and environment.yml files to aid setup of your environment, though this is not a necessity. Please do not add/remove files from the repository, unless specifically instructed otherwise.

## How to submit
1. Open and complete [response.ipynb](response.ipynb). Put all code, outputs and written answers inside that notebook.
2. Run all cells so outputs are included.
3. Your submission should be made via appropriate git practices.

Note, your use of git to submit your work consitutes part of the assessment.

## Hardware Requirements
We do not have a preference for the exact hardware you use to complete this task. You should be able to complete this task on fairly minimal hardware - we found no issues when running this on a system with 4GB of RAM and low spec CPU. That being said, if issues with respect to hardware contraints do arise, please do reach out and we will help find a solution which will enable you to complete the task.

## Environment setup

You are welcome to use any programming environment that you are comfortable with. An environment.yml and requirements.txt file have been provided to help setup environments. Generic steps are given below for setting up a conda environment or venv.

Recommended (conda):
1. From the repository root run:
   - `conda env create -f environment.yml`
   - `conda activate embeds`
2. (Optional) Ensure the kernel is available to Jupyter/VSCode:
   - `python -m ipykernel install --user --name=embeds --display-name="embeds"`

Alternative (venv + pip):
1. `python3 -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install --upgrade pip`
4. `pip install -r requirements.txt`
5. `python -m ipykernel install --user --name=embeds --display-name="embeds"`

Run the notebook:
- Launch Jupyter Lab / Notebook: `jupyter lab` or `jupyter notebook`, or open the notebook in VSCode and select the `embeds` kernel.

Notes:
- Work from the repository root so relative paths (e.g., to `data.txt`) resolve correctly.
- The environment installs packages listed in [requirements.txt](requirements.txt) via [environment.yml](environment.yml).
- Keep answers and outputs inside [response.ipynb](response.ipynb). Do not submit separate scripts unless requested.
