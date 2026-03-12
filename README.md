# Sentence Similarity Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3670A0?style=flat&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-FFD21E?style=flat&logo=huggingface&logoColor=000000)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat)

**Semantic similarity demo using sentence-transformer embeddings**

*Notebook-first project exploring how modern embedding models rank related sentences*

</div>

---

## Overview

This repo is a compact NLP experiment rather than a packaged application. It uses transformer-based sentence embeddings and cosine similarity to rank a corpus of 101 factual statements against a target sentence about polar bear fur.

The point of the project is not just to produce a ranking, but to show what embedding-based similarity does well and where it can mislead. In particular, semantically similar text is not the same thing as factually correct text.

## What It Covers

- sentence embeddings with `sentence-transformers`
- cosine similarity for ranking related text
- notebook-based inspection of the top matches
- discussion of semantic similarity versus factual accuracy

## Quick Start

### Option 1: `venv`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name=embeds --display-name="embeds"
jupyter lab
```

Open `response.ipynb`, select the `embeds` kernel, and run the notebook from the repository root.

### Option 2: Conda

```bash
conda env create -f environment.yml
conda activate embeds
python -m ipykernel install --user --name=embeds --display-name="embeds"
jupyter lab
```

## Repository Contents

```text
response.ipynb     Main notebook with the analysis
data.txt           Corpus of 101 candidate sentences
requirements.txt   Pip dependencies
environment.yml    Conda environment
```

## Notes

- Status: completed notebook demo
- Hardware: CPU is sufficient
- Main model: `sentence-transformers/all-mpnet-base-v2`
- Main takeaway: embedding similarity is useful for retrieval and clustering, but not reliable by itself for fact-checking

## Related Work

- [MatthewPaver profile](https://github.com/MatthewPaver)
- [Project Index](https://github.com/MatthewPaver/MatthewPaver/blob/main/Projects.md)
