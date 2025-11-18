# Sentence Similarity Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3670A0?style=flat&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗_Transformers-FFD21E?style=flat&logo=huggingface&logoColor=000000)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat)

**Semantic Sentence Similarity Using Transformer Embeddings**

*A demonstration of how modern NLP models identify semantically similar sentences*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Methodology](#-methodology)
- [Results & Insights](#-results--insights)
- [Getting Started](#-getting-started)
- [Repository Structure](#-repository-structure)
- [Dependencies](#-dependencies)
- [Hardware Requirements](#-hardware-requirements)

---

## 🎯 Overview

This project demonstrates semantic sentence similarity using transformer-based embedding models. Given a target sentence about polar bear fur, the system identifies which sentences from a corpus of 101 diverse factual statements are most semantically similar.

### What This Project Does

1. **Loads a corpus** of 101 diverse factual sentences
2. **Generates embeddings** using transformer-based sentence encoders
3. **Calculates similarity scores** using cosine similarity
4. **Ranks sentences** by semantic closeness to the target
5. **Analyses results** to understand how embeddings capture semantic meaning

### Use Cases

- Semantic search and retrieval
- Document similarity analysis
- Content recommendation systems
- Duplicate detection
- Understanding how embeddings represent meaning

---

## ✨ Key Features

- **State-of-the-Art Models**: Uses `sentence-transformers/all-mpnet-base-v2`, a top-performing model for semantic similarity
- **Multiple Model Comparison**: Benchmarks across different sentence transformer architectures
- **Semantic Analysis**: Explores how embeddings capture meaning versus factual accuracy
- **Cosine Similarity**: Implements cosine similarity for comparing embedding vectors
- **Comprehensive Analysis**: Detailed exploration of embedding behaviour and limitations

---

## 🔬 Methodology

### Target Sentence

```
A polar bear's fur is actually transparent, and not white (as is commonly believed).
```

### Approach

#### 1. Data Preprocessing
- Loads and cleans 101 candidate sentences from the corpus
- Preserves natural language structure
- Handles diverse factual statements

#### 2. Embedding Generation
- Converts sentences to fixed-length vector representations
- Uses sentence transformer models for encoding
- Generates embeddings suitable for similarity comparison

#### 3. Similarity Calculation
- Computes cosine similarity between target and all candidate embeddings
- Cosine similarity focuses on direction (semantic content) rather than magnitude
- Aligns with how sentence transformer models are trained

#### 4. Ranking
- Identifies the top 5 most semantically similar sentences
- Provides ranked list with similarity scores
- Enables analysis of what makes sentences semantically similar

### Model Choice

The primary model used is **`sentence-transformers/all-mpnet-base-v2`**, chosen because it:

- ✅ Performs strongly on semantic textual similarity benchmarks (STS)
- ✅ Produces stable sentence-level embeddings
- ✅ Is widely adopted as a baseline for retrieval tasks
- ✅ Offers a good balance between accuracy and computational cost

**Similarity Metric**: Cosine similarity is used as it focuses on the direction of embedding vectors (semantic content) rather than magnitude, which aligns with how these models are trained.

---

## 📊 Results & Insights

### Key Findings

The model successfully identifies semantically related sentences, with the top matches being:
- Direct paraphrases of the target sentence
- Sentences about polar bears and fur characteristics
- Related animal facts that share topical vocabulary

### Important Insights

This project demonstrates that:

1. **Semantic Similarity ≠ Factual Accuracy**
   - Embeddings excel at finding topically similar content
   - May rank factually contradictory statements highly if they share vocabulary
   - Should not be used alone for fact-checking

2. **Model Choice Matters**
   - Different sentence transformer models produce varying rankings
   - Top results are generally consistent across models
   - Model selection depends on use case requirements

3. **Cosine Similarity is Appropriate**
   - For sentence embeddings, cosine similarity effectively captures semantic relationships
   - Focuses on semantic content rather than magnitude
   - Standard approach for embedding-based similarity

4. **Use Cases**
   - ✅ Excellent for retrieval and semantic search
   - ✅ Great for finding topically related content
   - ⚠️ Requires additional verification for fact-checking applications
   - ⚠️ May prioritise semantic overlap over factual accuracy

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Conda (recommended) or pip/venv
- Jupyter Notebook or JupyterLab

### Installation

#### Using Conda (Recommended)

```bash
# Create environment from YAML file
conda env create -f environment.yml

# Activate the environment
conda activate embeds

# Install Jupyter kernel
python -m ipykernel install --user --name=embeds --display-name="embeds"
```

#### Using venv

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install Jupyter kernel
python -m ipykernel install --user --name=embeds --display-name="embeds"
```

### Running the Notebook

1. **Launch Jupyter:**
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

2. **Open `response.ipynb`** and select the `embeds` kernel

3. **Run all cells** to execute the analysis

**Note**: Work from the repository root directory so relative paths resolve correctly.

---

## 📁 Repository Structure

```
sentence-similarity-analysis/
├── response.ipynb          # Main notebook containing analysis, code, and results
├── data.txt                # Dataset of 101 diverse factual sentences
├── requirements.txt        # Python package dependencies
├── environment.yml         # Conda environment configuration
└── README.md               # This file
```

---

## 📚 Dependencies

### Core Libraries

- **transformers** — Hugging Face transformers library for model access
- **sentence-transformers** — Sentence embedding models and utilities
- **torch** — PyTorch for tensor operations and model execution
- **pandas** — Data manipulation and analysis
- **numpy** — Numerical computations
- **ipykernel** — Jupyter kernel support

### Installation

All dependencies are listed in `requirements.txt` and `environment.yml`. Install using:

```bash
# Using pip
pip install -r requirements.txt

# Using conda
conda env create -f environment.yml
```

---

## 💻 Hardware Requirements

This project runs efficiently on modest hardware:

- **RAM**: 4GB+ recommended (models load into memory)
- **CPU**: Standard CPU sufficient (no GPU required)
- **Storage**: ~500MB for models and dependencies

The `all-mpnet-base-v2` model is relatively lightweight and runs well on CPU, making this project accessible without specialised hardware.

---

## 💻 Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3670A0?style=flat&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗_Transformers-FFD21E?style=flat&logo=huggingface&logoColor=000000)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

</div>

---

## 📖 Additional Resources

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Hugging Face Sentence Transformers](https://huggingface.co/sentence-transformers)
- [Semantic Textual Similarity Benchmarks](https://github.com/brmson/dataset-sts)
- [Understanding Embeddings](https://www.pinecone.io/learn/embeddings/)

---

## 📝 License

This project is provided as a demonstration of sentence similarity techniques using transformer embeddings.

---

## 🤝 Contributing

This is a personal project, but suggestions and feedback are welcome! Feel free to open an issue or submit a pull request.

---

<div align="center">

**Exploring how embeddings capture semantic meaning**

[⬆ Back to Top](#sentence-similarity-analysis)

</div>
