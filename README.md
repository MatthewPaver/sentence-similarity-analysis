# Sentence Similarity Analysis

A demonstration of semantic sentence similarity using transformer-based embedding models. This project explores how modern NLP models can identify semantically similar sentences from a corpus of diverse factual statements.

## Overview

This project performs semantic similarity search using sentence embeddings. Given a target sentence about polar bear fur, the system:

1. **Loads a corpus** of 101 diverse factual sentences
2. **Generates embeddings** using transformer-based sentence encoders
3. **Calculates similarity scores** using cosine similarity
4. **Ranks sentences** by semantic closeness to the target
5. **Analyses results** to understand how embeddings capture semantic meaning

The goal is to identify which sentences in the corpus are most semantically similar to the target sentence, demonstrating how embedding models can be used for semantic search and retrieval tasks.

## Key Features

- **Sentence Embedding Models**: Uses `sentence-transformers/all-mpnet-base-v2`, a state-of-the-art model for semantic similarity
- **Similarity Metrics**: Implements cosine similarity for comparing embedding vectors
- **Model Comparison**: Includes benchmarking across multiple sentence transformer architectures
- **Semantic Analysis**: Explores how embeddings capture meaning versus factual accuracy

## Methodology

### Target Sentence
```
A polar bear's fur is actually transparent, and not white (as is commonly believed).
```

### Approach

1. **Data Preprocessing**: Loads and cleans 101 candidate sentences from the corpus, preserving natural language structure
2. **Embedding Generation**: Converts sentences to fixed-length vector representations using sentence transformers
3. **Similarity Calculation**: Computes cosine similarity between the target sentence embedding and all candidate embeddings
4. **Ranking**: Identifies the top 5 most semantically similar sentences

### Model Choice

The primary model used is **`sentence-transformers/all-mpnet-base-v2`**, chosen because it:
- Performs strongly on semantic textual similarity benchmarks (STS)
- Produces stable sentence-level embeddings
- Is widely adopted as a baseline for retrieval tasks
- Offers a good balance between accuracy and computational cost

**Similarity Metric**: Cosine similarity is used as it focuses on the direction of embedding vectors (semantic content) rather than magnitude, which aligns with how these models are trained.

## Results

The model successfully identifies semantically related sentences, with the top matches being:
- Direct paraphrases of the target sentence
- Sentences about polar bears and fur characteristics
- Related animal facts that share topical vocabulary

The analysis reveals that while embeddings excel at capturing semantic similarity and topical relationships, they should not be used alone for fact-checking, as they prioritise semantic overlap over factual accuracy.

## Repository Structure

- `response.ipynb` — Main notebook containing the analysis, code, and results
- `data.txt` — Dataset of 101 diverse factual sentences
- `requirements.txt` — Python package dependencies
- `environment.yml` — Conda environment configuration

## Setup

### Prerequisites

- Python 3.11+
- Conda (recommended) or pip/venv

### Installation

**Using Conda (recommended):**
```bash
conda env create -f environment.yml
conda activate embeds
python -m ipykernel install --user --name=embeds --display-name="embeds"
```

**Using venv:**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name=embeds --display-name="embeds"
```

### Running the Notebook

Launch Jupyter:
```bash
jupyter lab
# or
jupyter notebook
```

Open `response.ipynb` and select the `embeds` kernel.

**Note**: Work from the repository root so relative paths resolve correctly.

## Dependencies

- `transformers` — Hugging Face transformers library
- `sentence-transformers` — Sentence embedding models
- `torch` — PyTorch for tensor operations
- `pandas` — Data manipulation
- `numpy` — Numerical computations
- `ipykernel` — Jupyter kernel support

## Hardware Requirements

This project runs efficiently on modest hardware:
- **RAM**: 4GB+ recommended
- **CPU**: Standard CPU sufficient (no GPU required)
- **Storage**: ~500MB for models and dependencies

## Insights

This project demonstrates that:

1. **Semantic Similarity ≠ Factual Accuracy**: Embeddings excel at finding topically similar content but may rank factually contradictory statements highly if they share vocabulary
2. **Model Choice Matters**: Different sentence transformer models produce varying rankings, though top results are generally consistent
3. **Cosine Similarity is Appropriate**: For sentence embeddings, cosine similarity effectively captures semantic relationships
4. **Use Cases**: Embeddings are excellent for retrieval and semantic search but require additional verification for fact-checking applications

## License

This project is provided as a demonstration of sentence similarity techniques using transformer embeddings.
