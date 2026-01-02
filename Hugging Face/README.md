# Hugging Face Transformers 🤗

This module covers practical NLP using the **Hugging Face Transformers** library — the most popular framework for working with state-of-the-art pretrained language models.

---

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [Topics Covered](#topics-covered)
3. [Key Concepts](#key-concepts)
4. [Prerequisites](#prerequisites)
5. [Getting Started](#getting-started)
6. [Directory Structure](#directory-structure)

---

## Introduction

Hugging Face provides a unified API to access thousands of pretrained models for various NLP tasks. The `transformers` library makes it incredibly easy to:

- Perform **inference** on text using pretrained models
- **Tokenize** text for model consumption
- **Fine-tune** models on custom datasets for specific tasks

This module takes you from using simple pipelines to understanding the underlying components and finally training your own models.

---

## Topics Covered

### 1. Pipelines 🚀

📖 **Documentation:** [docs/pipelines.md](docs/pipelines.md)  
📓 **Notebook:** [notebooks/pipelines.ipynb](notebooks/pipelines.ipynb)

High-level abstractions for instant NLP capabilities:

- Sentiment Analysis
- Language Translation
- Zero-Shot Classification
- Text Generation
- Named Entity Recognition (NER)

```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
classifier("I love Hugging Face!")
```

---

### 2. Tokenizers 🔤

📖 **Documentation:** [docs/tokenizers.md](docs/tokenizers.md)  
📓 **Notebook:** [notebooks/hf_tokenizer.ipynb](notebooks/hf_tokenizer.ipynb)

Understanding how text becomes model input:

- Subword Tokenization (WordPiece, BPE)
- Special Tokens (`[CLS]`, `[SEP]`, `[PAD]`)
- Padding & Truncation strategies
- Batch Processing

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer("Hello world", return_tensors="pt")
```

---

### 3. Model Fine-Tuning 🎯

📖 **Documentation:** [docs/model-finetuning.md](docs/model-finetuning.md)  
📓 **Notebook:** [notebooks/model_finetuning.ipynb](notebooks/model_finetuning.ipynb)

Training models on your own data:

- Loading & Preprocessing Datasets
- Dynamic Padding with Data Collators
- Training Configuration
- Evaluation Metrics
- Making Predictions

```python
from transformers import Trainer, TrainingArguments
trainer = Trainer(model=model, args=training_args, ...)
trainer.train()
```

---

## Key Concepts

### The Hugging Face Ecosystem

| Component         | Description                                 |
| ----------------- | ------------------------------------------- |
| `transformers`    | Core library for models and tokenizers      |
| `datasets`        | Library for loading and processing datasets |
| `huggingface_hub` | Access to 100,000+ pretrained models        |
| `Trainer`         | High-level API for training and evaluation  |

### Model Architecture Flow

```text
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Raw Text   │ →  │  Tokenizer  │ →  │   Model     │ →  │   Output    │
│             │    │  (encode)   │    │  (forward)  │    │  (logits)   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

---

## Prerequisites

```bash
pip install transformers datasets torch scikit-learn
```

**Required packages:**

- `transformers` - Hugging Face transformers library
- `datasets` - Dataset loading and processing
- `torch` - PyTorch deep learning framework
- `scikit-learn` - For evaluation metrics
- `numpy` - Numerical operations

---

## Getting Started

### Recommended Learning Path:

```text
1. Pipelines     →  Quick results with minimal code
2. Tokenizers    →  Understand text preprocessing
3. Fine-Tuning   →  Train on your own data
```

Start with the documentation, then practice in the notebooks!

---

## 📁 Directory Structure

```text
Hugging Face/
├── README.md                    # This file
├── docs/
│   ├── pipelines.md             # Detailed pipelines guide
│   ├── tokenizers.md            # Detailed tokenizers guide
│   └── model-finetuning.md      # Detailed fine-tuning guide
├── notebooks/
│   ├── pipelines.ipynb          # Pipelines tutorial
│   ├── hf_tokenizer.ipynb       # Tokenization deep dive
│   └── model_finetunning.ipynb  # Model fine-tuning guide
└── results/                     # Training checkpoints
    ├── checkpoint-230/
    ├── checkpoint-460/
    └── checkpoint-690/
```

---

## 🔗 Resources

- [Hugging Face Documentation](https://huggingface.co/docs/transformers)
- [Hugging Face Model Hub](https://huggingface.co/models)
- [Hugging Face Datasets](https://huggingface.co/datasets)
- [Hugging Face Course](https://huggingface.co/course)
