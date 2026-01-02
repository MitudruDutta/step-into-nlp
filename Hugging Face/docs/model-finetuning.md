# Model Fine-Tuning with Hugging Face 🎯

Fine-tuning is the process of taking a pretrained model and training it further on your specific dataset. This guide covers everything you need to know about fine-tuning transformer models using Hugging Face's Trainer API.

---

## 📚 Table of Contents

1. [What is Fine-Tuning?](#what-is-fine-tuning)
2. [Why Fine-Tune?](#why-fine-tune)
3. [The Fine-Tuning Process](#the-fine-tuning-process)
4. [Step-by-Step Guide](#step-by-step-guide)
   - [Step 1: Load Dataset](#step-1-load-dataset)
   - [Step 2: Tokenization](#step-2-tokenization)
   - [Step 3: Data Collator](#step-3-data-collator)
   - [Step 4: Load Model](#step-4-load-model)
   - [Step 5: Define Metrics](#step-5-define-metrics)
   - [Step 6: Training Arguments](#step-6-training-arguments)
   - [Step 7: Create Trainer](#step-7-create-trainer)
   - [Step 8: Train](#step-8-train)
   - [Step 9: Evaluate](#step-9-evaluate)
   - [Step 10: Inference](#step-10-inference)
5. [Understanding the GLUE MRPC Dataset](#understanding-the-glue-mrpc-dataset)
6. [Training Arguments Explained](#training-arguments-explained)
7. [Dynamic vs Global Padding](#dynamic-vs-global-padding)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

---

## What is Fine-Tuning?

Fine-tuning adapts a **pretrained model** to a **specific task** by continuing training on task-specific data.

```text
┌─────────────────────┐          ┌─────────────────────┐
│   Pretrained Model  │    →     │   Fine-Tuned Model  │
│   (General NLP)     │          │   (Your Task)       │
│                     │          │                     │
│  - Language understanding      │  - Sentiment analysis
│  - Grammar                     │  - Paraphrase detection
│  - Context                     │  - Custom classification
└─────────────────────┘          └─────────────────────┘
           │                              │
           │    + Your Dataset           │
           │    + Task-specific training │
           └──────────────────────────────┘
```

### Transfer Learning Concept

Instead of training from scratch (which requires massive data and compute), we:

1. Start with a model pretrained on huge datasets (Wikipedia, books, etc.)
2. Fine-tune on our smaller, task-specific dataset
3. Achieve excellent results with less data and compute

---

## Why Fine-Tune?

### Benefits:

| Benefit               | Description                                              |
| --------------------- | -------------------------------------------------------- |
| **Less Data**         | Works with hundreds/thousands of examples (not millions) |
| **Less Compute**      | Hours of training (not weeks)                            |
| **Better Results**    | Leverages pretrained knowledge                           |
| **Domain Adaptation** | Specialize for your domain (legal, medical, etc.)        |

### When to Fine-Tune:

- ✅ Your task is similar to pretrained tasks but needs customization
- ✅ You have labeled data for your specific task
- ✅ Pretrained models don't perform well enough on your data
- ✅ You need a model optimized for your domain vocabulary

### When NOT to Fine-Tune:

- ❌ Zero-shot classification works well enough
- ❌ You have no labeled training data
- ❌ The pretrained pipeline already meets your needs

---

## The Fine-Tuning Process

```text
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Dataset   │ →  │  Tokenize   │ →  │   Trainer   │ →  │   Model     │
│   (HF Hub)  │    │   Data      │    │   (Train)   │    │  (Saved)    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
      │                  │                  │                  │
      ▼                  ▼                  ▼                  ▼
   load_dataset    tokenizer.map     trainer.train    trainer.save
```

---

## Step-by-Step Guide

### Required Imports

```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
```

---

### Step 1: Load Dataset

```python
# Load GLUE MRPC dataset
dataset = load_dataset("glue", "mrpc")
print(dataset)
# DatasetDict({
#     train: Dataset({features: ['sentence1', 'sentence2', 'label', 'idx'], num_rows: 3668}),
#     validation: Dataset({...}),
#     test: Dataset({...})
# })

# Examine a sample
print(dataset['train'][0])
# {
#   'sentence1': 'Amrozi accused his brother...',
#   'sentence2': 'Referring to him as only...',
#   'label': 1,  # 1 = paraphrase, 0 = not paraphrase
#   'idx': 0
# }
```

#### Dataset Structure:

- **sentence1**: First sentence
- **sentence2**: Second sentence
- **label**: 1 (equivalent) or 0 (not equivalent)
- **idx**: Index

---

### Step 2: Tokenization

```python
# Load tokenizer
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Define tokenization function
def tokenize_function(example):
    return tokenizer(
        example["sentence1"],
        example["sentence2"],
        truncation=True
    )

# Apply to entire dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)
print(tokenized_dataset)
# Now includes: input_ids, attention_mask, token_type_ids
```

#### What `batched=True` Does:

- Processes multiple examples at once
- Much faster than one-by-one
- Tokenizer handles batch efficiently

---

### Step 3: Data Collator

The **Data Collator** handles dynamic padding during batch creation.

```python
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

#### Why Dynamic Padding?

```
Batch 1:                          Batch 2:
┌──────────────────────┐          ┌────────────────┐
│ Sentence A (15 tok)  │          │ Sent X (8 tok) │
│ Sentence B (12 tok)  │          │ Sent Y (6 tok) │
│ Sentence C (10 tok)  │          │ Sent Z (9 tok) │
└──────────────────────┘          └────────────────┘
Padded to 15 tokens               Padded to 9 tokens
```

| Padding Type | Description            | Efficiency        |
| ------------ | ---------------------- | ----------------- |
| **Dynamic**  | Pad to batch max       | ✅ More efficient |
| **Global**   | Pad all to dataset max | ❌ Wasteful       |

---

### Step 4: Load Model

```python
# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pretrained model with classification head
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=2  # Binary classification
).to(device)
```

#### Model Architecture:

```
┌─────────────────────────────────────┐
│         BERT Base Model             │
│    (Pretrained on large corpus)     │
├─────────────────────────────────────┤
│    Classification Head (New)        │
│    Linear(768 → 2)                  │
│    (Randomly initialized)           │
└─────────────────────────────────────┘
```

---

### Step 5: Define Metrics

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    return {
        "accuracy": acc,
        "f1": f1
    }
```

#### Why Both Accuracy and F1?

| Metric       | Best For                                          |
| ------------ | ------------------------------------------------- |
| **Accuracy** | Balanced datasets                                 |
| **F1 Score** | Imbalanced datasets (combines precision & recall) |

---

### Step 6: Training Arguments

```python
training_args = TrainingArguments(
    output_dir="./results",              # Where to save
    eval_strategy="epoch",               # Evaluate after each epoch
    save_strategy="epoch",               # Save after each epoch
    per_device_train_batch_size=16,      # Batch size for training
    per_device_eval_batch_size=16,       # Batch size for evaluation
    num_train_epochs=3,                  # Number of epochs
    weight_decay=0.01,                   # L2 regularization
    load_best_model_at_end=True,         # Load best checkpoint when done
    metric_for_best_model="accuracy",    # Metric to determine best model
)
```

See [Training Arguments Explained](#training-arguments-explained) for all options.

---

### Step 7: Create Trainer

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)
```

---

### Step 8: Train

```python
# Start training
trainer.train()

# Output:
# Epoch 1/3: loss=0.45, accuracy=0.82, f1=0.87
# Epoch 2/3: loss=0.32, accuracy=0.85, f1=0.89
# Epoch 3/3: loss=0.25, accuracy=0.87, f1=0.91
```

#### What Happens During Training:

1. Load batch of data
2. Forward pass through model
3. Compute loss
4. Backward pass (gradients)
5. Update weights
6. Repeat for all batches
7. Evaluate on validation set
8. Save checkpoint if best

---

### Step 9: Evaluate

```python
# Evaluate on test set
results = trainer.evaluate(tokenized_dataset["test"])
print(results)
# {
#   'eval_loss': 0.28,
#   'eval_accuracy': 0.86,
#   'eval_f1': 0.90,
#   'eval_runtime': 12.5,
#   'eval_samples_per_second': 138.4
# }
```

---

### Step 10: Inference

```python
def predict(sentences):
    """Make predictions on new sentence pairs."""
    inputs = tokenizer(
        sentences["sentence1"],
        sentences["sentence2"],
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    predictions = torch.argmax(logits, dim=1)
    return predictions.cpu().numpy()

# Test on new examples
sample_sentences = {
    "sentence1": [
        "PCCW's chief operating officer, Mike Butcher, will report directly to Mr So",
        "The Nasdaq composite index increased 10.73, or 0.7 percent"
    ],
    "sentence2": [
        "Current Chief Operating Officer Mike Butcher will report to So",
        "The Nasdaq Composite index was lately up around 18 points"
    ]
}

predictions = predict(sample_sentences)
print("Predictions:", predictions)
# [1, 0] → First pair is paraphrase, second is not
```

---

## Understanding the GLUE MRPC Dataset

**MRPC** = Microsoft Research Paraphrase Corpus

### Task Definition:

Given two sentences, determine if they express the same meaning (paraphrase).

### Examples:

| Sentence 1            | Sentence 2               | Label              |
| --------------------- | ------------------------ | ------------------ |
| "The stock rose 5%"   | "Shares increased by 5%" | 1 (Paraphrase)     |
| "It rained yesterday" | "The weather was sunny"  | 0 (Not Paraphrase) |

### Dataset Statistics:

- Training: 3,668 pairs
- Validation: 408 pairs
- Test: 1,725 pairs
- Positive ratio: ~68% (slightly imbalanced)

---

## Training Arguments Explained

### Essential Arguments

| Argument                      | Description     | Typical Values   |
| ----------------------------- | --------------- | ---------------- |
| `output_dir`                  | Save directory  | `"./results"`    |
| `num_train_epochs`            | Training epochs | 2-5              |
| `per_device_train_batch_size` | Batch size      | 8, 16, 32        |
| `learning_rate`               | Learning rate   | 2e-5, 3e-5, 5e-5 |

### Evaluation Arguments

| Argument                | Description         | Options                        |
| ----------------------- | ------------------- | ------------------------------ |
| `eval_strategy`         | When to evaluate    | `"no"`, `"epoch"`, `"steps"`   |
| `eval_steps`            | Steps between evals | 500, 1000                      |
| `metric_for_best_model` | Best model metric   | `"accuracy"`, `"f1"`, `"loss"` |

### Saving Arguments

| Argument                 | Description              | Options                      |
| ------------------------ | ------------------------ | ---------------------------- |
| `save_strategy`          | When to save             | `"no"`, `"epoch"`, `"steps"` |
| `save_total_limit`       | Max checkpoints          | 2, 3, 5                      |
| `load_best_model_at_end` | Load best after training | `True`, `False`              |

### Regularization

| Argument       | Description       | Typical Values |
| -------------- | ----------------- | -------------- |
| `weight_decay` | L2 regularization | 0.01, 0.1      |
| `warmup_steps` | LR warmup steps   | 500, 1000      |
| `warmup_ratio` | Warmup as ratio   | 0.1            |

### Advanced Options

```python
TrainingArguments(
    # Gradient accumulation (simulate larger batches)
    gradient_accumulation_steps=4,

    # Mixed precision training (faster on modern GPUs)
    fp16=True,

    # Logging
    logging_dir="./logs",
    logging_steps=100,

    # Early stopping
    load_best_model_at_end=True,

    # Reproducibility
    seed=42,
)
```

---

## Dynamic vs Global Padding

### The Problem

Sequences have different lengths, but models need fixed-size inputs.

### Global Padding (Inefficient)

```python
# Pad ALL sequences to max length (512)
tokenizer(texts, padding='max_length', max_length=512)

# Result: Lots of wasted computation
# [token, token, PAD, PAD, PAD, ... PAD]  # 500 PADs!
```

### Dynamic Padding (Efficient)

```python
# DataCollatorWithPadding pads per batch
data_collator = DataCollatorWithPadding(tokenizer)

# Batch 1: sequences of 10, 12, 15 → pad to 15
# Batch 2: sequences of 8, 6, 9 → pad to 9
# Much less wasted computation!
```

### Comparison

| Approach    | Pros              | Cons                  |
| ----------- | ----------------- | --------------------- |
| **Global**  | Simple            | Wasteful, slow        |
| **Dynamic** | Efficient, faster | Slightly more complex |

---

## Best Practices

### 1. Start with a Pretrained Model

Always start with a model pretrained on similar data:

```python
# For English text classification
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# For multilingual
model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased")
```

### 2. Use Appropriate Batch Size

- Larger batch = faster training, but more memory
- If OOM error, reduce batch size
- Use gradient accumulation for effective larger batches

```python
TrainingArguments(
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,  # Effective batch = 32
)
```

### 3. Learning Rate Selection

- Too high → unstable training, divergence
- Too low → slow convergence
- Typical range: 1e-5 to 5e-5

### 4. Monitor Training

```python
TrainingArguments(
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=500,
)
```

### 5. Save and Load Checkpoints

```python
# Save
trainer.save_model("./my_model")

# Load
model = AutoModelForSequenceClassification.from_pretrained("./my_model")
```

---

## Troubleshooting

### Out of Memory (OOM)

```python
# Reduce batch size
per_device_train_batch_size=8

# Use gradient accumulation
gradient_accumulation_steps=4

# Enable mixed precision
fp16=True
```

### Training Loss Not Decreasing

- Lower learning rate
- Check data quality
- Increase model capacity

### Overfitting

- Add weight decay
- Reduce epochs
- Add dropout
- Use more training data

### MLflow/DagsHub Errors

```python
# If you get MLflow errors, uninstall:
# pip uninstall mlflow dagshub
# Then restart kernel
```

---

## 📓 Practice Notebook

For hands-on practice, see: [model_finetuning.ipynb](../notebooks/model_finetuning.ipynb)

---

## 🔗 Resources

- [Hugging Face Trainer Documentation](https://huggingface.co/docs/transformers/main_classes/trainer)
- [Training Arguments Reference](https://huggingface.co/docs/transformers/main_classes/trainer#trainingarguments)
- [Fine-Tuning Tutorial](https://huggingface.co/docs/transformers/training)
- [GLUE Benchmark](https://gluebenchmark.com/)
