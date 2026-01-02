# Hugging Face Tokenizers 🔤

Tokenizers are the bridge between human-readable text and the numerical representations that machine learning models understand. This guide covers everything you need to know about tokenization in Hugging Face.

---

## 📚 Table of Contents

1. [What is Tokenization?](#what-is-tokenization)
2. [Why Tokenization Matters](#why-tokenization-matters)
3. [Types of Tokenizers](#types-of-tokenizers)
4. [Using Hugging Face Tokenizers](#using-hugging-face-tokenizers)
5. [Special Tokens](#special-tokens)
6. [Padding and Truncation](#padding-and-truncation)
7. [Batch Tokenization](#batch-tokenization)
8. [End-to-End Workflow](#end-to-end-workflow)
9. [Advanced Features](#advanced-features)
10. [Best Practices](#best-practices)

---

## What is Tokenization?

Tokenization is the process of breaking down text into smaller units called **tokens**. These tokens are then converted to numerical IDs that models can process.

```text
"Hello, world!" → ["Hello", ",", "world", "!"] → [7592, 1010, 2088, 999]
```

### The Tokenization Process:

```text
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Raw Text   │ →  │   Tokens    │ →  │  Token IDs  │ →  │   Tensors   │
│             │    │  (strings)  │    │  (integers) │    │  (model)    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
      │                  │                  │                  │
      ▼                  ▼                  ▼                  ▼
 "Hello world"    ["Hello", "world"]   [7592, 2088]    tensor([...])
```

---

## Why Tokenization Matters

### 1. **Vocabulary Efficiency**

Modern tokenizers use subword tokenization to balance:

- **Small vocabulary size** (fewer parameters)
- **Handle unknown words** (break into known subwords)

### 2. **Consistency**

Same tokenizer must be used for training and inference:

```python
# ✅ Correct: Use matching tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# ❌ Wrong: Mismatched tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
```

### 3. **Special Token Handling**

Models expect specific special tokens that tokenizers automatically add.

---

## Types of Tokenizers

### 1. Word-Level Tokenization

Splits on whitespace and punctuation.

```text
"Don't stop" → ["Don't", "stop"] or ["Don", "'", "t", "stop"]
```

**Problem**: Large vocabulary, can't handle unknown words.

### 2. Character-Level Tokenization

Each character is a token.

```text
"Hello" → ["H", "e", "l", "l", "o"]
```

**Problem**: Very long sequences, loses semantic meaning.

### 3. Subword Tokenization (Most Common)

Breaks words into meaningful subunits.

```text
"unhappiness" → ["un", "##happiness"] or ["un", "happi", "ness"]
```

#### Popular Subword Algorithms:

| Algorithm         | Used By          | Example                  |
| ----------------- | ---------------- | ------------------------ |
| **WordPiece**     | BERT, DistilBERT | `##` prefix for subwords |
| **BPE**           | GPT, RoBERTa     | Byte-pair encoding       |
| **SentencePiece** | T5, ALBERT       | Language-agnostic        |
| **Unigram**       | XLNet            | Probabilistic model      |

---

## Using Hugging Face Tokenizers

### Loading a Tokenizer

```python
from transformers import AutoTokenizer

# Load tokenizer for a specific model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

### Basic Tokenization

```python
text = "Happiness lies within you"

# Tokenize text
output = tokenizer(text)
print(output)
# {
#   'input_ids': [101, 8114, 7459, 2306, 2017, 102],
#   'attention_mask': [1, 1, 1, 1, 1, 1]
# }
```

### Understanding the Output

| Key              | Description                      |
| ---------------- | -------------------------------- |
| `input_ids`      | Token IDs for the model          |
| `attention_mask` | 1 for real tokens, 0 for padding |
| `token_type_ids` | Segment IDs (for sentence pairs) |

### Converting Between Formats

```python
# IDs → Tokens
tokens = tokenizer.convert_ids_to_tokens(output['input_ids'])
print(tokens)
# ['[CLS]', 'happiness', 'lies', 'within', 'you', '[SEP]']

# IDs → Text
text = tokenizer.decode(output['input_ids'])
print(text)
# '[CLS] happiness lies within you [SEP]'

# IDs → Text (skip special tokens)
text = tokenizer.decode(output['input_ids'], skip_special_tokens=True)
print(text)
# 'happiness lies within you'
```

---

## Special Tokens

Special tokens are reserved tokens with specific meanings for the model.

### Common Special Tokens

| Token    | Name           | Purpose                                    | BERT ID |
| -------- | -------------- | ------------------------------------------ | ------- |
| `[CLS]`  | Classification | Start of sequence, used for classification | 101     |
| `[SEP]`  | Separator      | End of sequence / between segments         | 102     |
| `[PAD]`  | Padding        | Fill shorter sequences in a batch          | 0       |
| `[UNK]`  | Unknown        | Represents unknown tokens                  | 100     |
| `[MASK]` | Mask           | For masked language modeling               | 103     |

### Accessing Special Token IDs

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

print(tokenizer.cls_token_id)  # 101
print(tokenizer.sep_token_id)  # 102
print(tokenizer.pad_token_id)  # 0
print(tokenizer.unk_token_id)  # 100
print(tokenizer.mask_token_id) # 103
```

### How Special Tokens Are Added

```python
text = "Hello world"
tokens = tokenizer(text)

# Input:  "Hello world"
# Output: [CLS] hello world [SEP]
# IDs:    [101, 7592, 2088, 102]
```

---

## Padding and Truncation

When processing batches, all sequences must have the same length.

### Padding

Add `[PAD]` tokens to shorter sequences:

```python
texts = [
    "Short text",
    "This is a much longer piece of text"
]

# Without padding - different lengths
output = tokenizer(texts)
# Lengths: [4, 10]

# With padding - same length
output = tokenizer(texts, padding=True, return_tensors='pt')
# All sequences padded to length 10
```

### Padding Strategies

| Strategy               | Description                   |
| ---------------------- | ----------------------------- |
| `padding=True`         | Pad to longest in batch       |
| `padding='max_length'` | Pad to `max_length` parameter |
| `padding='longest'`    | Same as `True`                |

### Truncation

Cut sequences longer than maximum length:

```python
# Truncate to max_length
output = tokenizer(
    text,
    truncation=True,
    max_length=128
)
```

### Combined Padding and Truncation

```python
# Typical usage for training
output = tokenizer(
    texts,
    padding='max_length',
    max_length=512,
    truncation=True,
    return_tensors='pt'
)
```

### Attention Mask

The attention mask tells the model which tokens are real (1) vs padding (0):

```python
texts = ["Hello", "Hello world"]
output = tokenizer(texts, padding=True, return_tensors='pt')

print(output['input_ids'])
# tensor([[ 101, 7592,  102,    0],
#         [ 101, 7592, 2088,  102]])

print(output['attention_mask'])
# tensor([[1, 1, 1, 0],    # Last token is padding
#         [1, 1, 1, 1]])   # All real tokens
```

---

## Batch Tokenization

### Processing Multiple Texts

```python
texts = [
    "Happiness lies within you",
    "I love nature",
    "Machine learning is fascinating"
]

# Tokenize all at once
outputs = tokenizer(texts)

# With padding for model input
outputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors='pt'
)
```

### Tokenizing Sentence Pairs

For tasks like paraphrase detection or Q&A:

```python
sentence1 = "How old are you?"
sentence2 = "What is your age?"

# Tokenize as pair
output = tokenizer(sentence1, sentence2, return_tensors='pt')

# Result includes token_type_ids to distinguish sentences:
# Sentence 1: token_type_ids = 0
# Sentence 2: token_type_ids = 1
```

---

## End-to-End Workflow

Here's how tokenization fits into the complete inference pipeline:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# 1. Load tokenizer and model
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# 2. Prepare input texts
sequences = [
    "That phone case broke after 2 days of use",
    "That herbal tea has helped me so much"
]

# 3. Tokenize
tokens = tokenizer(
    sequences,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

# 4. Run model
output = model(**tokens)
print(output.logits)
# tensor([[ 4.12, -3.45],   # Negative sentiment
#         [-3.78,  4.01]])  # Positive sentiment

# 5. Post-process
probs = F.softmax(output.logits, dim=-1)
predictions = torch.argmax(probs, dim=1).tolist()
print(predictions)  # [0, 1] → [NEGATIVE, POSITIVE]
```

### Pipeline vs Manual Approach

The pipeline does all this automatically:

```python
from transformers import pipeline

# One line does everything above!
pipe = pipeline("sentiment-analysis")
result = pipe("My dog is cute")
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

---

## Advanced Features

### Fast Tokenizers

Hugging Face provides Rust-based "fast" tokenizers:

```python
# Load fast tokenizer explicitly
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased",
    use_fast=True  # Default in most cases
)
```

#### Benefits of Fast Tokenizers:

- 3-5x faster tokenization
- Offset mapping (character positions)
- Batch encoding optimizations

### Offset Mapping

Track character positions in original text:

```python
output = tokenizer(
    "Hello world",
    return_offsets_mapping=True
)

print(output['offset_mapping'])
# [(0, 0), (0, 5), (6, 11), (0, 0)]
# [CLS]    Hello   world   [SEP]
```

### Adding Special Tokens

```python
# Add custom tokens
tokenizer.add_special_tokens({
    'additional_special_tokens': ['[CUSTOM]']
})

# Don't forget to resize model embeddings
model.resize_token_embeddings(len(tokenizer))
```

---

## Best Practices

### 1. Match Tokenizer with Model

```python
# Always use the same checkpoint for both
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)
```

### 2. Use return_tensors for Model Input

```python
# Return PyTorch tensors
output = tokenizer(text, return_tensors='pt')

# Return TensorFlow tensors
output = tokenizer(text, return_tensors='tf')

# Return NumPy arrays
output = tokenizer(text, return_tensors='np')
```

### 3. Handle Long Sequences

```python
# Option 1: Truncate
tokenizer(text, truncation=True, max_length=512)

# Option 2: Split into chunks
# (for very long documents)
def chunk_text(text, chunk_size=500, overlap=50):
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(chunk)
    return chunks
```

### 4. Efficient Batch Processing

```python
# Process in batches for memory efficiency
from torch.utils.data import DataLoader

def tokenize_batch(batch):
    return tokenizer(
        batch,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

# Use DataLoader for large datasets
```

---

## 📓 Practice Notebook

For hands-on practice, see: [hf_tokenizer.ipynb](../notebooks/hf_tokenizer.ipynb)

---

## 🔗 Resources

- [Hugging Face Tokenizers Documentation](https://huggingface.co/docs/transformers/main_classes/tokenizer)
- [Tokenizer Summary](https://huggingface.co/docs/transformers/tokenizer_summary)
- [Fast Tokenizers](https://huggingface.co/docs/tokenizers)
