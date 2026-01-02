# Hugging Face Pipelines 🚀

Pipelines are Hugging Face's most convenient and beginner-friendly API for performing NLP tasks. They abstract away all the complexity of tokenization, model loading, and post-processing into a single, easy-to-use interface.

---

## 📚 Table of Contents

1. [What are Pipelines?](#what-are-pipelines)
2. [How Pipelines Work](#how-pipelines-work)
3. [Available Pipeline Tasks](#available-pipeline-tasks)
4. [Practical Examples](#practical-examples)
   - [Sentiment Analysis](#1-sentiment-analysis)
   - [Language Translation](#2-language-translation)
   - [Zero-Shot Classification](#3-zero-shot-classification)
   - [Text Generation](#4-text-generation)
   - [Named Entity Recognition](#5-named-entity-recognition-ner)
5. [Specifying Custom Models](#specifying-custom-models)
6. [Pipeline Parameters](#pipeline-parameters)
7. [Best Practices](#best-practices)

---

## What are Pipelines?

Pipelines are **high-level abstractions** that handle the entire end-to-end process of NLP inference. Instead of manually:

1. Loading a tokenizer
2. Preprocessing text
3. Loading a model
4. Running inference
5. Post-processing outputs

You can accomplish all of this with **a single line of code**:

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love this product!")
# Output: [{'label': 'POSITIVE', 'score': 0.9998}]
```

---

## How Pipelines Work

Under the hood, pipelines perform three main steps:

```text
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PREPROCESSING │ →  │    MODEL        │ →  │ POSTPROCESSING  │
│   (Tokenizer)   │    │   (Inference)   │    │   (Decoding)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                      │                      │
        ▼                      ▼                      ▼
   Raw text to            Forward pass          Logits to
   token IDs              through model         human-readable
                                                labels/text
```

### Step-by-Step Breakdown:

1. **Preprocessing (Tokenization)**

   - Converts raw text into token IDs
   - Adds special tokens (`[CLS]`, `[SEP]`)
   - Creates attention masks
   - Handles padding/truncation

2. **Model Inference**

   - Passes tokenized input through the neural network
   - Produces raw logits (unnormalized scores)

3. **Postprocessing**
   - Applies softmax to get probabilities
   - Maps predictions to human-readable labels
   - Formats output as dictionaries

---

## Available Pipeline Tasks

Hugging Face provides pipelines for many NLP tasks out of the box:

| Task                     | Pipeline Name              | Description                            |
| ------------------------ | -------------------------- | -------------------------------------- |
| Sentiment Analysis       | `sentiment-analysis`       | Classify text as positive/negative     |
| Text Classification      | `text-classification`      | General text classification            |
| Named Entity Recognition | `ner`                      | Extract entities (names, places, etc.) |
| Question Answering       | `question-answering`       | Answer questions from context          |
| Text Generation          | `text-generation`          | Generate text continuations            |
| Summarization            | `summarization`            | Summarize long documents               |
| Translation              | `translation`              | Translate between languages            |
| Zero-Shot Classification | `zero-shot-classification` | Classify without training              |
| Fill Mask                | `fill-mask`                | Predict masked words                   |
| Feature Extraction       | `feature-extraction`       | Get embeddings                         |

---

## Practical Examples

### 1. Sentiment Analysis

Classify text as positive or negative sentiment.

```python
from transformers import pipeline

# Create sentiment analysis pipeline
cls = pipeline("sentiment-analysis")

# Analyze negative sentiment
result = cls("Pushpa 2 movie is full of violence and gave me a headache")
# Output: [{'label': 'NEGATIVE', 'score': 0.9987}]

# Analyze positive sentiment
result = cls("12th fail is such an inspiring movie")
# Output: [{'label': 'POSITIVE', 'score': 0.9998}]
```

#### Use Cases:

- Customer review analysis
- Social media monitoring
- Brand sentiment tracking
- Product feedback analysis

---

### 2. Language Translation

Translate text from one language to another.

```python
from transformers import pipeline

# English to Hindi translation
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")

translation = translator("How are you?")
print(translation)
# Output: [{'translation_text': 'तुम कैसे हो?'}]
```

#### Available Translation Models:

| Model                        | Languages         |
| ---------------------------- | ----------------- |
| `Helsinki-NLP/opus-mt-en-hi` | English → Hindi   |
| `Helsinki-NLP/opus-mt-en-fr` | English → French  |
| `Helsinki-NLP/opus-mt-en-de` | English → German  |
| `Helsinki-NLP/opus-mt-en-es` | English → Spanish |
| `Helsinki-NLP/opus-mt-en-zh` | English → Chinese |

#### Use Cases:

- Content localization
- Real-time chat translation
- Document translation
- Multilingual customer support

---

### 3. Zero-Shot Classification

Classify text into custom categories **without any training**. The model uses natural language understanding to match text to your provided labels.

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")

result = classifier(
    "I bought the product but it is faulty, I would like to return it and get my money back",
    candidate_labels=["refund", "new order", "existing order"],
)

print(result)
# Output:
# {
#   'sequence': 'I bought the product but it is faulty...',
#   'labels': ['refund', 'existing order', 'new order'],
#   'scores': [0.94, 0.04, 0.02]
# }
```

#### Why Zero-Shot is Powerful:

- **No training data needed** - just provide label names
- **Flexible categories** - change labels anytime
- **Natural language labels** - use descriptive phrases

#### Use Cases:

- Customer support ticket routing
- Content categorization
- Intent detection
- Topic classification

---

### 4. Text Generation

Generate text continuations from a prompt.

```python
from transformers import pipeline

generator = pipeline("text-generation")

result = generator("To become happy in life, we need to focus on healthy diet and ")
print(result)
# Output: [{'generated_text': 'To become happy in life, we need to focus on
#          healthy diet and exercise. Regular physical activity...'}]
```

#### Generation Parameters:

```python
generator(
    "Once upon a time",
    max_length=100,        # Maximum tokens to generate
    num_return_sequences=3, # Number of different outputs
    temperature=0.7,       # Creativity (0=deterministic, 1=random)
    do_sample=True         # Enable sampling
)
```

#### Use Cases:

- Content creation assistance
- Story writing
- Code completion
- Email drafting

---

### 5. Named Entity Recognition (NER)

Extract named entities like people, organizations, and locations from text.

```python
from transformers import pipeline

ner = pipeline("ner")

result = ner(
    "I am Mitudru Dutta, and I am currently running Hugging Face models",
    grouped_entities=True  # Group multi-token entities
)

print(result)
# Output:
# [
#   {'entity_group': 'PER', 'word': 'Mitudru Dutta', 'score': 0.99},
#   {'entity_group': 'ORG', 'word': 'Hugging Face', 'score': 0.98}
# ]
```

#### Entity Types:

| Entity | Description   | Example                |
| ------ | ------------- | ---------------------- |
| `PER`  | Person        | "John Smith"           |
| `ORG`  | Organization  | "Google", "NASA"       |
| `LOC`  | Location      | "New York", "France"   |
| `MISC` | Miscellaneous | "Olympics", "COVID-19" |

#### Use Cases:

- Information extraction
- Knowledge graph construction
- Document indexing
- Compliance checking (PII detection)

---

## Specifying Custom Models

While pipelines use excellent defaults, you can specify any model from the Hugging Face Hub:

```python
from transformers import pipeline

# Use a specific model for better accuracy
pipe = pipeline(
    "sentiment-analysis",
    model="FacebookAI/roberta-large-mnli"
)

result = pipe("This restaurant is awesome")
# More accurate results with specialized model
```

### Finding Models:

1. Visit [Hugging Face Model Hub](https://huggingface.co/models)
2. Filter by task (e.g., "text-classification")
3. Sort by downloads/likes
4. Use the model ID in your pipeline

### Popular Models by Task:

| Task          | Recommended Model                                  |
| ------------- | -------------------------------------------------- |
| Sentiment     | `nlptown/bert-base-multilingual-uncased-sentiment` |
| NER           | `dslim/bert-base-NER`                              |
| Translation   | `Helsinki-NLP/opus-mt-*`                           |
| Summarization | `facebook/bart-large-cnn`                          |
| Q&A           | `deepset/roberta-base-squad2`                      |

---

## Pipeline Parameters

### Common Parameters:

```python
pipeline(
    task="sentiment-analysis",  # Task type
    model="model-name",         # Specific model (optional)
    tokenizer="tokenizer-name", # Specific tokenizer (optional)
    device=0,                   # GPU device ID (-1 for CPU)
    batch_size=8,               # Batch size for processing
)
```

### Inference Parameters:

```python
classifier(
    texts,                      # Input text(s)
    truncation=True,            # Truncate long sequences
    padding=True,               # Pad short sequences
    max_length=512,             # Maximum sequence length
)
```

---

## Best Practices

### 1. Batch Processing

Process multiple texts at once for better efficiency:

```python
texts = [
    "I love this product!",
    "This is terrible.",
    "Pretty good overall."
]
results = classifier(texts)  # Process all at once
```

### 2. GPU Acceleration

Use GPU for faster inference:

```python
# Use GPU if available
classifier = pipeline("sentiment-analysis", device=0)
```

### 3. Model Caching

Models are cached after first download. Set cache directory:

```python
import os
os.environ['TRANSFORMERS_CACHE'] = '/path/to/cache'
```

### 4. Error Handling

```python
try:
    result = classifier(text)
except Exception as e:
    print(f"Pipeline error: {e}")
```

---

## 📓 Practice Notebook

For hands-on practice, see: [pipelines.ipynb](../notebooks/pipelines.ipynb)

---

## 🔗 Resources

- [Hugging Face Pipelines Documentation](https://huggingface.co/docs/transformers/main_classes/pipelines)
- [Available Pipeline Tasks](https://huggingface.co/docs/transformers/task_summary)
- [Model Hub](https://huggingface.co/models)
