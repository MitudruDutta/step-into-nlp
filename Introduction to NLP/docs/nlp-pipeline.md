# 🔄 The NLP Application Pipeline

## Overview

Building an NLP application is not a one-step process—it requires a **thoughtful and iterative approach** that varies based on the specific task (classification, summarization, translation, etc.).

This guide provides an in-depth look at each stage of the NLP pipeline, from data acquisition to deployment and monitoring.

---

## 🏗️ Pipeline Architecture

### High-Level View

```text
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│ Data Acquisition│───▶│  Preprocessing   │───▶│ Feature Extraction  │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│ Monitor & Update│◀───│    Deployment    │◀───│   Model Building    │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
                              ▲                         │
                              │    ┌─────────────────┐  │
                              └────│   Evaluation    │◀─┘
                                   └─────────────────┘
```

### Pipeline Stages Summary

| Stage              | Purpose                      | Key Outcome                          |
| :----------------- | :--------------------------- | :----------------------------------- |
| Data Acquisition   | Collect raw text data        | Training/inference dataset           |
| Preprocessing      | Clean and normalize text     | Standardized text ready for analysis |
| Feature Extraction | Convert text to numbers      | Numerical representations            |
| Model Building     | Train predictive models      | Trained ML/DL model                  |
| Evaluation         | Measure performance          | Performance metrics                  |
| Deployment         | Serve model in production    | Live API/service                     |
| Monitoring         | Track real-world performance | Alerts and insights                  |

---

## 📥 Stage 1: Data Acquisition

### What is it?

Gathering raw text data from various sources to create your training dataset.

### Data Sources

| Source Type         | Examples                              | Pros                   | Cons                        |
| :------------------ | :------------------------------------ | :--------------------- | :-------------------------- |
| **Public Datasets** | Wikipedia, Common Crawl, Kaggle       | Free, large scale      | May not fit your domain     |
| **APIs**            | Twitter API, Reddit API, News APIs    | Real-time, structured  | Rate limits, costs          |
| **Web Scraping**    | Custom scrapers, Beautiful Soup       | Customizable           | Legal concerns, maintenance |
| **Internal Data**   | Customer emails, chat logs, documents | Domain-specific        | Privacy, labeling effort    |
| **Synthetic Data**  | LLM-generated data                    | Scalable, customizable | Quality concerns            |

### Best Practices

```python
# Example: Loading data from multiple sources
import pandas as pd
from datasets import load_dataset

# Public dataset
imdb = load_dataset("imdb")

# CSV file
custom_data = pd.read_csv("customer_reviews.csv")

# API data
import requests
response = requests.get("https://api.example.com/data")
api_data = response.json()
```

### Key Considerations

1. **Data Quality** — Garbage in, garbage out
2. **Data Quantity** — More data generally leads to better models
3. **Data Diversity** — Cover edge cases and variations
4. **Legal & Ethical** — Respect copyright, privacy, and terms of service
5. **Labeling** — For supervised learning, you need labeled data

---

## 🧹 Stage 2: Preprocessing

### What is it?

Cleaning and transforming raw text into a standardized format suitable for analysis.

### Preprocessing Pipeline

```
Raw Text
    │
    ▼
┌─────────────────────┐
│   Text Cleaning     │  Remove HTML, special chars, normalize unicode
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│   Tokenization      │  Split into words/subwords/sentences
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│   Normalization     │  Lowercase, expand contractions
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│   Stopword Removal  │  Remove common words (the, is, at)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Stemming/Lemmatization │  Reduce words to base form
└─────────────────────┘
    │
    ▼
Cleaned Text
```

### Preprocessing Techniques

| Technique                | Description               | Example                            |
| :----------------------- | :------------------------ | :--------------------------------- |
| **Tokenization**         | Split text into tokens    | "Hello world" → ["Hello", "world"] |
| **Lowercasing**          | Convert to lowercase      | "HELLO" → "hello"                  |
| **Stopword Removal**     | Remove common words       | "the cat is here" → "cat here"     |
| **Stemming**             | Reduce to word stem       | "running", "runs" → "run"          |
| **Lemmatization**        | Reduce to dictionary form | "better" → "good"                  |
| **Removing Punctuation** | Strip punctuation marks   | "Hello!" → "Hello"                 |
| **Removing Numbers**     | Strip numeric characters  | "Room 101" → "Room"                |

### Code Example

```python
import spacy
import re

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    """Complete preprocessing pipeline"""

    # 1. Basic cleaning
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation

    # 2. Tokenization and linguistic processing with spaCy
    doc = nlp(text)

    # 3. Lemmatization + Stopword removal
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and token.text.strip()
    ]

    return ' '.join(tokens)

# Example usage
raw_text = "The cats are running quickly towards the beautiful garden!"
clean_text = preprocess_text(raw_text)
print(clean_text)  # "cat run quickly beautiful garden"
```

### When to Use What?

| Use Case               | Recommended Preprocessing             |
| :--------------------- | :------------------------------------ |
| **Sentiment Analysis** | Light preprocessing, keep emojis      |
| **Topic Modeling**     | Heavy preprocessing, remove stopwords |
| **NER**                | Minimal preprocessing, keep casing    |
| **Transformers**       | Use model's built-in tokenizer        |

---

## 🔢 Stage 3: Feature Extraction

### What is it?

Converting text into numerical representations that machine learning models can process.

### Feature Extraction Methods

```
                        Feature Extraction Methods
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
    Traditional              Embeddings              Transformers
          │                       │                       │
    ┌─────┴─────┐           ┌─────┴─────┐           ┌─────┴─────┐
    │  Bag of   │           │  Word2Vec │           │   BERT    │
    │  Words    │           │  GloVe    │           │   GPT     │
    │  TF-IDF   │           │  FastText │           │   T5      │
    └───────────┘           └───────────┘           └───────────┘
```

### Method Comparison

| Method                | How it Works              | Pros                  | Cons                        |
| :-------------------- | :------------------------ | :-------------------- | :-------------------------- |
| **Bag of Words**      | Count word frequencies    | Simple, interpretable | No semantics, sparse        |
| **TF-IDF**            | Weight by importance      | Better than BoW       | Still no semantics          |
| **Word2Vec**          | Neural network embeddings | Captures semantics    | Static, one vector per word |
| **GloVe**             | Matrix factorization      | Good word analogies   | Static embeddings           |
| **FastText**          | Subword embeddings        | Handles OOV words     | Still static                |
| **BERT/Transformers** | Contextual embeddings     | State-of-the-art      | Computationally expensive   |

### Code Examples

#### Bag of Words & TF-IDF

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

documents = [
    "Natural language processing is amazing",
    "NLP helps computers understand human language",
    "Machine learning powers modern NLP"
]

# Bag of Words
bow_vectorizer = CountVectorizer()
bow_features = bow_vectorizer.fit_transform(documents)
print("Vocabulary:", bow_vectorizer.get_feature_names_out())

# TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_features = tfidf_vectorizer.fit_transform(documents)
```

#### Word Embeddings

```python
from gensim.models import Word2Vec

# Training Word2Vec
sentences = [
    ["natural", "language", "processing"],
    ["machine", "learning", "nlp"],
    ["deep", "learning", "neural", "networks"]
]

model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)
vector = model.wv["language"]  # Get vector for a word
similar = model.wv.most_similar("learning")  # Find similar words
```

#### Transformer Embeddings

```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

text = "Natural language processing is fascinating"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state  # Contextual embeddings
```

---

## 🤖 Stage 4: Model Building

### What is it?

Training machine learning or deep learning models on the extracted features.

### Model Selection Guide

| Task                         | Traditional ML                  | Deep Learning      | Transformers     |
| :--------------------------- | :------------------------------ | :----------------- | :--------------- |
| **Text Classification**      | Naive Bayes, SVM, Random Forest | CNN, LSTM          | BERT, RoBERTa    |
| **Named Entity Recognition** | CRF                             | BiLSTM-CRF         | BERT-NER         |
| **Sentiment Analysis**       | Logistic Regression, SVM        | LSTM, CNN          | BERT, DistilBERT |
| **Machine Translation**      | Statistical MT                  | Seq2Seq, Attention | T5, mBART        |
| **Question Answering**       | —                               | BiDAF              | BERT-QA, T5      |
| **Text Generation**          | —                               | LSTM, GRU          | GPT, LLaMA       |

### Training Pipeline

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Split data
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# 2. Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 3. Predict
predictions = model.predict(X_test)

# 4. Evaluate
print(classification_report(y_test, predictions))
```

### Modern Approach: Fine-tuning Transformers

```python
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

---

## 📊 Stage 5: Evaluation

### What is it?

Measuring model performance using appropriate metrics.

### Common Metrics

| Metric         | Use Case                | Formula               |
| :------------- | :---------------------- | :-------------------- |
| **Accuracy**   | Balanced classification | (TP + TN) / Total     |
| **Precision**  | When FP is costly       | TP / (TP + FP)        |
| **Recall**     | When FN is costly       | TP / (TP + FN)        |
| **F1-Score**   | Imbalanced data         | 2 × (P × R) / (P + R) |
| **BLEU**       | Translation             | n-gram overlap        |
| **ROUGE**      | Summarization           | Recall-oriented       |
| **Perplexity** | Language models         | exp(cross-entropy)    |

### Evaluation Code

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# Classification metrics
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, average='weighted'))
print("Recall:", recall_score(y_true, y_pred, average='weighted'))
print("F1:", f1_score(y_true, y_pred, average='weighted'))

# Detailed report
print(classification_report(y_true, y_pred))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
```

---

## 🚀 Stage 6: Deployment

### What is it?

Making the model available for real-world use.

### Deployment Options

| Option                       | Best For            | Complexity |
| :--------------------------- | :------------------ | :--------- |
| **REST API (Flask/FastAPI)** | Small-medium scale  | Low        |
| **Docker + Kubernetes**      | Scalable production | Medium     |
| **Serverless (AWS Lambda)**  | Sporadic traffic    | Low-Medium |
| **Cloud ML Services**        | Enterprise scale    | Medium     |
| **Edge Deployment**          | Mobile/IoT          | High       |

### Simple FastAPI Deployment

```python
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    # Preprocess
    processed = preprocess(input.text)
    # Predict
    prediction = model.predict([processed])
    return {"prediction": prediction[0]}

# Run: uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## 📈 Stage 7: Monitor & Update

### What is it?

Tracking model performance in production and iterating.

### Monitoring Checklist

- [ ] **Performance Metrics** — Track accuracy, latency, throughput
- [ ] **Data Drift** — Monitor input data distribution changes
- [ ] **Model Drift** — Watch for degrading predictions
- [ ] **Error Analysis** — Log and review failures
- [ ] **A/B Testing** — Compare model versions
- [ ] **Feedback Loop** — Collect user feedback for retraining

### Continuous Improvement Cycle

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│   Monitor  ───▶  Analyze  ───▶  Retrain  ───▶  Deploy       │
│      ▲                                            │          │
│      └────────────────────────────────────────────┘          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 💡 Key Insights

### The Pipeline is Iterative

> Poor model performance might indicate a need for:
>
> - Better preprocessing
> - More training data
> - Different feature extraction
> - Hyperparameter tuning

### Start Simple, Scale Gradually

1. **Baseline First** — Start with simple models (Logistic Regression, Naive Bayes)
2. **Measure** — Establish baseline metrics
3. **Improve** — Try more complex approaches
4. **Validate** — Ensure improvements are real
5. **Deploy** — Only deploy when confident

### Common Pitfalls to Avoid

| Pitfall                  | Solution                             |
| :----------------------- | :----------------------------------- |
| Skipping preprocessing   | Always clean and normalize data      |
| Ignoring class imbalance | Use appropriate sampling/metrics     |
| Data leakage             | Proper train/test splitting          |
| Overfitting              | Regularization, cross-validation     |
| No baseline              | Always compare against simple models |

---

## 🔗 Navigation

← [What is NLP?](what-is-nlp.md) | [Back to Introduction](README.md) | [Next: NLP Tools →](nlp-tools.md)

---

_Understanding the pipeline is crucial for building robust NLP applications. Now let's explore the tools that make it all possible!_
