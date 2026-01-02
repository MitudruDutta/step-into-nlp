# Bag of Words (BoW) - Complete Guide

## Overview

Bag of Words (BoW) is one of the most fundamental and widely-used text representation techniques in Natural Language Processing. It transforms text into fixed-length numerical vectors by counting word occurrences, completely ignoring grammar and word order. Think of it as putting all words from a document into a "bag" and counting how many times each word appears.

Despite its simplicity, BoW remains surprisingly effective for many NLP tasks, especially when combined with appropriate preprocessing and machine learning algorithms.

---

## Table of Contents

1. [Core Concept](#core-concept)
2. [How It Works](#how-it-works)
3. [Step-by-Step Example](#step-by-step-example)
4. [Implementation](#implementation)
5. [Practical Application: Spam Detection](#practical-application-spam-detection)
6. [Advantages and Limitations](#advantages-and-limitations)
7. [Best Practices](#best-practices)
8. [When to Use BoW](#when-to-use-bow)

---

## Core Concept

The Bag of Words model represents text as a "bag" (multiset) of words, disregarding grammar and word order but keeping track of multiplicity (count). The key insight is that for many classification tasks, the presence or frequency of certain words is more important than their position.

**Analogy**: Imagine dumping all words from a document into a bag, then counting what you have. "The cat sat on the mat" becomes: {the: 2, cat: 1, sat: 1, on: 1, mat: 1}.

### Key Characteristics

| Characteristic         | Description                             |
| :--------------------- | :-------------------------------------- |
| **Representation**     | Sparse vector of word counts            |
| **Order Preservation** | None - word order is lost               |
| **Vocabulary**         | Fixed - determined from training corpus |
| **Dimensionality**     | Equal to vocabulary size                |

---

## How It Works

The BoW process follows three main steps:

### Step 1: Build Vocabulary

Collect all unique words across all documents to create a vocabulary (dictionary).

### Step 2: Create Document Vectors

For each document, count the occurrence of each vocabulary word.

### Step 3: Form Document-Term Matrix

Each row represents a document, each column represents a word from the vocabulary.

```
           word1  word2  word3  ...  wordN
Document 1:  [3,     0,     2,   ...    1]
Document 2:  [0,     2,     1,   ...    0]
Document 3:  [1,     1,     0,   ...    2]
```

---

## Step-by-Step Example

Let's work through a concrete example with two sentences:

**Corpus:**

1. "People watch Campusx"
2. "Campusx watch Campusx"

### Building the Vocabulary

Unique words (sorted alphabetically): `[campusx, people, watch]`

### Creating Vectors

| Sentence   | campusx | people | watch |
| :--------- | :-----: | :----: | :---: |
| Sentence 1 |    1    |   1    |   1   |
| Sentence 2 |    2    |   0    |   1   |

**Resulting Matrix:**

```
[[1, 1, 1],   # People watch Campusx
 [2, 0, 1]]   # Campusx watch Campusx
```

---

## Implementation

### Using Scikit-learn's CountVectorizer

```python
from sklearn.feature_extraction.text import CountVectorizer

# Sample corpus
corpus = [
    "Thor is eating pizza",
    "Loki is eating pizza with Thor"
]

# Create vectorizer
vectorizer = CountVectorizer()

# Fit and transform
bow_matrix = vectorizer.fit_transform(corpus)

# View vocabulary
print("Vocabulary:", vectorizer.vocabulary_)
# Output: {'thor': 5, 'is': 1, 'eating': 0, 'pizza': 3, 'loki': 2, 'with': 6}

# View as array
print("BoW Matrix:\n", bow_matrix.toarray())
# Output:
# [[1, 1, 0, 1, 0, 1, 0],   # Thor is eating pizza
#  [1, 1, 1, 1, 0, 1, 1]]   # Loki is eating pizza with Thor

# Get feature names
print("Features:", vectorizer.get_feature_names_out())
# Output: ['eating', 'is', 'loki', 'pizza', 'thor', 'with']
```

### Creating a DataFrame for Better Visualization

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "Thor is eating pizza",
    "Loki is eating pizza with Thor",
    "Hulk smash"
]

vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(corpus)

# Create readable DataFrame
df = pd.DataFrame(
    bow_matrix.toarray(),
    columns=vectorizer.get_feature_names_out(),
    index=['Doc 1', 'Doc 2', 'Doc 3']
)

print(df)
```

**Output:**

```
       eating  hulk  is  loki  pizza  smash  thor  with
Doc 1       1     0   1     0      1      0     1     0
Doc 2       1     0   1     1      1      0     1     1
Doc 3       0     1   0     0      0      1     0     0
```

### Important CountVectorizer Parameters

```python
vectorizer = CountVectorizer(
    max_features=1000,        # Limit vocabulary size
    min_df=2,                 # Ignore terms appearing in < 2 docs
    max_df=0.95,              # Ignore terms appearing in > 95% of docs
    stop_words='english',     # Remove English stop words
    lowercase=True,           # Convert to lowercase
    ngram_range=(1, 1),       # Use unigrams only (1,2) for uni+bigrams
    binary=False              # True for presence (0/1), False for counts
)
```

---

## Practical Application: Spam Detection

Here's a complete example using BoW for spam classification:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample spam dataset
data = {
    'text': [
        "Free money now!!!",
        "Hi, how are you?",
        "Win a free iPhone today!",
        "Meeting at 3pm tomorrow",
        "Congratulations! You won $1000",
        "Can we reschedule our call?",
        "URGENT: Claim your prize now",
        "Thanks for your help yesterday"
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}

df = pd.DataFrame(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.25, random_state=42
)

# Create BoW representation
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Train classifier
classifier = MultinomialNB()
classifier.fit(X_train_bow, y_train)

# Predict and evaluate
predictions = classifier.predict(X_test_bow)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
print(classification_report(y_test, predictions))
```

### Key Points for Spam Detection:

1. **fit_transform()** on training data - learns vocabulary and transforms
2. **transform()** on test data - uses existing vocabulary (no fitting!)
3. **Out-of-vocabulary words** in test data are ignored

---

## Advantages and Limitations

### Advantages ✅

| Advantage            | Description                       |
| :------------------- | :-------------------------------- |
| **Simplicity**       | Easy to understand and implement  |
| **Efficiency**       | Fast computation and training     |
| **Interpretability** | Feature importance is transparent |
| **Baseline**         | Strong baseline for many tasks    |
| **No Training**      | No pre-training required          |

### Limitations ❌

| Limitation              | Description                       |
| :---------------------- | :-------------------------------- |
| **No Semantics**        | Cannot capture word meaning       |
| **No Order**            | "dog bites man" = "man bites dog" |
| **High Dimensionality** | Vocabulary size can be huge       |
| **Sparse Vectors**      | Most values are zero              |
| **OOV Problem**         | Cannot handle new words           |

### The Word Order Problem

Consider these sentences:

- "The movie was not good, it was great"
- "The movie was great, it was not good"

In BoW, both sentences have identical representations because word order is ignored. This is a significant limitation for sentiment analysis and other order-sensitive tasks.

---

## Best Practices

### 1. Preprocessing is Essential

Always preprocess your text before applying BoW:

```python
import re
from sklearn.feature_extraction.text import CountVectorizer

def preprocess(text):
    text = text.lower()                    # Lowercase
    text = re.sub(r'[^\w\s]', '', text)    # Remove punctuation
    text = re.sub(r'\d+', '', text)        # Remove numbers
    return text

# Apply preprocessing
corpus = [preprocess(doc) for doc in raw_corpus]
vectorizer = CountVectorizer(stop_words='english')
bow_matrix = vectorizer.fit_transform(corpus)
```

### 2. Limit Vocabulary Size

Large vocabularies cause:

- Memory issues
- Overfitting
- Slow computation

```python
vectorizer = CountVectorizer(
    max_features=5000,    # Keep only top 5000 features
    min_df=5,             # Ignore rare words (< 5 documents)
    max_df=0.8            # Ignore too common words (> 80% documents)
)
```

### 3. Consider Binary BoW

For some tasks, word presence is more useful than frequency:

```python
vectorizer = CountVectorizer(binary=True)
# All non-zero counts become 1
```

### 4. Combine with N-grams

Capture some word order with bigrams/trigrams:

```python
vectorizer = CountVectorizer(ngram_range=(1, 2))
# Includes both unigrams and bigrams
```

---

## When to Use BoW

### Good Use Cases ✓

- **Simple text classification** (spam, sentiment, topic)
- **Baseline model** before trying complex approaches
- **Limited resources** (memory, computation)
- **Small datasets** where embeddings may overfit
- **Quick prototyping** and experiments
- **Keyword-based tasks** where exact words matter

### Consider Alternatives When

- **Semantic similarity** is important (use embeddings)
- **Word order** matters (use n-grams or RNNs)
- **State-of-the-art** performance needed (use BERT/transformers)
- **Out-of-vocabulary** words are common (use FastText)

---

## Comparison with Other Methods

| Method       | Captures Semantics | Preserves Order | Handles OOV | Dimensionality |
| :----------- | :----------------: | :-------------: | :---------: | :------------: |
| **BoW**      |         ❌         |       ❌        |     ❌      |      High      |
| **TF-IDF**   |         ❌         |       ❌        |     ❌      |      High      |
| **Word2Vec** |         ✅         |       ❌        |     ❌      |      Low       |
| **FastText** |         ✅         |       ❌        |     ✅      |      Low       |
| **BERT**     |         ✅         |       ✅        |     ✅      |      Low       |

---

## Practice Exercise

Try implementing BoW on the spam dataset included in this module:

1. Load `spam.csv`
2. Preprocess the text (lowercase, remove punctuation)
3. Create BoW representation
4. Train a Naive Bayes classifier
5. Evaluate on test data

See [bag_of_words.ipynb](bag_of_words.ipynb) for a complete implementation.

---

## Key Takeaways

1. **BoW** converts text to numerical vectors by counting word occurrences
2. **Word order is lost** - this is both a limitation and a feature
3. **Preprocessing** is crucial for good results
4. **CountVectorizer** in scikit-learn makes implementation easy
5. **Best for** simple classification tasks and as a baseline
6. **Consider n-grams** to capture some word order information
7. **Vocabulary management** is important for efficiency

---

## Further Reading

- [Scikit-learn CountVectorizer Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
- [Text Feature Extraction Guide](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [Naive Bayes Classification](https://scikit-learn.org/stable/modules/naive_bayes.html)

---

## Next Steps

After understanding Bag of Words, proceed to:

➡️ [N-Grams](ngrams.md) — Capture word sequences for better context
➡️ [TF-IDF](tfidf.md) — Weight words by importance
