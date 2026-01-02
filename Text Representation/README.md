# Text Representation in NLP 🔢

Text Representation, also known as **Feature Engineering**, is the process of converting raw text into numerical formats that machine learning algorithms can understand. This is arguably the most critical step in NLP—effective representation is often more important than the algorithm itself.

This module covers everything from basic encoding methods to advanced word embeddings, with hands-on implementations using real-world datasets.

---

## � Directory Structure

```
Text Representation/
├── README.md                          # This file
├── docs/                              # 📖 Documentation guides
│   ├── bag_of_words.md
│   ├── ngrams.md
│   ├── tfidf.md
│   ├── word_embeddings.md
│   └── text_classification.md
├── notebooks/                         # 💻 Jupyter notebooks
│   ├── bag_of_words.ipynb
│   ├── 10_bag_of_n_grams.ipynb
│   ├── tf_idf.ipynb
│   ├── spacy_word_vectors.ipynb
│   └── text_classification.ipynb
└── data/                              # 📊 Datasets
    ├── Ecommerce_data.csv
    ├── Emotion_classify_Data.csv
    ├── Fake_Real_Data.csv
    ├── movies_sentiment_data.csv
    ├── news_dataset.json
    └── spam.csv
```

---

## 📚 Table of Contents

| Topic                   | Detailed Guide                               | Quick Reference                               | Notebook                                       | Description                               |
| :---------------------- | :------------------------------------------- | :-------------------------------------------- | :--------------------------------------------- | :---------------------------------------- |
| **Bag of Words**        | [📖 Full Guide](docs/bag_of_words.md)        | [🔗 Section](#1-bag-of-words-bow-)            | [💻 Code](notebooks/bag_of_words.ipynb)        | Count-based text representation           |
| **N-Grams**             | [📖 Full Guide](docs/ngrams.md)              | [🔗 Section](#2-n-grams-)                     | [💻 Code](notebooks/10_bag_of_n_grams.ipynb)   | Capturing word sequences                  |
| **TF-IDF**              | [📖 Full Guide](docs/tfidf.md)               | [🔗 Section](#3-tf-idf-)                      | [💻 Code](notebooks/tf_idf.ipynb)              | Term frequency-inverse document frequency |
| **Word Embeddings**     | [📖 Full Guide](docs/word_embeddings.md)     | [🔗 Section](#4-word-embeddings-)             | [💻 Code](notebooks/spacy_word_vectors.ipynb)  | Dense vector representations              |
| **Text Classification** | [📖 Full Guide](docs/text_classification.md) | [🔗 Section](#5-text-classification-project-) | [💻 Code](notebooks/text_classification.ipynb) | End-to-end classification pipeline        |

---

## 🎯 Why Text Representation Matters

Computers cannot process raw text directly—they need numbers. The way we convert text to numbers directly impacts:

1. **Model Performance**: Better representations lead to better predictions
2. **Semantic Understanding**: Capturing meaning, not just words
3. **Computational Efficiency**: Sparse vs dense representations
4. **Generalization**: Handling unseen words and contexts

---

## 📉 Primitive Encoding Methods

Before diving into advanced techniques, let's understand basic approaches and their limitations:

### Label Encoding

Assigns a unique integer to each word in the vocabulary.

```python
vocabulary = {"cat": 0, "dog": 1, "bird": 2, "fish": 3}

# "cat dog" → [0, 1]
# "bird fish" → [2, 3]
```

| Pros                | Cons                                    |
| :------------------ | :-------------------------------------- |
| Simple to implement | No semantic meaning (is "dog" > "cat"?) |
| Memory efficient    | Order implies false relationships       |
|                     | Doesn't scale with vocabulary           |

### One-Hot Encoding (OHE)

Creates a binary vector for each word where only one position is 1.

```python
# Vocabulary: ["cat", "dog", "bird", "fish"]

"cat"  → [1, 0, 0, 0]
"dog"  → [0, 1, 0, 0]
"bird" → [0, 0, 1, 0]
"fish" → [0, 0, 0, 1]
```

| Pros                     | Cons                                        |
| :----------------------- | :------------------------------------------ |
| No ordinal relationships | Extremely sparse matrices                   |
| Easy to implement        | High dimensionality (vocabulary size)       |
|                          | Cannot handle OOV (out-of-vocabulary) words |
|                          | No semantic similarity captured             |

---

## 1. Bag of Words (BoW) 🎒

**Bag of Words** represents text as a vector of word counts, completely disregarding grammar and word order.

### How It Works

```
Document 1: "I love NLP and I love Python"
Document 2: "NLP is fun"

Vocabulary: ["I", "love", "NLP", "and", "Python", "is", "fun"]

Document 1 → [2, 2, 1, 1, 1, 0, 0]  (I appears 2 times, love 2 times, etc.)
Document 2 → [0, 0, 1, 0, 0, 1, 1]
```

### Implementation with Scikit-learn

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "I love NLP and I love Python",
    "NLP is fun",
    "Python is amazing for NLP"
]

vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(corpus)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("BoW Matrix:\n", bow_matrix.toarray())
```

**Output:**

```
Vocabulary: ['amazing' 'and' 'for' 'fun' 'is' 'love' 'nlp' 'python']
BoW Matrix:
 [[0 1 0 0 0 2 1 1]
  [0 0 0 1 1 0 1 0]
  [1 0 1 0 1 0 1 1]]
```

### Pros and Cons

| Pros                      | Cons                        |
| :------------------------ | :-------------------------- |
| Simple and intuitive      | Loses word order completely |
| Works well for many tasks | High dimensionality         |
| Easy to interpret         | Sparse matrices             |
|                           | Treats all words equally    |

### Use Cases

- Spam detection
- Document classification
- Sentiment analysis (basic)
- Search engines (simple matching)

---

## 2. N-Grams 📊

**N-Grams** extend Bag of Words by capturing sequences of N consecutive words, preserving some context.

### Types of N-Grams

| N   | Name    | Example ("I love NLP") |
| :-- | :------ | :--------------------- |
| 1   | Unigram | "I", "love", "NLP"     |
| 2   | Bigram  | "I love", "love NLP"   |
| 3   | Trigram | "I love NLP"           |

### Why N-Grams Matter

Consider sentiment analysis:

- "not good" as bigram → captures negation
- "not" + "good" as unigrams → loses the negative meaning

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = ["this movie is not good", "this movie is good"]

# Unigrams only
unigram_vec = CountVectorizer(ngram_range=(1, 1))
print("Unigrams:", unigram_vec.fit_transform(corpus).toarray())

# Bigrams
bigram_vec = CountVectorizer(ngram_range=(2, 2))
print("Bigrams:", bigram_vec.fit_transform(corpus).toarray())

# Both Unigrams and Bigrams
combined_vec = CountVectorizer(ngram_range=(1, 2))
print("Combined:", combined_vec.fit_transform(corpus).toarray())
```

### Trade-offs

| Larger N                    | Smaller N             |
| :-------------------------- | :-------------------- |
| More context preserved      | Less context          |
| Sparser matrices            | Denser representation |
| Higher dimensionality       | Lower dimensionality  |
| May overfit                 | May underfit          |
| Better for specific phrases | Better generalization |

### Best Practices

```python
# Common configuration: unigrams + bigrams
vectorizer = CountVectorizer(
    ngram_range=(1, 2),  # Include both unigrams and bigrams
    max_features=10000,   # Limit vocabulary size
    min_df=2,             # Ignore terms appearing in < 2 documents
    max_df=0.95           # Ignore terms appearing in > 95% of documents
)
```

---

## 3. TF-IDF ⚖️

**TF-IDF (Term Frequency-Inverse Document Frequency)** is a statistical measure that evaluates how important a word is to a document within a corpus. It balances word frequency against document frequency.

### The Formula

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

Where:

**Term Frequency (TF)**: How often a term appears in a document

$$\text{TF}(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d}$$

**Inverse Document Frequency (IDF)**: How rare a term is across all documents

$$\text{IDF}(t) = \log\left(\frac{\text{Total number of documents}}{\text{Number of documents containing term } t}\right)$$

### Intuition

| Word Type                 | TF       | IDF       | TF-IDF   |
| :------------------------ | :------- | :-------- | :------- |
| Common word ("the", "is") | High     | Low       | Low      |
| Rare but meaningful       | Moderate | High      | **High** |
| Very rare                 | Low      | Very High | Moderate |

**TF-IDF highlights words that are:**

- Frequent in the current document (high TF)
- Rare across all documents (high IDF)

### Implementation

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "Thor eating pizza, Loki is eating pizza",
    "Apple is announcing new iPhone tomorrow",
    "Tesla is announcing new Model-3 tomorrow",
    "Google is announcing new Pixel tomorrow"
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

# View vocabulary with IDF scores
for word, idx in vectorizer.vocabulary_.items():
    print(f"{word}: IDF = {vectorizer.idf_[idx]:.3f}")
```

**Output (example):**

```
pizza: IDF = 1.916      # Rare - only in doc 1
eating: IDF = 1.916     # Rare - only in doc 1
announcing: IDF = 1.223 # Common - in docs 2, 3, 4
is: IDF = 1.000         # Very common - in all docs
```

### TF-IDF vs BoW

| Aspect               | Bag of Words              | TF-IDF                                   |
| :------------------- | :------------------------ | :--------------------------------------- |
| **What it measures** | Raw word counts           | Importance-weighted frequency            |
| **Common words**     | High values               | Downweighted                             |
| **Rare words**       | Low values                | Upweighted (if meaningful)               |
| **Best for**         | Short texts, simple tasks | Longer documents, search, classification |

### Use Cases

- **Search engines**: Ranking documents by relevance
- **Document classification**: News categorization
- **Keyword extraction**: Finding important terms
- **Plagiarism detection**: Document similarity
- **Recommendation systems**: Content-based filtering

---

## 4. Word Embeddings 💎

**Word Embeddings** are dense, low-dimensional vector representations that capture semantic meaning. Unlike BoW/TF-IDF (sparse vectors with thousands of dimensions), embeddings typically have 100-300 dimensions.

### Key Insight

Words with similar meanings have similar vectors:

```
vector("king") - vector("man") + vector("woman") ≈ vector("queen")
```

### Types of Word Embeddings

| Model        | Training Approach                  | Characteristics                        |
| :----------- | :--------------------------------- | :------------------------------------- |
| **Word2Vec** | Prediction-based (CBOW, Skip-gram) | Fast training, good for analogies      |
| **GloVe**    | Count-based + prediction           | Captures global statistics             |
| **FastText** | Character n-grams                  | Handles OOV words, morphology          |
| **ELMo**     | Bidirectional LSTM                 | Context-dependent embeddings           |
| **BERT**     | Transformer                        | State-of-the-art contextual embeddings |

### Word2Vec Architectures

**CBOW (Continuous Bag of Words):**

- Predicts target word from context words
- "The cat [?] on the mat" → predicts "sat"
- Faster training, better for frequent words

**Skip-gram:**

- Predicts context words from target word
- "sat" → predicts "The", "cat", "on", "the", "mat"
- Better for rare words, smaller datasets

### Using Word Vectors with spaCy

```python
import spacy

# Load large model (includes 300-dim vectors)
nlp = spacy.load("en_core_web_lg")

# Check vector properties
doc = nlp("dog cat banana")
for token in doc:
    print(f"{token.text}: has_vector={token.has_vector}, shape={token.vector.shape}")
```

**Output:**

```
dog: has_vector=True, shape=(300,)
cat: has_vector=True, shape=(300,)
banana: has_vector=True, shape=(300,)
```

### Semantic Similarity

```python
import spacy

nlp = spacy.load("en_core_web_lg")

# Compare word similarities
base_word = nlp("bread")
words = nlp("sandwich burger pizza car airplane")

for token in words:
    similarity = token.similarity(base_word)
    print(f"{token.text} <-> bread: {similarity:.3f}")
```

**Output:**

```
sandwich <-> bread: 0.562
burger <-> bread: 0.474
pizza <-> bread: 0.432
car <-> bread: 0.121
airplane <-> bread: 0.087
```

### The Famous Analogy

```python
import spacy
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_lg")

king = nlp.vocab["king"].vector
man = nlp.vocab["man"].vector
woman = nlp.vocab["woman"].vector
queen = nlp.vocab["queen"].vector

# king - man + woman ≈ queen
result = king - man + woman

similarity = cosine_similarity([result], [queen])
print(f"Similarity to 'queen': {similarity[0][0]:.3f}")
# Output: ~0.72
```

### Embeddings vs Sparse Representations

| Aspect                  | BoW/TF-IDF             | Word Embeddings     |
| :---------------------- | :--------------------- | :------------------ |
| **Dimensionality**      | High (vocabulary size) | Low (100-300)       |
| **Sparsity**            | Very sparse            | Dense               |
| **Semantic similarity** | None                   | Captured            |
| **OOV handling**        | Cannot handle          | FastText can handle |
| **Training required**   | No (just counting)     | Yes (large corpus)  |
| **Memory**              | High                   | Lower               |

---

## 5. Text Classification Project 🏆

This module includes a complete text classification project that brings together all concepts.

### Project: E-commerce Product Classification

**Task**: Classify product descriptions into 4 categories:

- Electronics
- Household
- Books
- Clothing & Accessories

### Pipeline Overview

```
Raw Text → Preprocessing → Vectorization → Model Training → Evaluation
```

### Implementation Steps

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# 1. Load data
df = pd.read_csv("Ecommerce_data.csv")

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['Text'], df['label'], test_size=0.2, random_state=42
)

# 3. Vectorize with TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 5. Evaluate
predictions = model.predict(X_test_tfidf)
print(classification_report(y_test, predictions))
```

---

## 📁 Datasets Included

| File                        | Description          | Use Case                   |
| :-------------------------- | :------------------- | :------------------------- |
| `spam.csv`                  | SMS spam collection  | Binary classification      |
| `Ecommerce_data.csv`        | Product descriptions | Multi-class classification |
| `Emotion_classify_Data.csv` | Text emotions        | Sentiment/emotion analysis |
| `Fake_Real_Data.csv`        | News articles        | Fake news detection        |
| `movies_sentiment_data.csv` | Movie reviews        | Sentiment analysis         |
| `news_dataset.json`         | News articles        | Topic classification       |

---

## 🛠️ Tools & Libraries

| Library                | Purpose                                     |
| :--------------------- | :------------------------------------------ |
| **scikit-learn**       | CountVectorizer, TfidfVectorizer, ML models |
| **spaCy**              | Word vectors, NLP pipeline                  |
| **pandas**             | Data manipulation                           |
| **numpy**              | Numerical operations                        |
| **matplotlib/seaborn** | Visualization                               |

---

## 🎓 Learning Path

```
1. Bag of Words (bag_of_words.ipynb)
   └── Understand basic count-based representation

2. N-Grams (10_bag_of_n_grams.ipynb)
   └── Learn to capture word sequences

3. TF-IDF (tf_idf.ipynb)
   └── Master importance-weighted representations

4. Word Embeddings (spacy_word_vectors.ipynb)
   └── Explore dense semantic representations

5. Text Classification (text_classification.ipynb)
   └── Apply everything in a real project
```

---

## 📊 Comparison Summary

| Method       | Dimensionality | Semantic | Context | OOV | Speed            |
| :----------- | :------------- | :------- | :------ | :-- | :--------------- |
| One-Hot      | Very High      | ❌       | ❌      | ❌  | ⚡ Fast          |
| Bag of Words | High           | ❌       | ❌      | ❌  | ⚡ Fast          |
| N-Grams      | Very High      | Partial  | Partial | ❌  | ⚡ Fast          |
| TF-IDF       | High           | ❌       | ❌      | ❌  | ⚡ Fast          |
| Word2Vec     | Low (300)      | ✅       | ❌      | ❌  | 🐢 Slow to train |
| FastText     | Low (300)      | ✅       | ❌      | ✅  | 🐢 Slow to train |
| BERT         | Low (768)      | ✅       | ✅      | ✅  | 🐌 Very slow     |

---

## 🎯 When to Use What

| Task                         | Recommended Approach            |
| :--------------------------- | :------------------------------ |
| Simple classification        | TF-IDF + Naive Bayes/SVM        |
| Sentiment analysis           | TF-IDF or Word Embeddings       |
| Semantic similarity          | Word Embeddings                 |
| Search/Information retrieval | TF-IDF                          |
| State-of-the-art NLP         | Transformer embeddings (BERT)   |
| Small dataset                | TF-IDF (no training needed)     |
| Large dataset                | Word Embeddings or Transformers |

---

## 📚 Further Reading

- [Word2Vec Paper](https://arxiv.org/abs/1301.3781)
- [GloVe Paper](https://nlp.stanford.edu/pubs/glove.pdf)
- [TF-IDF Explained](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [spaCy Word Vectors](https://spacy.io/usage/vectors-similarity)
- [Scikit-learn Text Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)

---

## ▶️ Next Steps

After mastering text representation, proceed to:

- **Practical NLP with Hugging Face** — Transformer-based models
- **Deep Learning for NLP** — Neural network architectures
