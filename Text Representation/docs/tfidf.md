# TF-IDF (Term Frequency-Inverse Document Frequency) - Complete Guide

## Overview

TF-IDF is one of the most powerful and widely-used techniques for text representation in NLP and information retrieval. It goes beyond simple word counting by evaluating how important a word is to a document within a collection of documents (corpus).

The key insight is that words appearing frequently in a single document but rarely across all documents are likely more important for characterizing that document.

---

## Table of Contents

1. [The Intuition](#the-intuition)
2. [The Mathematics](#the-mathematics)
3. [Step-by-Step Calculation](#step-by-step-calculation)
4. [Implementation](#implementation)
5. [TF-IDF vs Bag of Words](#tf-idf-vs-bag-of-words)
6. [Use Cases](#use-cases)
7. [Best Practices](#best-practices)
8. [Advanced Topics](#advanced-topics)

---

## The Intuition

### The Problem with Raw Counts

Consider a document about machine learning. Words like "the", "is", "and" might appear 100+ times, while "neural" and "network" appear only 10 times each. Raw counts would suggest "the" is more important, but we know that's not true.

### The TF-IDF Solution

TF-IDF balances two factors:

| Factor                               | What it Measures                          | Effect                             |
| :----------------------------------- | :---------------------------------------- | :--------------------------------- |
| **TF** (Term Frequency)              | How often a word appears in THIS document | Higher TF → More important locally |
| **IDF** (Inverse Document Frequency) | How rare a word is ACROSS ALL documents   | Higher IDF → More unique           |

### The Key Insight

$$\text{TF-IDF} = \text{Local Importance} \times \text{Global Rarity}$$

| Word Type                 | TF       | IDF       | TF-IDF   |
| :------------------------ | :------- | :-------- | :------- |
| Common word ("the", "is") | High     | Low       | **Low**  |
| Rare but meaningful       | Moderate | High      | **High** |
| Very rare (typos, names)  | Low      | Very High | Moderate |

**TF-IDF highlights words that are:**

- Frequent in the current document (high TF)
- Rare across all documents (high IDF)

---

## The Mathematics

### The TF-IDF Formula

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

Where:

- $t$ = term (word)
- $d$ = document
- $D$ = corpus (collection of all documents)

### Term Frequency (TF)

How often a term appears in a document, normalized by document length:

$$\text{TF}(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d}$$

**Alternative TF formulas:**

- **Raw count**: $f_{t,d}$ (just the count)
- **Boolean**: 1 if present, 0 otherwise
- **Log normalization**: $1 + \log(f_{t,d})$

### Inverse Document Frequency (IDF)

How rare a term is across the entire corpus:

$$\text{IDF}(t) = \log\left(\frac{N}{df_t}\right)$$

Where:

- $N$ = Total number of documents
- $df_t$ = Number of documents containing term $t$

**Scikit-learn's IDF formula** (with smoothing):

$$\text{IDF}(t) = \log\left(\frac{N + 1}{df_t + 1}\right) + 1$$

The +1 smoothing prevents division by zero and ensures IDF is always positive.

---

## Step-by-Step Calculation

### Example Corpus

Let's calculate TF-IDF manually for this corpus:

| Doc | Text                                       |
| :-- | :----------------------------------------- |
| D1  | "Thor eating pizza, Loki is eating pizza"  |
| D2  | "Apple is announcing new iPhone tomorrow"  |
| D3  | "Tesla is announcing new Model-3 tomorrow" |
| D4  | "Google is announcing new Pixel tomorrow"  |

### Step 1: Calculate Document Frequency (DF)

| Term       | Documents Containing | DF  |
| :--------- | :------------------- | :-- |
| pizza      | D1                   | 1   |
| eating     | D1                   | 1   |
| thor       | D1                   | 1   |
| loki       | D1                   | 1   |
| announcing | D2, D3, D4           | 3   |
| new        | D2, D3, D4           | 3   |
| tomorrow   | D2, D3, D4           | 3   |
| is         | D1, D2, D3, D4       | 4   |

### Step 2: Calculate IDF

$$\text{IDF}(t) = \log\left(\frac{N}{df_t}\right) = \log\left(\frac{4}{df_t}\right)$$

| Term       | DF  | IDF = log(4/DF)  |
| :--------- | :-- | :--------------- |
| pizza      | 1   | log(4/1) = 1.386 |
| eating     | 1   | log(4/1) = 1.386 |
| announcing | 3   | log(4/3) = 0.288 |
| is         | 4   | log(4/4) = 0.000 |

**Interpretation:**

- "pizza" has high IDF (rare, only in D1)
- "is" has IDF of 0 (appears everywhere, not distinctive)

### Step 3: Calculate TF-IDF for Document 1

| Term   | TF (in D1)  | IDF   | TF-IDF |
| :----- | :---------- | :---- | :----- |
| eating | 2/8 = 0.25  | 1.386 | 0.347  |
| pizza  | 2/8 = 0.25  | 1.386 | 0.347  |
| thor   | 1/8 = 0.125 | 1.386 | 0.173  |
| is     | 1/8 = 0.125 | 0.000 | 0.000  |

**Key insight**: "is" has zero TF-IDF despite appearing in the document because it's too common.

---

## Implementation

### Using Scikit-learn's TfidfVectorizer

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

corpus = [
    "Thor eating pizza, Loki is eating pizza",
    "Apple is announcing new iPhone tomorrow",
    "Tesla is announcing new Model-3 tomorrow",
    "Google is announcing new Pixel tomorrow"
]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

# View as DataFrame
df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=vectorizer.get_feature_names_out(),
    index=['Thor Doc', 'Apple Doc', 'Tesla Doc', 'Google Doc']
)

print(df.round(3))
```

**Output:**

```
           announcing  apple  eating  google  iphone  ...  thor  tomorrow
Thor Doc        0.000  0.000   0.655   0.000   0.000  ...  0.373     0.000
Apple Doc       0.356  0.468   0.000   0.000   0.468  ...  0.000     0.356
Tesla Doc       0.356  0.000   0.000   0.000   0.000  ...  0.000     0.356
Google Doc      0.356  0.000   0.000   0.468   0.000  ...  0.000     0.356
```

### View IDF Values

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "Thor eating pizza, Loki is eating pizza",
    "Apple is announcing new iPhone tomorrow",
    "Tesla is announcing new Model-3 tomorrow",
    "Google is announcing new Pixel tomorrow"
]

vectorizer = TfidfVectorizer()
vectorizer.fit_transform(corpus)

# Display IDF scores
for word, idx in sorted(vectorizer.vocabulary_.items()):
    print(f"{word:15} IDF = {vectorizer.idf_[idx]:.3f}")
```

**Output:**

```
announcing      IDF = 1.223
apple           IDF = 1.916
eating          IDF = 1.916
google          IDF = 1.916
iphone          IDF = 1.916
is              IDF = 1.000
loki            IDF = 1.916
model           IDF = 1.916
new             IDF = 1.223
pixel           IDF = 1.916
pizza           IDF = 1.916
tesla           IDF = 1.916
thor            IDF = 1.916
tomorrow        IDF = 1.223
```

### Important TfidfVectorizer Parameters

```python
vectorizer = TfidfVectorizer(
    max_features=5000,        # Limit vocabulary size
    min_df=2,                 # Ignore terms in < 2 documents
    max_df=0.95,              # Ignore terms in > 95% of documents
    stop_words='english',     # Remove stop words
    ngram_range=(1, 2),       # Use unigrams and bigrams
    norm='l2',                # L2 normalization (default)
    sublinear_tf=True,        # Use 1 + log(tf) instead of raw tf
    use_idf=True,             # Enable IDF weighting
    smooth_idf=True           # Add 1 to prevent division by zero
)
```

---

## TF-IDF vs Bag of Words

| Aspect               | Bag of Words              | TF-IDF                        |
| :------------------- | :------------------------ | :---------------------------- |
| **What it measures** | Raw word counts           | Importance-weighted frequency |
| **Common words**     | High values               | Downweighted automatically    |
| **Rare words**       | Low values                | Upweighted (if meaningful)    |
| **Document length**  | Affects values            | Normalized                    |
| **Computation**      | Slightly faster           | Slightly slower               |
| **Best for**         | Short texts, simple tasks | Longer documents, search      |

### Code Comparison

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

corpus = [
    "the movie was great",
    "the movie was not great at all"
]

# Bag of Words
bow = CountVectorizer()
bow_matrix = bow.fit_transform(corpus)

# TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(corpus)

print("=== Bag of Words ===")
print(pd.DataFrame(bow_matrix.toarray(), columns=bow.get_feature_names_out()))

print("\n=== TF-IDF ===")
print(pd.DataFrame(tfidf_matrix.toarray().round(3), columns=tfidf.get_feature_names_out()))
```

---

## Use Cases

### 1. Search Engines (Information Retrieval)

TF-IDF is foundational to search engine ranking:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Document collection
documents = [
    "Python programming tutorial for beginners",
    "Advanced machine learning with Python",
    "Web development with JavaScript",
    "Data science and machine learning"
]

# Query
query = ["machine learning Python"]

# Vectorize documents and query
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents)
query_vector = vectorizer.transform(query)

# Calculate similarity
similarities = cosine_similarity(query_vector, doc_vectors)[0]

# Rank documents
ranked_indices = similarities.argsort()[::-1]

print("Search Results:")
for i, idx in enumerate(ranked_indices):
    print(f"{i+1}. {documents[idx]} (score: {similarities[idx]:.3f})")
```

### 2. Document Classification

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Sample data
texts = [
    "Stock market crashes amid economic fears",
    "New study reveals health benefits of exercise",
    "Scientists discover new exoplanet",
    "Football team wins championship game",
    "Tech company reports record profits",
    "Researchers develop new vaccine",
    "Basketball player scores 50 points",
    "Economic growth exceeds expectations"
]
labels = ['business', 'health', 'science', 'sports',
          'business', 'health', 'sports', 'business']

# Split and vectorize
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42
)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train and predict
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
predictions = model.predict(X_test_tfidf)
```

### 3. Keyword Extraction

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

document = """
Machine learning is a subset of artificial intelligence.
Machine learning algorithms learn from data and improve over time.
Deep learning is a type of machine learning using neural networks.
"""

# Use TF-IDF to find important terms
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform([document])

# Get top keywords
feature_names = vectorizer.get_feature_names_out()
tfidf_scores = tfidf_matrix.toarray()[0]

# Sort by TF-IDF score
keyword_indices = tfidf_scores.argsort()[::-1][:10]

print("Top Keywords:")
for idx in keyword_indices:
    print(f"  {feature_names[idx]}: {tfidf_scores[idx]:.3f}")
```

### 4. Document Similarity / Plagiarism Detection

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

doc1 = "Machine learning is a branch of artificial intelligence"
doc2 = "Artificial intelligence includes machine learning"
doc3 = "I love eating pizza and pasta"

documents = [doc1, doc2, doc3]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Calculate pairwise similarities
similarity_matrix = cosine_similarity(tfidf_matrix)

print("Similarity Matrix:")
print(similarity_matrix.round(3))

# Output:
# [[1.    0.573 0.   ]   doc1-doc1, doc1-doc2, doc1-doc3
#  [0.573 1.    0.   ]   doc2-doc1, doc2-doc2, doc2-doc3
#  [0.    0.    1.   ]]  doc3-doc1, doc3-doc2, doc3-doc3
```

### 5. Recommendation Systems

Content-based filtering using TF-IDF:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Article descriptions
articles = {
    "Article 1": "Python programming data science machine learning",
    "Article 2": "JavaScript web development frontend backend",
    "Article 3": "Machine learning deep learning neural networks",
    "Article 4": "Web scraping Python data collection",
    "Article 5": "React JavaScript frontend development"
}

# User's read article
user_read = "Article 1"

# Find similar articles
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(articles.values())

# Get similarity to user's article
article_idx = list(articles.keys()).index(user_read)
similarities = cosine_similarity(tfidf_matrix[article_idx], tfidf_matrix)[0]

# Recommend top 3 similar articles
recommendations = sorted(
    [(list(articles.keys())[i], sim) for i, sim in enumerate(similarities) if i != article_idx],
    key=lambda x: x[1],
    reverse=True
)[:3]

print(f"Since you read '{user_read}', you might like:")
for article, score in recommendations:
    print(f"  {article} (similarity: {score:.3f})")
```

---

## Best Practices

### 1. Always Use Stop Word Removal

```python
vectorizer = TfidfVectorizer(stop_words='english')
```

### 2. Consider Sublinear TF

For long documents, use logarithmic TF to dampen frequent terms:

```python
vectorizer = TfidfVectorizer(sublinear_tf=True)
# Uses 1 + log(tf) instead of raw tf
```

### 3. Limit Vocabulary

```python
vectorizer = TfidfVectorizer(
    max_features=10000,  # Top 10K features
    min_df=5,            # Appear in at least 5 documents
    max_df=0.9           # Appear in at most 90% of documents
)
```

### 4. Combine with N-grams

```python
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),  # Unigrams and bigrams
    max_features=10000
)
```

### 5. L2 Normalization (Default)

Ensures all document vectors have unit length for fair comparison:

```python
vectorizer = TfidfVectorizer(norm='l2')  # Default
```

---

## Advanced Topics

### Custom IDF Weights

You can provide custom IDF values:

```python
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

# First, get counts
count_vec = CountVectorizer()
count_matrix = count_vec.fit_transform(corpus)

# Then apply TF-IDF with custom settings
tfidf_transformer = TfidfTransformer(
    norm='l2',
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=True
)
tfidf_matrix = tfidf_transformer.fit_transform(count_matrix)
```

### BM25: TF-IDF Alternative

BM25 is a more sophisticated ranking function used in modern search engines:

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}$$

```python
# Using rank_bm25 library
from rank_bm25 import BM25Okapi

corpus = [
    "Hello there good man!",
    "It is quite windy in London",
    "How is the weather today?"
]

tokenized_corpus = [doc.lower().split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

query = "windy London"
tokenized_query = query.lower().split()

scores = bm25.get_scores(tokenized_query)
print(scores)
```

---

## Common Pitfalls

### 1. Not Using transform() on Test Data

```python
# WRONG - fits new vocabulary on test data
X_test_tfidf = vectorizer.fit_transform(X_test)

# CORRECT - uses vocabulary from training data
X_test_tfidf = vectorizer.transform(X_test)
```

### 2. Including Test Data in fit()

```python
# WRONG - data leakage!
vectorizer.fit(all_data)

# CORRECT - fit only on training data
vectorizer.fit(X_train)
```

### 3. Ignoring Sparse Matrix Format

```python
# TF-IDF returns sparse matrix for efficiency
tfidf_matrix = vectorizer.fit_transform(corpus)
type(tfidf_matrix)  # scipy.sparse.csr_matrix

# Only convert to dense when necessary
dense_matrix = tfidf_matrix.toarray()  # Uses more memory!
```

---

## Key Takeaways

1. **TF-IDF** measures word importance by balancing frequency and rarity
2. **High TF-IDF** = frequent in document, rare across corpus
3. **Stop words** naturally get low TF-IDF scores
4. **Use `TfidfVectorizer`** from scikit-learn for easy implementation
5. **Combine with N-grams** for better phrase detection
6. **Perfect for** search, classification, keyword extraction
7. **Use `transform()`** (not `fit_transform()`) on test data

---

## Practice Exercise

1. Load the spam dataset (`spam.csv`)
2. Create TF-IDF representations
3. Train a classifier (Naive Bayes or SVM)
4. Analyze which words have highest TF-IDF in spam vs ham
5. Compare performance with Bag of Words

See [tf_idf.ipynb](tf_idf.ipynb) for a complete implementation.

---

## Further Reading

- [TF-IDF Wikipedia](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Scikit-learn TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [BM25: The Next Generation of TF-IDF](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables)
- [Stanford IR Book - Scoring and Ranking](https://nlp.stanford.edu/IR-book/html/htmledition/scoring-term-weighting-and-the-vector-space-model-1.html)

---

## Next Steps

➡️ [Word Embeddings](word_embeddings.md) — Dense semantic representations
➡️ [Text Classification](text_classification.md) — Apply TF-IDF in a real project
