# N-Grams - Complete Guide

## Overview

N-grams extend the Bag of Words concept by considering sequences of N consecutive words instead of individual words. This simple modification allows us to capture some word order and context, significantly improving performance on many NLP tasks while maintaining computational efficiency.

The "N" in N-gram refers to the number of words in each sequence. When N=1, we have unigrams (single words); N=2 gives us bigrams (word pairs); N=3 produces trigrams (word triplets), and so on.

---

## Table of Contents

1. [What are N-Grams](#what-are-n-grams)
2. [Types of N-Grams](#types-of-n-grams)
3. [Why N-Grams Matter](#why-n-grams-matter)
4. [Implementation](#implementation)
5. [N-Gram Trade-offs](#n-gram-trade-offs)
6. [Best Practices](#best-practices)
7. [Applications](#applications)

---

## What are N-Grams

An N-gram is a contiguous sequence of N items from a given text. In NLP, these items are typically words (though they can also be characters).

### The Basic Idea

**Example Sentence**: "I love machine learning"

|  N  | Type    | N-grams                                      |
| :-: | :------ | :------------------------------------------- |
|  1  | Unigram | "I", "love", "machine", "learning"           |
|  2  | Bigram  | "I love", "love machine", "machine learning" |
|  3  | Trigram | "I love machine", "love machine learning"    |
|  4  | 4-gram  | "I love machine learning"                    |

### Key Insight

N-grams capture **local word order**, allowing the model to distinguish between:

- "not good" vs "good" (unigrams lose the negation)
- "New York" vs "York New" (bigrams preserve proper nouns)
- "machine learning" vs "learning machine" (bigrams capture compound terms)

---

## Types of N-Grams

### Unigrams (N=1)

Single words - equivalent to standard Bag of Words.

```
"The quick brown fox" → ["The", "quick", "brown", "fox"]
```

**Characteristics:**

- Smallest vocabulary size
- No context captured
- Fastest computation

### Bigrams (N=2)

Consecutive word pairs - the most commonly used N-gram.

```
"The quick brown fox" → ["The quick", "quick brown", "brown fox"]
```

**Characteristics:**

- Captures basic word relationships
- Identifies compound terms
- Moderate vocabulary increase

### Trigrams (N=3)

Three consecutive words - captures more context.

```
"The quick brown fox" → ["The quick brown", "quick brown fox"]
```

**Characteristics:**

- Better context understanding
- Significant vocabulary increase
- More sparse representations

### Higher-Order N-Grams (N≥4)

Longer sequences - rarely used due to sparsity.

```
"The quick brown fox" → ["The quick brown fox"]
```

**Characteristics:**

- Very specific phrases
- Extremely sparse
- Risk of overfitting

---

## Why N-Grams Matter

### Problem with Bag of Words

Consider these two reviews:

1. "The movie was **not good**, it was bad"
2. "The movie was **good**, it was not bad"

**BoW Representation** (unigrams only):
Both sentences have identical word counts! The model cannot distinguish between positive and negative sentiment.

### N-Gram Solution

**Bigram Representation:**

| Feature    | Sentence 1 | Sentence 2 |
| :--------- | :--------: | :--------: |
| "not good" |     1      |     0      |
| "was good" |     0      |     1      |
| "not bad"  |     0      |     1      |
| "was bad"  |     1      |     0      |

Now the model can learn that "not good" indicates negative sentiment while "was good" indicates positive sentiment.

### Key Benefits

1. **Negation Handling**: "not happy" ≠ "happy"
2. **Compound Terms**: "New York", "machine learning", "ice cream"
3. **Phrases**: "on the other hand", "as well as"
4. **Context Clues**: "very good" vs "not very good"

---

## Implementation

### Using Scikit-learn's CountVectorizer

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "Thor is eating pizza",
    "Loki is eating pizza"
]

# Bigrams only
bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))
bigram_matrix = bigram_vectorizer.fit_transform(corpus)

print("Bigram Features:", bigram_vectorizer.get_feature_names_out())
# Output: ['eating pizza', 'is eating', 'loki is', 'thor is']
```

### Combined Unigrams + Bigrams

The most common approach is to include both unigrams and bigrams:

```python
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

corpus = [
    "I love NLP",
    "NLP is amazing",
    "I love learning"
]

# Unigrams and Bigrams together
vectorizer = CountVectorizer(ngram_range=(1, 2))
matrix = vectorizer.fit_transform(corpus)

# Create DataFrame for visualization
df = pd.DataFrame(
    matrix.toarray(),
    columns=vectorizer.get_feature_names_out(),
    index=['Doc 1', 'Doc 2', 'Doc 3']
)

print(df)
```

**Output:**

```
       amazing  is  is amazing  learning  love  love learning  love nlp  nlp  nlp is
Doc 1        0   0           0         0     1              0         1    1       0
Doc 2        1   1           1         0     0              0         0    1       1
Doc 3        0   0           0         1     1              1         0    0       0
```

### Trigrams

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "I love machine learning very much",
    "Machine learning is fascinating"
]

# Unigrams, Bigrams, and Trigrams
vectorizer = CountVectorizer(ngram_range=(1, 3))
matrix = vectorizer.fit_transform(corpus)

print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
print("Sample features:", list(vectorizer.vocabulary_.keys())[:10])
```

### Character N-Grams

N-grams can also be applied at the character level:

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = ["hello", "world"]

# Character trigrams
char_vectorizer = CountVectorizer(
    analyzer='char',
    ngram_range=(3, 3)
)
char_matrix = char_vectorizer.fit_transform(corpus)

print("Character trigrams:", char_vectorizer.get_feature_names_out())
# Output: ['ell', 'hel', 'llo', 'orl', 'rld', 'wor']
```

**Use cases for character N-grams:**

- Language detection
- Handling misspellings
- Authorship attribution
- Named entity recognition for unseen names

---

## N-Gram Trade-offs

### The Sparsity Problem

As N increases, the vocabulary grows exponentially:

|  N  | Typical Vocabulary Size | Sparsity  |
| :-: | :---------------------: | :-------: |
|  1  |         10,000          |    Low    |
|  2  |         100,000         |  Medium   |
|  3  |       1,000,000+        |   High    |
|  4  |       10,000,000+       | Very High |

### Visualizing the Trade-off

```
                    Context Captured
                           ↑
                           |
    4-gram  ●              |               ● Overfitting Risk
                           |
    Trigram     ●          |          ●
                           |
    Bigram          ●      |     ●
                           |
    Unigram            ●   | ●
                           |
                           +------------------------→
                                    Vocabulary Size
```

### Practical Recommendations

| N-gram Range | When to Use                             |
| :----------- | :-------------------------------------- |
| **(1, 1)**   | Baseline, very large datasets           |
| **(1, 2)**   | Most common choice, balanced            |
| **(1, 3)**   | When phrases matter, medium datasets    |
| **(2, 2)**   | Focus on word pairs only                |
| **(2, 3)**   | Skip individual words, focus on context |

---

## Best Practices

### 1. Start with (1, 2) Range

Unigrams + bigrams is the sweet spot for most tasks:

```python
vectorizer = CountVectorizer(ngram_range=(1, 2))
```

### 2. Limit Vocabulary Size

Always set `max_features` to prevent memory issues:

```python
vectorizer = CountVectorizer(
    ngram_range=(1, 2),
    max_features=10000,  # Keep only top 10K features
    min_df=5             # Ignore n-grams appearing in < 5 docs
)
```

### 3. Remove Stop Word N-Grams

Many bigrams with stop words are useless ("the the", "is is"):

```python
vectorizer = CountVectorizer(
    ngram_range=(1, 2),
    stop_words='english'
)
```

### 4. Use with TF-IDF

N-grams work even better with TF-IDF weighting:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000
)
```

### 5. Consider Domain-Specific N-Grams

Some domains have important multi-word terms:

- Medical: "blood pressure", "heart attack"
- Legal: "breach of contract", "due process"
- Technical: "machine learning", "neural network"

---

## Applications

### 1. Sentiment Analysis

N-grams help capture negation and intensifiers:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Example: sentiment classification
texts = [
    "This movie is not good",
    "This movie is very good",
    "I didn't like the movie",
    "I really liked the movie"
]
labels = [0, 1, 0, 1]  # 0=negative, 1=positive

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(texts)

# Important features will include:
# "not good" (negative), "very good" (positive)
# "didn't like" (negative), "really liked" (positive)
```

### 2. Spam Detection

Spam often has characteristic phrases:

```python
# Spam indicators (bigrams/trigrams)
spam_phrases = [
    "free money",
    "click here",
    "act now",
    "limited time offer",
    "congratulations you won"
]
```

### 3. Language Modeling

N-grams predict the next word:

```python
# Given: "I love machine"
# P(learning | "I love machine") from trigram counts

# Trigram frequency: "love machine learning" = 100
# Bigram frequency: "love machine" = 150
# P(learning | love machine) ≈ 100/150 = 0.67
```

### 4. Named Entity Recognition

Compound names and organizations:

```python
# Important bigrams for NER
entities = [
    "New York",
    "United States",
    "White House",
    "World Health Organization"
]
```

### 5. Text Similarity

N-gram overlap for document comparison:

```python
def ngram_similarity(text1, text2, n=2):
    """Calculate Jaccard similarity based on n-grams"""
    vectorizer = CountVectorizer(ngram_range=(n, n), binary=True)
    matrix = vectorizer.fit_transform([text1, text2])

    # Jaccard similarity
    intersection = (matrix[0].toarray() & matrix[1].toarray()).sum()
    union = (matrix[0].toarray() | matrix[1].toarray()).sum()

    return intersection / union if union > 0 else 0
```

---

## Complete Example: News Classification

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Sample news data
news_data = {
    'text': [
        "Stock market reaches new highs today",
        "Scientists discover new planet in solar system",
        "Football team wins championship",
        "New smartphone released with advanced features",
        "Economic growth exceeds expectations",
        "Researchers find cure for disease",
        "Basketball player breaks scoring record",
        "Tech company announces new product launch"
    ],
    'category': ['business', 'science', 'sports', 'tech',
                 'business', 'science', 'sports', 'tech']
}

df = pd.DataFrame(news_data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['category'], test_size=0.25, random_state=42
)

# Compare unigrams vs unigrams+bigrams
print("=== Unigrams Only ===")
uni_vectorizer = TfidfVectorizer(ngram_range=(1, 1))
X_train_uni = uni_vectorizer.fit_transform(X_train)
print(f"Features: {len(uni_vectorizer.vocabulary_)}")

print("\n=== Unigrams + Bigrams ===")
bi_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_bi = bi_vectorizer.fit_transform(X_train)
print(f"Features: {len(bi_vectorizer.vocabulary_)}")

# Train and evaluate with bigrams
X_test_bi = bi_vectorizer.transform(X_test)
model = MultinomialNB()
model.fit(X_train_bi, y_train)
predictions = model.predict(X_test_bi)

print("\nClassification Report:")
print(classification_report(y_test, predictions))
```

---

## Comparison Summary

| Aspect          | Unigrams | Bigrams  | Trigrams   |
| :-------------- | :------- | :------- | :--------- |
| **Vocabulary**  | Smallest | Medium   | Large      |
| **Context**     | None     | Local    | More local |
| **Sparsity**    | Low      | Medium   | High       |
| **Computation** | Fast     | Moderate | Slow       |
| **Negation**    | Misses   | Captures | Captures   |
| **Phrases**     | Misses   | Captures | Captures   |

---

## Key Takeaways

1. **N-grams capture word order** that Bag of Words misses
2. **Bigrams (1,2) are the most practical** choice for most applications
3. **Higher N increases sparsity** exponentially
4. **Always limit vocabulary size** with `max_features` and `min_df`
5. **Combine with TF-IDF** for better results
6. **Character N-grams** are useful for specific tasks like language detection

---

## Practice Exercise

1. Load a text classification dataset
2. Compare model performance with:
   - Unigrams only (1,1)
   - Unigrams + Bigrams (1,2)
   - Unigrams + Bigrams + Trigrams (1,3)
3. Observe the vocabulary size increase
4. Check if accuracy improves with more N-grams

See [10_bag_of_n_grams.ipynb](10_bag_of_n_grams.ipynb) for a complete implementation.

---

## Further Reading

- [N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf) - Stanford NLP Book
- [Scikit-learn N-gram Documentation](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [Character-level N-grams](https://en.wikipedia.org/wiki/N-gram#Character_n-grams)

---

## Next Steps

➡️ [TF-IDF](tfidf.md) — Weight N-grams by importance
➡️ [Word Embeddings](word_embeddings.md) — Dense semantic representations
