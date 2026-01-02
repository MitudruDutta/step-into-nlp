# Word Embeddings - Complete Guide

## Overview

Word embeddings are dense, low-dimensional vector representations of words that capture semantic meaning and relationships. Unlike Bag of Words or TF-IDF (which produce sparse vectors with thousands of dimensions), embeddings typically have 100-300 dimensions where each dimension represents a learned abstract feature.

The revolutionary insight behind word embeddings is that words with similar meanings end up with similar vectors, enabling mathematical operations on language that preserve semantic relationships.

---

## Table of Contents

1. [Why Word Embeddings](#why-word-embeddings)
2. [The Famous Analogy](#the-famous-analogy)
3. [Types of Word Embeddings](#types-of-word-embeddings)
4. [Word2Vec Deep Dive](#word2vec-deep-dive)
5. [GloVe](#glove)
6. [FastText](#fasttext)
7. [Using spaCy Word Vectors](#using-spacy-word-vectors)
8. [Contextual Embeddings](#contextual-embeddings)
9. [Practical Applications](#practical-applications)
10. [Best Practices](#best-practices)

---

## Why Word Embeddings

### The Problem with Sparse Representations

Traditional methods like Bag of Words and TF-IDF have fundamental limitations:

| Limitation              | Description                                                |
| :---------------------- | :--------------------------------------------------------- |
| **No Semantics**        | "happy" and "joyful" are as different as "happy" and "car" |
| **High Dimensionality** | Vocabulary size can be 100,000+ dimensions                 |
| **Sparsity**            | Most values are zero                                       |
| **No Generalization**   | Cannot understand unseen words                             |

### The Embedding Solution

Word embeddings address all these issues:

```
Sparse Vector (BoW):    [0, 0, 0, 1, 0, 0, ..., 0]  # 100,000 dimensions
Dense Vector (Embed):   [0.25, -0.1, 0.8, ...]      # 300 dimensions
```

| Feature                 | Sparse (BoW/TF-IDF)      | Dense (Embeddings)       |
| :---------------------- | :----------------------- | :----------------------- |
| **Dimensionality**      | 10,000 - 100,000         | 100 - 300                |
| **Sparsity**            | Very sparse (>99% zeros) | Dense (no zeros)         |
| **Semantic similarity** | Not captured             | Captured                 |
| **Memory**              | High                     | Low                      |
| **Training required**   | No                       | Yes (or use pre-trained) |

---

## The Famous Analogy

The most striking demonstration of word embeddings is the ability to perform semantic arithmetic:

$$\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}$$

### What This Means

The vector operation captures:

- **"king"** contains the concept of "royalty" + "male"
- Subtracting **"man"** removes the "male" concept
- Adding **"woman"** adds the "female" concept
- Result is close to **"queen"** (royalty + female)

### More Analogies

| Analogy                 | Vector Arithmetic                |
| :---------------------- | :------------------------------- |
| man:woman :: king:?     | king - man + woman ≈ queen       |
| Paris:France :: Tokyo:? | Tokyo - Paris + France ≈ Japan   |
| big:bigger :: small:?   | small - big + bigger ≈ smaller   |
| walk:walking :: swim:?  | swim - walk + walking ≈ swimming |

### Code Example

```python
import spacy
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_lg")

# Get word vectors from vocabulary
king = nlp.vocab["king"].vector
man = nlp.vocab["man"].vector
woman = nlp.vocab["woman"].vector
queen = nlp.vocab["queen"].vector

# Perform the analogy: king - man + woman
result = king - man + woman

# Check similarity to "queen"
similarity = cosine_similarity([result], [queen])
print(f"Similarity to 'queen': {similarity[0][0]:.3f}")
# Output: ~0.72
```

---

## Types of Word Embeddings

### Overview Comparison

| Model        | Training Approach                  | Key Characteristics         | OOV Handling     |
| :----------- | :--------------------------------- | :-------------------------- | :--------------- |
| **Word2Vec** | Prediction-based (CBOW, Skip-gram) | Fast, good for analogies    | ❌ Cannot handle |
| **GloVe**    | Count-based + factorization        | Captures global statistics  | ❌ Cannot handle |
| **FastText** | Character n-grams                  | Handles morphology          | ✅ Can handle    |
| **ELMo**     | Bidirectional LSTM                 | Context-dependent           | ✅ Can handle    |
| **BERT**     | Transformer encoder                | State-of-the-art contextual | ✅ Can handle    |

### Static vs Contextual Embeddings

**Static Embeddings** (Word2Vec, GloVe, FastText):

- Same vector for a word regardless of context
- "bank" has one vector for both "river bank" and "money bank"

**Contextual Embeddings** (ELMo, BERT):

- Different vectors based on context
- "bank" gets different vectors for different meanings

---

## Word2Vec Deep Dive

Word2Vec, developed by Google in 2013, revolutionized NLP by demonstrating that neural networks could learn meaningful word representations.

### Two Architectures

#### CBOW (Continuous Bag of Words)

Predicts the target word from surrounding context words.

```
Context: ["The", "cat", "___", "on", "the", "mat"]
Target: "sat"

Input: [The, cat, on, the, mat] → Neural Network → Output: sat
```

**Characteristics:**

- Faster to train
- Better for frequent words
- Smooths over distributional information

#### Skip-gram

Predicts context words from the target word.

```
Target: "sat"
Predict: ["The", "cat", "on", "the", "mat"]

Input: sat → Neural Network → Output: [The, cat, on, the, mat]
```

**Characteristics:**

- Better for rare words
- Works well with small datasets
- Each target-context pair is a training example

### Visual Comparison

```
CBOW:
[The] [cat] [?] [on] [mat]  →  Neural Network  →  [sat]
      ↓    ↓       ↓    ↓
    Context Words           →      Target Word

Skip-gram:
          [sat]             →  Neural Network  →  [The, cat, on, mat]
            ↓
       Target Word          →    Context Words
```

### Training Word2Vec with Gensim

```python
from gensim.models import Word2Vec

# Sample corpus (list of tokenized sentences)
sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "in", "the", "park"],
    ["cats", "and", "dogs", "are", "pets"],
    ["the", "mat", "is", "on", "the", "floor"]
]

# Train Word2Vec model
model = Word2Vec(
    sentences=sentences,
    vector_size=100,     # Embedding dimensions
    window=5,            # Context window size
    min_count=1,         # Minimum word frequency
    sg=1,                # 1 for Skip-gram, 0 for CBOW
    workers=4            # Parallel training threads
)

# Get word vector
cat_vector = model.wv['cat']
print(f"Vector shape: {cat_vector.shape}")

# Find similar words
similar_words = model.wv.most_similar('cat', topn=5)
print("Similar to 'cat':", similar_words)
```

---

## GloVe

GloVe (Global Vectors for Word Representation), developed by Stanford, combines count-based and prediction-based methods.

### Key Idea

GloVe learns from the **co-occurrence matrix** - how often words appear together across the entire corpus.

$$J = \sum_{i,j=1}^{V} f(X_{ij}) \left( \vec{w}_i^T \vec{\tilde{w}}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2$$

Where:

- $X_{ij}$ = co-occurrence count of words i and j
- $\vec{w}_i$ = word vector for word i
- $f(x)$ = weighting function to handle rare co-occurrences

### Advantages

- Captures global corpus statistics
- Efficient training
- Good performance on word analogy tasks

### Using Pre-trained GloVe

```python
import numpy as np

def load_glove(file_path):
    """Load GloVe embeddings from file"""
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Load pre-trained GloVe
glove = load_glove('glove.6B.300d.txt')

# Get vector
king_vector = glove['king']
print(f"'king' vector shape: {king_vector.shape}")

# Compute similarity
def cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print(f"king-queen similarity: {cosine_sim(glove['king'], glove['queen']):.3f}")
```

---

## FastText

FastText, developed by Facebook, extends Word2Vec by representing words as bags of character n-grams.

### Key Innovation

Instead of learning a vector for "where", FastText learns vectors for:

- The word itself: "where"
- Character n-grams: "<wh", "whe", "her", "ere", "re>"

The word vector is the sum of all its n-gram vectors.

### Advantages

1. **Handles OOV words**: Even unseen words can get vectors from their n-grams
2. **Captures morphology**: "running", "runner", "ran" share character patterns
3. **Works well for morphologically rich languages**

### Example

```
Word: "where"
Character n-grams (n=3): <wh, whe, her, ere, re>

Word: "wherever"
Character n-grams (n=3): <wh, whe, her, ere, rev, eve, ver, er>

Shared n-grams: <wh, whe, her, ere → Similar vectors!
```

### Training FastText

```python
from gensim.models import FastText

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "in", "the", "park"],
    ["cats", "and", "dogs", "are", "pets"]
]

# Train FastText model
model = FastText(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    min_n=2,        # Minimum n-gram length
    max_n=5         # Maximum n-gram length
)

# Get vector for known word
cat_vector = model.wv['cat']

# Get vector for OOV word!
unknown_vector = model.wv['catlike']  # Works even if not in training data!
```

---

## Using spaCy Word Vectors

spaCy provides easy access to pre-trained word vectors through its language models.

### Loading Models

```python
import spacy

# Small model (no vectors)
# nlp = spacy.load("en_core_web_sm")

# Medium model (480k keys, 20k unique vectors)
# nlp = spacy.load("en_core_web_md")

# Large model (685k keys, 685k unique vectors)
nlp = spacy.load("en_core_web_lg")
```

### Checking Vector Properties

```python
import spacy

nlp = spacy.load("en_core_web_lg")

doc = nlp("dog cat banana")
for token in doc:
    print(f"{token.text}:")
    print(f"  has_vector: {token.has_vector}")
    print(f"  vector_shape: {token.vector.shape}")
    print(f"  is_oov: {token.is_oov}")
```

**Output:**

```
dog:
  has_vector: True
  vector_shape: (300,)
  is_oov: False
cat:
  has_vector: True
  vector_shape: (300,)
  is_oov: False
banana:
  has_vector: True
  vector_shape: (300,)
  is_oov: False
```

### Computing Semantic Similarity

```python
import spacy

nlp = spacy.load("en_core_web_lg")

# Word-to-word similarity
base_word = nlp("bread")
words = nlp("sandwich burger pizza car airplane")

print("Similarity to 'bread':")
for token in words:
    similarity = token.similarity(base_word)
    print(f"  {token.text}: {similarity:.3f}")
```

**Output:**

```
Similarity to 'bread':
  sandwich: 0.562
  burger: 0.474
  pizza: 0.432
  car: 0.121
  airplane: 0.087
```

### Document Similarity

```python
import spacy

nlp = spacy.load("en_core_web_lg")

doc1 = nlp("I love machine learning and artificial intelligence")
doc2 = nlp("Deep learning is a subset of machine learning")
doc3 = nlp("I enjoy eating pizza and pasta")

print(f"doc1 vs doc2: {doc1.similarity(doc2):.3f}")  # High similarity
print(f"doc1 vs doc3: {doc1.similarity(doc3):.3f}")  # Low similarity
```

### Direct Vocabulary Access

```python
import spacy

nlp = spacy.load("en_core_web_lg")

# Access vectors directly from vocabulary
king = nlp.vocab["king"].vector
queen = nlp.vocab["queen"].vector
man = nlp.vocab["man"].vector
woman = nlp.vocab["woman"].vector

# Perform analogy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

result = king - man + woman
similarity = cosine_similarity([result], [queen])[0][0]
print(f"king - man + woman ≈ queen: {similarity:.3f}")
```

---

## Contextual Embeddings

Modern NLP has moved beyond static embeddings to contextual representations.

### The Problem with Static Embeddings

Consider the word "bank":

- "I deposited money in the bank"
- "The river bank was muddy"

With Word2Vec/GloVe, "bank" has the **same vector** in both sentences, despite having completely different meanings.

### ELMo (Embeddings from Language Models)

ELMo generates context-dependent embeddings using bidirectional LSTMs:

```python
# Conceptual example (requires TensorFlow Hub)
# bank1 = elmo("I deposited money in the bank")["bank"]
# bank2 = elmo("The river bank was muddy")["bank"]
# bank1 ≠ bank2  # Different vectors!
```

### BERT (Bidirectional Encoder Representations from Transformers)

BERT provides state-of-the-art contextual embeddings:

```python
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Get contextual embeddings
text = "The bank was closed for the holiday"
inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)

# outputs.last_hidden_state contains contextual embeddings
# Shape: [1, sequence_length, 768]
embeddings = outputs.last_hidden_state
print(f"Embedding shape: {embeddings.shape}")
```

### Static vs Contextual Comparison

| Aspect          | Static (Word2Vec)            | Contextual (BERT)             |
| :-------------- | :--------------------------- | :---------------------------- |
| **Polysemy**    | One vector per word          | Different vectors per context |
| **Training**    | Unsupervised on large corpus | Pre-trained + fine-tuning     |
| **Speed**       | Fast lookup                  | Slower (model inference)      |
| **Memory**      | Low                          | High                          |
| **Performance** | Good                         | State-of-the-art              |

---

## Practical Applications

### 1. Semantic Search

```python
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nlp = spacy.load("en_core_web_lg")

# Document collection
documents = [
    "Python is a programming language",
    "Machine learning uses algorithms to learn from data",
    "Natural language processing analyzes text",
    "Pizza is my favorite food"
]

# Create document vectors
doc_vectors = np.array([nlp(doc).vector for doc in documents])

# Search query
query = "artificial intelligence and data science"
query_vector = nlp(query).vector.reshape(1, -1)

# Find most similar documents
similarities = cosine_similarity(query_vector, doc_vectors)[0]
ranked_indices = similarities.argsort()[::-1]

print("Search Results:")
for i, idx in enumerate(ranked_indices):
    print(f"{i+1}. {documents[idx]} (score: {similarities[idx]:.3f})")
```

### 2. Text Classification with Embeddings

```python
import spacy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

nlp = spacy.load("en_core_web_lg")

# Sample data
texts = [
    "Great movie, loved it!",
    "Terrible film, waste of time",
    "Amazing performance by the actors",
    "Boring and predictable plot"
]
labels = [1, 0, 1, 0]  # 1=positive, 0=negative

# Convert texts to vectors
X = np.array([nlp(text).vector for text in texts])
y = np.array(labels)

# Train classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predict
new_text = "This was an excellent movie"
new_vector = nlp(new_text).vector.reshape(1, -1)
prediction = clf.predict(new_vector)
print(f"Prediction: {'Positive' if prediction[0] == 1 else 'Negative'}")
```

### 3. Finding Similar Words

```python
import spacy

nlp = spacy.load("en_core_web_lg")

def find_similar_words(word, topn=10):
    """Find most similar words using spaCy vectors"""
    word_vector = nlp.vocab[word].vector

    # Compare with all words in vocabulary
    similarities = []
    for w in nlp.vocab:
        if w.has_vector and w.text.isalpha() and w.text != word:
            sim = nlp.vocab[word].similarity(w)
            similarities.append((w.text, sim))

    # Sort by similarity
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:topn]

# Find similar words
similar = find_similar_words("computer")
print("Words similar to 'computer':")
for word, sim in similar:
    print(f"  {word}: {sim:.3f}")
```

### 4. Clustering Documents

```python
import spacy
import numpy as np
from sklearn.cluster import KMeans

nlp = spacy.load("en_core_web_lg")

documents = [
    "Python programming tutorial",
    "Machine learning algorithms",
    "Cooking recipes for dinner",
    "JavaScript web development",
    "Baking chocolate cake",
    "Deep learning neural networks"
]

# Get document vectors
doc_vectors = np.array([nlp(doc).vector for doc in documents])

# Cluster
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(doc_vectors)

for doc, cluster in zip(documents, clusters):
    print(f"Cluster {cluster}: {doc}")
```

---

## Best Practices

### 1. Choose the Right Model Size

```python
# For development/testing
nlp = spacy.load("en_core_web_sm")  # No vectors, fast

# For production with vectors
nlp = spacy.load("en_core_web_md")  # Good balance

# For best quality
nlp = spacy.load("en_core_web_lg")  # Full vectors
```

### 2. Handle OOV Words

```python
import spacy

nlp = spacy.load("en_core_web_lg")

def get_vector_safe(word):
    """Get vector with OOV handling"""
    token = nlp.vocab[word]
    if token.has_vector:
        return token.vector
    else:
        # Fallback: average of subwords or zero vector
        return nlp("").vector  # Zero vector

# Or use FastText which handles OOV naturally
```

### 3. Normalize Vectors

```python
import numpy as np

def normalize(vector):
    """L2 normalize a vector"""
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector

# Normalized vectors work better with cosine similarity
```

### 4. Average Word Vectors for Documents

```python
import spacy
import numpy as np

nlp = spacy.load("en_core_web_lg")

def document_vector(doc_text):
    """Compute document vector as average of word vectors"""
    doc = nlp(doc_text)
    vectors = [token.vector for token in doc if token.has_vector]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(300)  # Return zero vector if no words have vectors
```

### 5. Consider Domain-Specific Embeddings

For specialized domains (medical, legal, scientific), consider:

- Fine-tuning pre-trained embeddings on domain corpus
- Using domain-specific pre-trained models (BioBERT, LegalBERT)

---

## Key Takeaways

1. **Word embeddings** represent words as dense vectors capturing semantic meaning
2. **Similar words** have similar vectors (cosine similarity)
3. **Semantic arithmetic** is possible: king - man + woman ≈ queen
4. **Word2Vec** (Skip-gram, CBOW) learns from local context
5. **GloVe** learns from global co-occurrence statistics
6. **FastText** handles OOV words via character n-grams
7. **Contextual embeddings** (BERT, ELMo) provide different vectors based on context
8. **spaCy** provides easy access to pre-trained vectors

---

## Practice Exercise

1. Load `en_core_web_lg` spaCy model
2. Find the 10 most similar words to "computer"
3. Verify the king-queen analogy
4. Build a simple document similarity system
5. Compare similarity scores for related vs unrelated word pairs

See [spacy_word_vectors.ipynb](spacy_word_vectors.ipynb) for a complete implementation.

---

## Further Reading

- [Word2Vec Paper](https://arxiv.org/abs/1301.3781)
- [GloVe Paper](https://nlp.stanford.edu/pubs/glove.pdf)
- [FastText Paper](https://arxiv.org/abs/1607.04606)
- [spaCy Word Vectors Guide](https://spacy.io/usage/vectors-similarity)
- [BERT Paper](https://arxiv.org/abs/1810.04805)

---

## Next Steps

➡️ [Text Classification](text_classification.md) — Apply embeddings in a real project
