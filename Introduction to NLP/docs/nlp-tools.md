# 🛠️ The NLP Toolbelt

## Overview

The NLP ecosystem offers a variety of tools, each suited for different use cases. Understanding when to use each tool is crucial for building effective applications.

This guide provides an in-depth exploration of the major NLP libraries, their strengths, weaknesses, and practical usage examples.

---

## 📊 Tool Categories at a Glance

```text
                           NLP Tools Landscape
    ┌────────────────────────────────────────────────────────────┐
    │                                                            │
    │   🚀 State-of-the-Art        ⚡ Production-Grade           │
    │   ─────────────────          ──────────────────            │
    │   • Hugging Face             • spaCy                       │
    │   • OpenAI API               • Gensim                      │
    │   • LangChain                                              │
    │   • LlamaIndex                                             │
    │                                                            │
    │   📖 Research & Education    🔧 Specialized                │
    │   ─────────────────────      ────────────                  │
    │   • NLTK                     • Flair                       │
    │                              • Stanza                      │
    │                              • AllenNLP                    │
    │                                                            │
    └────────────────────────────────────────────────────────────┘
```

### Quick Comparison

| Category                | Tools                           | Best For                     | Performance | Learning Curve |
| :---------------------- | :------------------------------ | :--------------------------- | :---------- | :------------- |
| **🚀 State-of-the-Art** | Hugging Face, OpenAI, LangChain | Modern high-performance apps | ⭐⭐⭐⭐⭐  | Medium-High    |
| **⚡ Production-Grade** | spaCy, Gensim                   | Speed and efficiency         | ⭐⭐⭐⭐    | Medium         |
| **📖 Educational**      | NLTK                            | Learning and research        | ⭐⭐⭐      | Low            |

---

## 🤗 Hugging Face Transformers

### What is it?

The most popular library for working with transformer models, providing access to thousands of pre-trained models.

### Key Features

| Feature        | Description                 |
| :------------- | :-------------------------- |
| **Model Hub**  | 200,000+ pre-trained models |
| **Pipelines**  | Easy-to-use inference API   |
| **Trainer**    | Simplified fine-tuning      |
| **Tokenizers** | Fast tokenization library   |
| **Datasets**   | Access to 50,000+ datasets  |

### Strengths

- ✅ State-of-the-art performance
- ✅ Vast model selection
- ✅ Active community and updates
- ✅ Excellent documentation
- ✅ Easy to fine-tune models

### Weaknesses

- ❌ Requires GPU for best performance
- ❌ Large model sizes
- ❌ Higher computational costs
- ❌ Steeper learning curve for customization

### Installation

```bash
pip install transformers datasets accelerate
```

### Quick Start Examples

#### Sentiment Analysis Pipeline

```python
from transformers import pipeline

# Load sentiment analysis pipeline
classifier = pipeline("sentiment-analysis")

# Single prediction
result = classifier("I love learning about NLP!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Batch prediction
texts = [
    "This movie was fantastic!",
    "The food was terrible.",
    "It was okay, nothing special."
]
results = classifier(texts)
for text, result in zip(texts, results):
    print(f"{text} → {result['label']} ({result['score']:.2%})")
```

#### Named Entity Recognition

```python
from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)

text = "Elon Musk founded SpaceX in California and later acquired Twitter."
entities = ner(text)

for entity in entities:
    print(f"{entity['word']}: {entity['entity_group']} ({entity['score']:.2%})")
# Elon Musk: PER (99.87%)
# SpaceX: ORG (99.45%)
# California: LOC (99.92%)
# Twitter: ORG (98.76%)
```

#### Text Generation

```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = "Natural language processing is"
result = generator(
    prompt,
    max_length=50,
    num_return_sequences=1,
    temperature=0.7
)
print(result[0]['generated_text'])
```

#### Question Answering

```python
from transformers import pipeline

qa = pipeline("question-answering")

context = """
Natural Language Processing (NLP) is a field of artificial intelligence
that focuses on the interaction between computers and humans through
natural language. The ultimate goal of NLP is to enable computers to
understand, interpret, and generate human language in a valuable way.
"""

question = "What is the goal of NLP?"
result = qa(question=question, context=context)
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['score']:.2%}")
```

#### Fine-tuning a Model

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

# Load dataset
dataset = load_dataset("imdb")

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# Tokenize dataset
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

tokenized_dataset = dataset.map(tokenize, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"].select(range(1000)),
    eval_dataset=tokenized_dataset["test"].select(range(500)),
)

trainer.train()
```

### When to Use Hugging Face

| ✅ Use For                          | ❌ Avoid For                       |
| :---------------------------------- | :--------------------------------- |
| State-of-the-art performance needed | Simple text processing tasks       |
| Pre-trained model availability      | Limited computational resources    |
| Fine-tuning on custom data          | Real-time low-latency requirements |
| Research and experimentation        | Small datasets                     |

---

## 🔷 spaCy

### What is it?

An industrial-strength NLP library designed for production use, with focus on speed and efficiency.

### Key Features

| Feature          | Description                            |
| :--------------- | :------------------------------------- |
| **Speed**        | Optimized Cython code                  |
| **Pipelines**    | Customizable processing pipelines      |
| **Models**       | Pre-trained statistical models         |
| **NER**          | Built-in named entity recognition      |
| **Integrations** | Works with transformers, deep learning |

### Strengths

- ✅ Extremely fast processing
- ✅ Production-ready out of the box
- ✅ Excellent documentation
- ✅ Consistent API design
- ✅ Easy to extend and customize

### Weaknesses

- ❌ Fewer pre-trained models than Hugging Face
- ❌ Less flexible for research
- ❌ Commercial models require license
- ❌ Limited language support compared to NLTK

### Installation

```bash
pip install spacy
python -m spacy download en_core_web_sm  # Small model
python -m spacy download en_core_web_md  # Medium model (with word vectors)
python -m spacy download en_core_web_lg  # Large model
python -m spacy download en_core_web_trf # Transformer model
```

### Model Comparison

| Model             | Size   | Speed  | Accuracy | Vectors    |
| :---------------- | :----- | :----- | :------- | :--------- |
| `en_core_web_sm`  | 12 MB  | ⚡⚡⚡ | Good     | No         |
| `en_core_web_md`  | 40 MB  | ⚡⚡   | Better   | Yes (300d) |
| `en_core_web_lg`  | 560 MB | ⚡     | Best     | Yes (300d) |
| `en_core_web_trf` | 438 MB | 🐢     | Best     | Contextual |

### Quick Start Examples

#### Basic Processing Pipeline

```python
import spacy

# Load model
nlp = spacy.load("en_core_web_sm")

# Process text
text = "Apple is looking at buying a startup in California for $1 billion."
doc = nlp(text)

# Tokenization
print("Tokens:")
for token in doc:
    print(f"  {token.text:12} | {token.pos_:6} | {token.dep_:10} | {token.lemma_}")
```

#### Named Entity Recognition

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Elon Musk founded SpaceX in 2002. The company is based in Hawthorne, California.")

print("Named Entities:")
for ent in doc.ents:
    print(f"  {ent.text:20} | {ent.label_:10} | {spacy.explain(ent.label_)}")

# Output:
# Elon Musk            | PERSON     | People, including fictional
# SpaceX               | ORG        | Companies, agencies, institutions
# 2002                 | DATE       | Absolute or relative dates or periods
# Hawthorne            | GPE        | Countries, cities, states
# California           | GPE        | Countries, cities, states
```

#### Dependency Parsing

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The quick brown fox jumps over the lazy dog.")

print("Dependency Parse:")
for token in doc:
    print(f"  {token.text:10} ─({token.dep_:10})─> {token.head.text}")

# Visualize in Jupyter
from spacy import displacy
displacy.render(doc, style="dep", jupyter=True)
```

#### Custom Pipeline Component

```python
import spacy
from spacy.language import Language

@Language.component("custom_sentencizer")
def custom_sentencizer(doc):
    """Custom component that adds sentence boundaries"""
    for token in doc[:-1]:
        if token.text in [".", "!", "?"]:
            doc[token.i + 1].is_sent_start = True
    return doc

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("custom_sentencizer", before="parser")

doc = nlp("Hello world. How are you? I'm fine!")
for sent in doc.sents:
    print(sent.text)
```

#### Text Similarity

```python
import spacy

# Need medium or large model for word vectors
nlp = spacy.load("en_core_web_md")

doc1 = nlp("I like pizza")
doc2 = nlp("I enjoy eating Italian food")
doc3 = nlp("The weather is nice today")

print(f"doc1 vs doc2: {doc1.similarity(doc2):.2f}")  # Higher similarity
print(f"doc1 vs doc3: {doc1.similarity(doc3):.2f}")  # Lower similarity
```

#### Rule-Based Matching

```python
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Define pattern for phone numbers
phone_pattern = [
    {"SHAPE": "ddd"},
    {"ORTH": "-"},
    {"SHAPE": "ddd"},
    {"ORTH": "-"},
    {"SHAPE": "dddd"}
]
matcher.add("PHONE_NUMBER", [phone_pattern])

doc = nlp("Call me at 123-456-7890 or 987-654-3210")
matches = matcher(doc)

for match_id, start, end in matches:
    span = doc[start:end]
    print(f"Found: {span.text}")
```

### When to Use spaCy

| ✅ Use For                  | ❌ Avoid For                    |
| :-------------------------- | :------------------------------ |
| Production NLP pipelines    | Research/experimentation        |
| Speed-critical applications | Cutting-edge model performance  |
| Named Entity Recognition    | Highly customized architectures |
| Text preprocessing at scale | Languages without models        |

---

## 📚 NLTK (Natural Language Toolkit)

### What is it?

The classic NLP library for Python, comprehensive and educational, with a vast collection of text processing libraries and linguistic data.

### Key Features

| Feature           | Description                          |
| :---------------- | :----------------------------------- |
| **Corpora**       | 100+ linguistic corpora and lexicons |
| **Algorithms**    | Wide range of NLP algorithms         |
| **Educational**   | Excellent learning resources         |
| **Extensibility** | Easy to prototype new ideas          |

### Strengths

- ✅ Comprehensive and educational
- ✅ Extensive corpora and resources
- ✅ Great for learning NLP
- ✅ Wide algorithm coverage
- ✅ Strong linguistic foundation

### Weaknesses

- ❌ Slower than spaCy
- ❌ Not production-optimized
- ❌ Older API design
- ❌ Less maintained than alternatives

### Installation

```bash
pip install nltk

# Download data
python -c "import nltk; nltk.download('popular')"
```

### Quick Start Examples

#### Tokenization

```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

text = "Hello Mr. Smith! How are you doing today? The weather is great."

# Sentence tokenization
sentences = sent_tokenize(text)
print("Sentences:", sentences)

# Word tokenization
words = word_tokenize(text)
print("Words:", words)
```

#### Part-of-Speech Tagging

```python
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize

text = "The quick brown fox jumps over the lazy dog"
tokens = word_tokenize(text)
tagged = pos_tag(tokens)

print("POS Tags:")
for word, tag in tagged:
    print(f"  {word:10} → {tag}")
```

#### Stemming and Lemmatization

```python
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ["running", "runs", "ran", "easily", "fairly", "better"]

print("Word          | Stem       | Lemma")
print("-" * 40)
for word in words:
    stem = stemmer.stem(word)
    lemma = lemmatizer.lemmatize(word, pos='v')
    print(f"{word:13} | {stem:10} | {lemma}")
```

#### Stopword Removal

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

text = "This is an example showing how to remove stopwords from text"
tokens = word_tokenize(text.lower())

filtered = [word for word in tokens if word not in stop_words]
print("Original:", tokens)
print("Filtered:", filtered)
```

#### Frequency Distribution

```python
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import gutenberg

# Load a book
text = gutenberg.raw('austen-emma.txt')
tokens = word_tokenize(text.lower())

# Remove punctuation
words = [word for word in tokens if word.isalpha()]

# Frequency distribution
fdist = FreqDist(words)
print("Most common words:")
for word, count in fdist.most_common(10):
    print(f"  {word}: {count}")

# Plot (if matplotlib installed)
# fdist.plot(30)
```

#### Named Entity Recognition (Chunking)

```python
import nltk
from nltk import ne_chunk, pos_tag
from nltk.tokenize import word_tokenize

text = "Barack Obama was the 44th President of the United States."
tokens = word_tokenize(text)
tagged = pos_tag(tokens)
entities = ne_chunk(tagged)

print(entities)
# Can visualize with: entities.draw()
```

#### Sentiment Analysis with VADER

```python
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

texts = [
    "This movie is absolutely wonderful!",
    "The food was terrible and the service was slow.",
    "It was an okay experience, nothing special.",
]

for text in texts:
    scores = sia.polarity_scores(text)
    print(f"\n'{text}'")
    print(f"  Positive: {scores['pos']:.2f}")
    print(f"  Negative: {scores['neg']:.2f}")
    print(f"  Neutral:  {scores['neu']:.2f}")
    print(f"  Compound: {scores['compound']:.2f}")
```

### When to Use NLTK

| ✅ Use For                   | ❌ Avoid For            |
| :--------------------------- | :---------------------- |
| Learning NLP concepts        | Production applications |
| Academic research            | Speed-critical tasks    |
| Accessing linguistic corpora | Large-scale processing  |
| Prototyping ideas            | Modern deep learning    |

---

## 📊 Gensim

### What is it?

A library for topic modeling and document similarity, specialized in unsupervised learning on large text corpora.

### Key Features

| Feature             | Description                         |
| :------------------ | :---------------------------------- |
| **Topic Modeling**  | LDA, LSI, HDP implementations       |
| **Word Embeddings** | Word2Vec, FastText, Doc2Vec         |
| **Scalability**     | Streams data, handles large corpora |
| **Similarity**      | Document similarity queries         |

### Strengths

- ✅ Memory-efficient streaming
- ✅ Excellent for topic modeling
- ✅ Word2Vec implementation
- ✅ Handles large datasets
- ✅ Well-documented

### Weaknesses

- ❌ Limited to specific tasks
- ❌ Not for general NLP
- ❌ Requires preprocessing
- ❌ Slower training than some alternatives

### Installation

```bash
pip install gensim
```

### Quick Start Examples

#### Word2Vec Training

```python
from gensim.models import Word2Vec

# Sample corpus (list of tokenized sentences)
sentences = [
    ["machine", "learning", "is", "fascinating"],
    ["deep", "learning", "uses", "neural", "networks"],
    ["natural", "language", "processing", "involves", "text"],
    ["nlp", "and", "machine", "learning", "are", "related"],
    ["word", "embeddings", "capture", "semantic", "meaning"],
]

# Train Word2Vec model
model = Word2Vec(
    sentences,
    vector_size=100,  # Embedding dimension
    window=5,         # Context window
    min_count=1,      # Minimum word frequency
    workers=4,        # Parallel training
    epochs=100        # Training iterations
)

# Get word vector
vector = model.wv["learning"]
print(f"Vector for 'learning': {vector[:5]}...")  # First 5 dimensions

# Find similar words
similar = model.wv.most_similar("learning", topn=5)
print("\nWords similar to 'learning':")
for word, score in similar:
    print(f"  {word}: {score:.3f}")

# Word analogies
# result = model.wv.most_similar(positive=['king', 'woman'], negative=['man'])
```

#### Topic Modeling with LDA

```python
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from nltk.tokenize import word_tokenize

# Sample documents
documents = [
    "Machine learning algorithms learn from data",
    "Neural networks are used in deep learning",
    "NLP processes human language with computers",
    "Text classification is an NLP task",
    "Clustering groups similar documents together",
]

# Preprocess
def preprocess(text):
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t.isalpha() and t not in STOPWORDS]

processed_docs = [preprocess(doc) for doc in documents]

# Create dictionary and corpus
dictionary = corpora.Dictionary(processed_docs)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# Train LDA model
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=2,
    passes=15,
    random_state=42
)

# Print topics
print("Discovered Topics:")
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic {idx}: {topic}")
```

#### Document Similarity

```python
from gensim import corpora, models, similarities
from gensim.parsing.preprocessing import preprocess_string

documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with many layers",
    "Natural language processing deals with human language",
    "Computer vision processes images and videos",
    "Reinforcement learning learns through trial and error",
]

# Preprocess
processed = [preprocess_string(doc) for doc in documents]

# Create dictionary and corpus
dictionary = corpora.Dictionary(processed)
corpus = [dictionary.doc2bow(doc) for doc in processed]

# Create TF-IDF model
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

# Create similarity index
index = similarities.SparseMatrixSimilarity(corpus_tfidf, num_features=len(dictionary))

# Query
query = "artificial intelligence and machine learning"
query_processed = preprocess_string(query)
query_bow = dictionary.doc2bow(query_processed)
query_tfidf = tfidf[query_bow]

# Get similarities
sims = index[query_tfidf]
print(f"\nQuery: '{query}'")
print("\nSimilarities:")
for idx, sim in sorted(enumerate(sims), key=lambda x: x[1], reverse=True):
    print(f"  {sim:.3f}: {documents[idx]}")
```

### When to Use Gensim

| ✅ Use For          | ❌ Avoid For             |
| :------------------ | :----------------------- |
| Topic modeling      | General NLP tasks        |
| Word embeddings     | Text classification      |
| Document similarity | Named entity recognition |
| Large corpora       | Real-time processing     |

---

## 🔗 Combining Tools

### Real-World Integration Example

```python
"""
Example: Building a complete NLP pipeline using multiple tools
Task: Analyze customer reviews for topics and sentiment
"""

import spacy
from transformers import pipeline
from gensim import corpora
from gensim.models import LdaModel

# 1. Use spaCy for preprocessing (fast and efficient)
nlp = spacy.load("en_core_web_sm")

def preprocess_with_spacy(text):
    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]
    return tokens

# 2. Use Hugging Face for sentiment analysis (accurate)
sentiment_analyzer = pipeline("sentiment-analysis")

# 3. Use Gensim for topic modeling (efficient)
def extract_topics(documents, num_topics=3):
    processed = [preprocess_with_spacy(doc) for doc in documents]
    dictionary = corpora.Dictionary(processed)
    corpus = [dictionary.doc2bow(doc) for doc in processed]
    lda = LdaModel(corpus, num_topics=num_topics, id2word=dictionary)
    return lda

# Sample reviews
reviews = [
    "The product quality is excellent! Fast shipping too.",
    "Terrible experience. Product arrived damaged.",
    "Great customer service, they resolved my issue quickly.",
    "The delivery was late and the packaging was poor.",
    "Amazing product, exactly what I expected. Will buy again!",
]

# Analyze
print("=" * 60)
print("CUSTOMER REVIEW ANALYSIS")
print("=" * 60)

print("\n📊 SENTIMENT ANALYSIS (Hugging Face)")
print("-" * 40)
for review in reviews:
    result = sentiment_analyzer(review)[0]
    emoji = "😊" if result['label'] == 'POSITIVE' else "😞"
    print(f"{emoji} {result['label']:8} ({result['score']:.1%}): {review[:50]}...")

print("\n📚 TOPIC MODELING (Gensim)")
print("-" * 40)
lda_model = extract_topics(reviews, num_topics=2)
for idx, topic in lda_model.print_topics():
    print(f"Topic {idx + 1}: {topic}")

print("\n🔍 ENTITY EXTRACTION (spaCy)")
print("-" * 40)
for review in reviews[:2]:
    doc = nlp(review)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    print(f"Review: {review}")
    print(f"Entities: {entities if entities else 'None found'}\n")
```

---

## 📋 Tool Selection Guide

### Decision Flowchart

```text
                    What do you need?
                          │
         ┌────────────────┼────────────────┐
         │                │                │
    Learning NLP    Production App    Research/SOTA
         │                │                │
         ▼                ▼                ▼
       NLTK            spaCy         Hugging Face
                          │
              ┌───────────┴───────────┐
              │                       │
        Simple NLP              Deep Learning
              │                       │
              ▼                       ▼
           spaCy              Hugging Face
```

### Quick Reference

| Task                         | Recommended Tool      | Alternative  |
| :--------------------------- | :-------------------- | :----------- |
| **Learning NLP**             | NLTK                  | spaCy        |
| **Production Pipeline**      | spaCy                 | Hugging Face |
| **Text Classification**      | Hugging Face          | spaCy        |
| **Named Entity Recognition** | spaCy                 | Hugging Face |
| **Sentiment Analysis**       | Hugging Face          | NLTK (VADER) |
| **Topic Modeling**           | Gensim                | —            |
| **Word Embeddings**          | Gensim / Hugging Face | spaCy        |
| **Text Generation**          | Hugging Face          | —            |
| **Question Answering**       | Hugging Face          | —            |

---

## 🔗 Navigation

← [NLP Pipeline](nlp-pipeline.md) | [Back to Introduction](README.md)

---

_Choose the right tool for the right job. Often, the best solutions combine multiple tools to leverage their individual strengths!_
