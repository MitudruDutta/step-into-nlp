# Complete NLP Pipeline 🔄

This guide demonstrates how to combine all text preprocessing techniques into a complete, production-ready NLP pipeline using spaCy.

---

## 📖 What is an NLP Pipeline?

An **NLP pipeline** is a sequence of processing steps that transform raw text into structured, analyzable data. Each step builds upon the previous one.

```
Raw Text → Tokenization → POS Tagging → Lemmatization → NER → Clean Output
```

---

## 🎯 Why Use a Pipeline?

1. **Consistency**: Same processing for all text
2. **Efficiency**: Process once, use many components
3. **Modularity**: Add/remove components as needed
4. **Reproducibility**: Identical results every time
5. **Production-ready**: Easy to deploy and maintain

---

## 📊 spaCy Pipeline Components

### Understanding spaCy Pipelines

```python
import spacy

# Blank model - no components, just tokenization
nlp_blank = spacy.blank("en")
print(f"Blank model components: {nlp_blank.pipe_names}")
# Output: []

# Pre-trained model - full pipeline
nlp = spacy.load("en_core_web_sm")
print(f"Pre-trained model components: {nlp.pipe_names}")
# Output: ['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner']
```

### Pipeline Components Explained

| Component         | Purpose                         | Output                     |
| :---------------- | :------------------------------ | :------------------------- |
| `tok2vec`         | Token-to-vector encoding        | Vector representations     |
| `tagger`          | Part-of-speech tagging          | `token.pos_`, `token.tag_` |
| `parser`          | Dependency parsing              | `token.dep_`, `token.head` |
| `attribute_ruler` | Rule-based attribute assignment | Custom attributes          |
| `lemmatizer`      | Lemmatization                   | `token.lemma_`             |
| `ner`             | Named Entity Recognition        | `doc.ents`                 |

---

## 🛠️ Building a Complete Pipeline

### Basic Pipeline Usage

```python
import spacy

nlp = spacy.load("en_core_web_sm")

text = "Captain America ate 100$ of samosa. Then he said I can do this all day."

doc = nlp(text)

# Access all linguistic features
for token in doc:
    print(f"{token.text:15} | POS: {token.pos_:6} | Lemma: {token.lemma_}")
```

**Output:**

```
Captain         | POS: PROPN  | Lemma: Captain
America         | POS: PROPN  | Lemma: America
ate             | POS: VERB   | Lemma: eat
100             | POS: NUM    | Lemma: 100
$               | POS: SYM    | Lemma: $
of              | POS: ADP    | Lemma: of
samosa          | POS: NOUN   | Lemma: samosa
.               | POS: PUNCT  | Lemma: .
...
```

### Blank vs Pre-trained Models

```python
# Blank model - only tokenization
nlp_blank = spacy.blank("en")
doc_blank = nlp_blank("Captain America ate 100$ of samosa")

for token in doc_blank:
    print(f"{token.text:15} | POS: {token.pos_:6} | Lemma: {token.lemma_}")
```

**Output with blank model:**

```
Captain         | POS:        | Lemma: Captain
America         | POS:        | Lemma: America
ate             | POS:        | Lemma: ate
...
```

Notice: No POS tags or proper lemmas because the blank model lacks those components!

---

## 📝 Custom Pipeline Creation

### Adding Components Selectively

```python
import spacy

# Start with blank model
nlp = spacy.blank("en")

# Add only the NER component from a pre-trained model
source_nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("ner", source=source_nlp)

print(f"Components: {nlp.pipe_names}")
# Output: ['ner']

# Now we can do NER but not POS tagging
doc = nlp("Tesla Inc is going to acquire Twitter for $45 billion")

for ent in doc.ents:
    print(f"{ent.text} | {ent.label_}")
```

### Adding the Sentencizer

```python
nlp = spacy.blank("en")
nlp.add_pipe('sentencizer')

doc = nlp("Apple is great. Google is also great. Microsoft too.")

for sentence in doc.sents:
    print(sentence.text)
```

**Output:**

```
Apple is great.
Google is also great.
Microsoft too.
```

---

## 💻 Complete Preprocessing Function

### Production-Ready Pipeline

```python
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text,
                    remove_stopwords=True,
                    lemmatize=True,
                    remove_punct=True,
                    lowercase=True):
    """
    Complete text preprocessing pipeline.

    Args:
        text: Raw input text
        remove_stopwords: Remove common stop words
        lemmatize: Convert to base form
        remove_punct: Remove punctuation
        lowercase: Convert to lowercase

    Returns:
        Preprocessed text string
    """
    doc = nlp(text)

    tokens = []
    for token in doc:
        # Skip unwanted token types
        if remove_punct and token.is_punct:
            continue
        if token.is_space:
            continue
        if remove_stopwords and token.is_stop:
            continue

        # Get the appropriate form
        if lemmatize:
            word = token.lemma_
        else:
            word = token.text

        if lowercase:
            word = word.lower()

        tokens.append(word)

    return " ".join(tokens)

# Example usage
text = "The researchers were studying various complex studies about student learning habits."

print("Original:", text)
print("Processed:", preprocess_text(text))
```

**Output:**

```
Original: The researchers were studying various complex studies about student learning habits.
Processed: researcher study various complex study student learn habit
```

---

## 🔍 Extracting Multiple Features

### Comprehensive Text Analysis

```python
import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")

def analyze_text(text):
    """Extract comprehensive features from text."""
    doc = nlp(text)

    analysis = {
        'tokens': [token.text for token in doc],
        'lemmas': [token.lemma_ for token in doc if not token.is_punct],
        'pos_tags': [(token.text, token.pos_) for token in doc],
        'entities': [(ent.text, ent.label_) for ent in doc.ents],
        'noun_chunks': [chunk.text for chunk in doc.noun_chunks],
        'sentences': [sent.text for sent in doc.sents],
        'word_count': len([t for t in doc if not t.is_punct and not t.is_space]),
        'pos_distribution': Counter([token.pos_ for token in doc])
    }

    return analysis

# Example
text = """
Tesla Inc announced today that CEO Elon Musk will visit the
new Gigafactory in Austin, Texas on January 15, 2024.
The company expects to produce 500,000 vehicles this quarter.
"""

results = analyze_text(text)

print("=== Entities ===")
for entity, label in results['entities']:
    print(f"  {entity:20} → {label}")

print("\n=== Noun Chunks ===")
for chunk in results['noun_chunks']:
    print(f"  {chunk}")

print("\n=== POS Distribution ===")
for pos, count in results['pos_distribution'].most_common(5):
    print(f"  {pos}: {count}")
```

---

## 🌐 Multi-Language Pipeline

```python
import spacy

# English pipeline
nlp_en = spacy.load("en_core_web_sm")

# French pipeline
nlp_fr = spacy.load("fr_core_news_sm")

# Process text in both languages
en_text = "Tesla Inc is going to acquire Twitter for $45 billion"
fr_text = "Tesla Inc va racheter Twitter pour 45 milliards de dollars"

print("English NER:")
doc_en = nlp_en(en_text)
for ent in doc_en.ents:
    print(f"  {ent.text} | {ent.label_}")

print("\nFrench NER:")
doc_fr = nlp_fr(fr_text)
for ent in doc_fr.ents:
    print(f"  {ent.text} | {ent.label_}")
```

---

## ⚡ Pipeline Optimization

### Processing Large Volumes

```python
import spacy

nlp = spacy.load("en_core_web_sm")

# For large datasets, use nlp.pipe() for batch processing
texts = [
    "Apple announced new products.",
    "Google released an update.",
    "Microsoft acquired a startup.",
    # ... thousands more
]

# Efficient batch processing
for doc in nlp.pipe(texts, batch_size=50):
    # Process each document
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    print(entities)
```

### Disabling Unused Components

```python
# If you only need tokenization and NER, disable others for speed
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])

# Or use context manager for temporary disabling
with nlp.select_pipes(enable=["ner"]):
    doc = nlp("Tesla announced earnings today")
    print([(ent.text, ent.label_) for ent in doc.ents])
```

---

## 📊 Pipeline Visualization

```python
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Tesla Inc is going to acquire Twitter for $45 billion")

# Entity visualization
displacy.render(doc, style="ent")

# Dependency parsing visualization
displacy.render(doc, style="dep")
```

---

## 🎯 Complete Pipeline Example

### End-to-End Processing

```python
import spacy
import re
from spacy.lang.en.stop_words import STOP_WORDS

class NLPPipeline:
    def __init__(self, model="en_core_web_sm"):
        self.nlp = spacy.load(model)

    def clean_text(self, text):
        """Step 1: Basic cleaning"""
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def process(self, text):
        """Step 2: Full NLP processing"""
        cleaned = self.clean_text(text)
        return self.nlp(cleaned)

    def get_tokens(self, doc, remove_stops=True, lemmatize=True):
        """Step 3: Token extraction"""
        tokens = []
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            if remove_stops and token.is_stop:
                continue
            tokens.append(token.lemma_ if lemmatize else token.text)
        return tokens

    def get_entities(self, doc):
        """Step 4: Entity extraction"""
        return [(ent.text, ent.label_) for ent in doc.ents]

    def get_pos_tags(self, doc):
        """Step 5: POS tagging"""
        return [(token.text, token.pos_) for token in doc
                if not token.is_space]

    def full_analysis(self, text):
        """Complete analysis pipeline"""
        doc = self.process(text)
        return {
            'original': text,
            'cleaned_tokens': self.get_tokens(doc),
            'entities': self.get_entities(doc),
            'pos_tags': self.get_pos_tags(doc),
            'sentences': [sent.text for sent in doc.sents]
        }

# Usage
pipeline = NLPPipeline()

text = """
Check out https://example.com for details!
Apple CEO Tim Cook announced that the company will invest
$1 billion in Austin, Texas facilities by December 2024.
"""

results = pipeline.full_analysis(text)

print("Cleaned Tokens:", results['cleaned_tokens'][:10])
print("Entities:", results['entities'])
print("Sentences:", len(results['sentences']))
```

---

## 🎓 Best Practices

1. **Choose the right model**:

   - `sm`: Fast, good for most tasks
   - `md`: Better accuracy, includes word vectors
   - `lg`: Best accuracy, largest vectors

2. **Disable unused components** for speed

3. **Use `nlp.pipe()`** for batch processing

4. **Cache processed documents** when possible

5. **Profile your pipeline** to find bottlenecks

6. **Test with edge cases**: empty strings, very long texts, special characters

---

## 📚 Further Reading

- [spaCy Pipelines](https://spacy.io/usage/processing-pipelines)
- [spaCy Models](https://spacy.io/models)
- [Custom Pipeline Components](https://spacy.io/usage/processing-pipelines#custom-components)

---

## 🎉 Summary

You've now learned the complete text preprocessing pipeline:

1. ✂️ **Tokenization** - Breaking text into tokens
2. ⚖️ **Stemming/Lemmatization** - Normalizing words
3. 🚫 **Stop Words** - Filtering noise
4. 🏷️ **POS Tagging** - Grammatical analysis
5. 🎯 **NER** - Entity extraction
6. 🧪 **Regex** - Pattern matching
7. 🔄 **Pipeline** - Combining everything

Each technique has its place, and the best pipeline depends on your specific NLP task!
