# Stop Words Removal 🚫

Stop words are common, high-frequency words that typically carry little semantic meaning in text analysis. Understanding when and how to remove them is crucial for effective NLP preprocessing.

---

## 📖 What Are Stop Words?

**Stop words** are words that appear frequently in a language but contribute minimal meaning to the content. They serve grammatical purposes rather than conveying information.

### Examples of Stop Words

| Category            | Examples                         |
| :------------------ | :------------------------------- |
| **Articles**        | a, an, the                       |
| **Prepositions**    | in, on, at, to, for, with        |
| **Conjunctions**    | and, but, or, if, while          |
| **Pronouns**        | I, you, he, she, it, we, they    |
| **Auxiliary Verbs** | is, am, are, was, were, be, been |
| **Common Adverbs**  | very, really, just, also         |

---

## 🎯 Why Remove Stop Words?

### Benefits

1. **Reduces Noise**: Focuses on content-bearing words
2. **Decreases Dimensionality**: Smaller vocabulary = faster models
3. **Improves Performance**: Better accuracy in many classification tasks
4. **Saves Resources**: Less memory and processing time
5. **Enhances Interpretability**: Topic models show meaningful words

### Example Impact

```
Original:    "The quick brown fox jumps over the lazy dog"
After:       "quick brown fox jumps lazy dog"

Original:    "This is a very good movie and I really liked it"
After:       "good movie liked"
```

---

## 🛠️ Stop Words in spaCy

### Viewing the Stop Word List

```python
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# Check how many stop words spaCy knows
print(f"Number of stop words: {len(STOP_WORDS)}")
# Output: 326

# View some stop words
print(list(STOP_WORDS)[:20])
```

### Identifying Stop Words in Text

```python
nlp = spacy.load("en_core_web_sm")

doc = nlp("We just opened our wings, the flying part is coming soon")

print("Stop words found:")
for token in doc:
    if token.is_stop:
        print(f"  '{token.text}'")
```

**Output:**

```
Stop words found:
  'We'
  'just'
  'our'
  'the'
  'is'
  'soon'
```

---

## 💻 Implementing Stop Word Removal

### Basic Function

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def remove_stop_words(text):
    doc = nlp(text)
    filtered = [token.text for token in doc if not token.is_stop]
    return " ".join(filtered)

# Example usage
text = "Musk wants time to prepare for a trial over his acquisition"
clean_text = remove_stop_words(text)
print(clean_text)
# Output: "Musk wants time prepare trial acquisition"
```

### With Additional Cleaning

```python
def preprocess(text):
    doc = nlp(text)
    # Remove stop words and punctuation, convert to lowercase
    tokens = [
        token.text.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and token.text.strip()
    ]
    return " ".join(tokens)
```

---

## 📊 Processing DataFrames

### Real-World Example: DOJ Press Releases

```python
import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    doc = nlp(text)
    return " ".join([token.text for token in doc if not token.is_stop])

# Load data
df = pd.read_json("doj_press.json", lines=True)

# Filter rows with topics
df = df[df["topics"].str.len() != 0]

# Apply preprocessing
df["contents_clean"] = df["contents"].apply(preprocess)

# Compare lengths
original_length = len(df["contents"].iloc[0])
clean_length = len(df["contents_clean"].iloc[0])

print(f"Original: {original_length} characters")
print(f"After stop word removal: {clean_length} characters")
print(f"Reduction: {((original_length - clean_length) / original_length * 100):.1f}%")
```

---

## ⚙️ Customizing Stop Words

### Adding Custom Stop Words

```python
import spacy

nlp = spacy.load("en_core_web_sm")

# Add domain-specific stop words
custom_stops = {"said", "according", "reported", "stated", "announced"}
nlp.Defaults.stop_words |= custom_stops  # Union operation

# Or using update
nlp.Defaults.stop_words.update(custom_stops)
```

### Removing Words from Stop List

```python
# Keep certain words that are usually stop words
words_to_keep = {"not", "no", "never", "without"}

for word in words_to_keep:
    nlp.Defaults.stop_words.discard(word)
```

### Creating a Fully Custom List

```python
# Define your own stop word list
my_stop_words = {"the", "a", "an", "is", "are", "was", "were"}

def remove_custom_stops(text, stop_words):
    words = text.split()
    return " ".join([w for w in words if w.lower() not in stop_words])
```

---

## ⚠️ When NOT to Remove Stop Words

Stop word removal can **harm** performance in these scenarios:

### 1. Sentiment Analysis

```python
text1 = "this is a good movie"
text2 = "this is not a good movie"

print(remove_stop_words(text1))  # "good movie"
print(remove_stop_words(text2))  # "good movie" ← Lost the negation!
```

**Problem**: Removing "not" changes the meaning completely!

### 2. Question Answering / Chatbots

```python
question = "How are you doing today?"
print(remove_stop_words(question))  # "today?" ← Lost the question context!
```

**Problem**: Question words (how, what, when, where, why) are often stop words.

### 3. Machine Translation

```python
text = "I am going to the store"
print(remove_stop_words(text))  # "going store"
```

**Problem**: Grammatical structure is destroyed, making translation impossible.

### 4. Named Entity Recognition

Some stop words provide context for entity recognition:

- "The University of California" → "University California"
- "Bank of America" → "Bank America"

### 5. Language Models

Modern transformers (BERT, GPT) learn from ALL words, including stop words. They capture context and relationships that depend on these words.

---

## 📈 When TO Remove Stop Words

| Task                             | Remove Stop Words? | Reason                     |
| :------------------------------- | :----------------- | :------------------------- |
| **Topic Modeling**               | ✅ Yes             | Focus on content words     |
| **Keyword Extraction**           | ✅ Yes             | Extract meaningful terms   |
| **Document Classification**      | 🤔 Maybe           | Test both approaches       |
| **Search/Information Retrieval** | ✅ Yes             | Match on content words     |
| **Bag-of-Words Models**          | ✅ Yes             | Reduce dimensionality      |
| **TF-IDF Analysis**              | ✅ Yes             | Focus on distinctive words |
| **Sentiment Analysis**           | ❌ No              | Negation words matter      |
| **Machine Translation**          | ❌ No              | Need complete grammar      |
| **Question Answering**           | ❌ No              | Question words important   |
| **Text Generation**              | ❌ No              | Need fluent output         |

---

## 🔄 Stop Words in Different Languages

spaCy provides stop word lists for many languages:

```python
from spacy.lang.de.stop_words import STOP_WORDS as DE_STOPS
from spacy.lang.fr.stop_words import STOP_WORDS as FR_STOPS
from spacy.lang.es.stop_words import STOP_WORDS as ES_STOPS

print(f"German stop words: {len(DE_STOPS)}")
print(f"French stop words: {len(FR_STOPS)}")
print(f"Spanish stop words: {len(ES_STOPS)}")
```

---

## 🎓 Best Practices

1. **Always test with and without** stop word removal
   - Measure impact on your specific task
2. **Be conservative initially**

   - Start with standard stop words
   - Add domain-specific ones based on data analysis

3. **Consider your model**

   - Traditional ML (Naive Bayes, SVM): Often benefits from removal
   - Deep Learning (BERT, transformers): Usually keep all words

4. **Examine your stop word list**

   - Review what's being removed
   - Check for task-relevant words

5. **Document your choices**
   - Record which stop words were used/modified
   - Enables reproducibility

---

## 💡 Pro Tips

### Check Stop Word Impact

```python
from collections import Counter

doc = nlp(text)

stop_count = sum(1 for token in doc if token.is_stop)
total_count = len(doc)

print(f"Stop words: {stop_count}/{total_count} ({stop_count/total_count*100:.1f}%)")
```

### Visualize Most Common Stop Words

```python
stop_tokens = [token.text.lower() for token in doc if token.is_stop]
common_stops = Counter(stop_tokens).most_common(10)

for word, count in common_stops:
    print(f"{word}: {count}")
```

---

## 📚 Further Reading

- [spaCy Stop Words](https://spacy.io/usage/rule-based-matching#adding-patterns)
- [NLTK Stop Words](https://www.nltk.org/book/ch02.html)
- [When Stop Word Removal Helps](https://www.sciencedirect.com/science/article/pii/S0957417417308345)

---

## ▶️ Next Steps

After stop word removal, proceed to:

- [Part-of-Speech Tagging](pos.md) - Understand grammatical roles
- [Named Entity Recognition](ner.md) - Extract named entities
