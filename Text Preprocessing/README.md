# Text Preprocessing in NLP 📝

Text preprocessing is the **foundational step** in any Natural Language Processing (NLP) pipeline. Raw text data is inherently messy, unstructured, and filled with noise. Before any machine learning model can extract meaningful insights, the text must be cleaned, normalized, and transformed into a format that algorithms can process effectively.

This module provides a comprehensive, hands-on exploration of essential text preprocessing techniques using Python's most powerful NLP libraries.

---

## � Directory Structure

```
Text Preprocessing/
├── README.md                          # This file
├── docs/                              # 📖 Documentation guides
│   ├── tokenization.md
│   ├── stemming_lemmatization.md
│   ├── stop_words.md
│   ├── pos.md
│   ├── ner.md
│   ├── regex.md
│   └── pipeline.md
├── notebooks/                         # 💻 Jupyter notebooks
│   ├── tokenization.ipynb
│   ├── stemming_lemmatization.ipynb
│   ├── stop_words.ipynb
│   ├── pos.ipynb
│   ├── ner.ipynb
│   ├── regex.ipynb
│   └── pipeline.ipynb
└── data/                              # 📊 Sample datasets
    ├── doj_press.json
    ├── news_story.txt
    └── students.txt
```

---

## 📚 Table of Contents

| Topic                        | Documentation                              | Notebook                                          | Description                                       |
| :--------------------------- | :----------------------------------------- | :------------------------------------------------ | :------------------------------------------------ |
| **Tokenization**             | [📖 Guide](docs/tokenization.md)           | [💻 Code](notebooks/tokenization.ipynb)           | Breaking text into words, sentences, and subwords |
| **Stemming & Lemmatization** | [📖 Guide](docs/stemming_lemmatization.md) | [💻 Code](notebooks/stemming_lemmatization.ipynb) | Reducing words to their root forms                |
| **Stop Words Removal**       | [📖 Guide](docs/stop_words.md)             | [💻 Code](notebooks/stop_words.ipynb)             | Filtering out common, low-information words       |
| **Part-of-Speech Tagging**   | [📖 Guide](docs/pos.md)                    | [💻 Code](notebooks/pos.ipynb)                    | Labeling words with grammatical categories        |
| **Named Entity Recognition** | [📖 Guide](docs/ner.md)                    | [💻 Code](notebooks/ner.ipynb)                    | Identifying and classifying named entities        |
| **Regular Expressions**      | [📖 Guide](docs/regex.md)                  | [💻 Code](notebooks/regex.ipynb)                  | Pattern matching for text extraction              |
| **Complete Pipeline**        | [📖 Guide](docs/pipeline.md)               | [💻 Code](notebooks/pipeline.ipynb)               | End-to-end text preprocessing workflow            |

---

## 1. Tokenization ✂️

**Tokenization** is the process of splitting text into smaller, meaningful units called **tokens**. It serves as the entry point for all NLP tasks.

### Types of Tokenization

| Type                       | Description                               | Example                                  | Use Case                                    |
| :------------------------- | :---------------------------------------- | :--------------------------------------- | :------------------------------------------ |
| **Word Tokenization**      | Splits text by whitespace and punctuation | "Hello world!" → ["Hello", "world", "!"] | Traditional NLP, bag-of-words models        |
| **Sentence Tokenization**  | Divides text into sentences               | Paragraph → List of sentences            | Document summarization, translation         |
| **Character Tokenization** | Breaks text into individual characters    | "cat" → ["c", "a", "t"]                  | Character-level models, spelling correction |
| **Subword Tokenization**   | Splits words into frequent subunits       | "unhappiness" → ["un", "happi", "ness"]  | Modern transformers (BERT, GPT)             |

### Why Subword Tokenization Matters

Modern language models like BERT and GPT use subword tokenization (BPE, WordPiece, SentencePiece) to:

- Handle **out-of-vocabulary (OOV)** words gracefully
- Balance vocabulary size with sequence length
- Capture morphological patterns in language

---

## 2. Stemming & Lemmatization ⚖️

Both techniques reduce words to their base or root form to treat variations of a word as the same token.

### Stemming

A **rule-based, heuristic** approach that chops off word suffixes.

```
"running", "runs", "ran" → "run"
"studies", "studying" → "studi" (may produce non-words!)
```

**Algorithms:** Porter Stemmer, Snowball Stemmer, Lancaster Stemmer

### Lemmatization

A **dictionary-based, linguistic** approach that returns the actual root word (lemma).

```
"running", "runs", "ran" → "run"
"better" → "good" (understands irregular forms)
```

### Comparison Table

| Aspect            | Stemming                  | Lemmatization                             |
| :---------------- | :------------------------ | :---------------------------------------- |
| **Approach**      | Rule-based suffix removal | Dictionary + morphological analysis       |
| **Speed**         | Faster                    | Slower                                    |
| **Output**        | May produce non-words     | Always produces valid words               |
| **Context-Aware** | No                        | Yes (considers POS)                       |
| **Best For**      | Search engines, indexing  | Chatbots, sentiment analysis, translation |

---

## 3. Stop Words Removal 🚫

**Stop words** are high-frequency words (e.g., "the", "is", "at", "and") that carry minimal semantic meaning but appear frequently in text.

### Why Remove Stop Words?

- **Reduces noise** in the data
- **Decreases dimensionality** of feature space
- **Improves model performance** by focusing on meaningful content
- **Speeds up processing** time

### When to Keep Stop Words

- **Sentiment Analysis:** "not good" vs "good" — negation matters!
- **Question Answering:** "What is..." — question words are important
- **Machine Translation:** Grammatical structure depends on all words

### Custom Stop Words

You can extend or modify default stop word lists based on your domain:

```python
# Adding domain-specific stop words
custom_stops = nlp.Defaults.stop_words.union({"said", "according"})
```

---

## 4. Part-of-Speech (POS) Tagging 🏷️

**POS tagging** assigns grammatical labels to each token in a sentence, revealing its syntactic role.

### Common POS Tags

| Tag     | Meaning     | Example          |
| :------ | :---------- | :--------------- |
| `NOUN`  | Noun        | cat, city, idea  |
| `VERB`  | Verb        | run, think, is   |
| `ADJ`   | Adjective   | beautiful, quick |
| `ADV`   | Adverb      | quickly, very    |
| `PROPN` | Proper Noun | Google, Paris    |
| `DET`   | Determiner  | the, a, this     |
| `PRON`  | Pronoun     | he, she, it      |
| `ADP`   | Adposition  | in, on, at       |
| `CONJ`  | Conjunction | and, but, or     |

### Applications of POS Tagging

- **Information Extraction:** Extract nouns as key entities
- **Word Sense Disambiguation:** "bank" (river bank vs. financial bank)
- **Grammar Checking:** Identify incorrect word usage
- **Text-to-Speech:** Proper pronunciation based on word type

---

## 5. Named Entity Recognition (NER) 🎯

**NER** identifies and classifies named entities in text into predefined categories such as persons, organizations, locations, dates, and more.

### Common Entity Types

| Entity       | Label     | Example                       |
| :----------- | :-------- | :---------------------------- |
| Person       | `PERSON`  | "Elon Musk", "Dr. Smith"      |
| Organization | `ORG`     | "Google", "United Nations"    |
| Location     | `GPE`     | "New York", "France"          |
| Date         | `DATE`    | "January 2024", "next Monday" |
| Money        | `MONEY`   | "$500", "fifty dollars"       |
| Time         | `TIME`    | "3:00 PM", "noon"             |
| Percentage   | `PERCENT` | "25%", "fifty percent"        |

### Real-World Applications

- **News Analysis:** Extract people, organizations, and locations from articles
- **Resume Parsing:** Identify names, companies, skills, and dates
- **Customer Support:** Detect product names, dates, and order numbers
- **Legal Documents:** Extract parties, dates, and monetary amounts

---

## 6. Regular Expressions (Regex) 🧪

**Regular expressions** are powerful patterns for matching, searching, and extracting structured data from unstructured text.

### Common Regex Patterns

| Pattern             | Matches            | Example                |
| :------------------ | :----------------- | :--------------------- |
| `\d+`               | One or more digits | "123", "42"            |
| `\w+`               | Word characters    | "hello", "test123"     |
| `\s+`               | Whitespace         | spaces, tabs, newlines |
| `[A-Z][a-z]+`       | Capitalized words  | "Hello", "Python"      |
| `\b\w+@\w+\.\w+\b`  | Email addresses    | "user@example.com"     |
| `\d{3}-\d{3}-\d{4}` | Phone numbers      | "555-123-4567"         |

### When to Use Regex in NLP

- **Data Cleaning:** Remove URLs, HTML tags, special characters
- **Entity Extraction:** Phone numbers, emails, dates, IDs
- **Pattern Matching:** Hashtags, mentions, specific formats
- **Text Validation:** Verify input formats before processing

---

## 7. The Complete NLP Pipeline 🔄

A typical text preprocessing pipeline combines multiple techniques in sequence:

```
Raw Text
    ↓
1. Tokenization (split into tokens)
    ↓
2. Lowercasing (optional, task-dependent)
    ↓
3. Stop Words Removal (filter noise)
    ↓
4. Stemming/Lemmatization (normalize words)
    ↓
5. POS Tagging (grammatical analysis)
    ↓
6. NER (entity extraction)
    ↓
Clean, Structured Data
```

---

## 🛠️ Tools & Libraries Used

| Library   | Purpose                                            |
| :-------- | :------------------------------------------------- |
| **spaCy** | Industrial-strength NLP with pre-trained pipelines |
| **NLTK**  | Classic NLP library with extensive resources       |
| **re**    | Python's built-in regex module                     |

### spaCy Models

```bash
# Install the small English model
python -m spacy download en_core_web_sm

# For better accuracy, use the medium or large model
python -m spacy download en_core_web_md
python -m spacy download en_core_web_lg
```

---

## 💻 Quick Start Example

```python
import spacy

# Load the English NLP pipeline
nlp = spacy.load("en_core_web_sm")

text = "Apple Inc. was founded by Steve Jobs in California on April 1, 1976."

# Process the text
doc = nlp(text)

# Tokenization & POS Tagging
print("Token Analysis:")
for token in doc:
    print(f"  {token.text:12} | POS: {token.pos_:6} | Lemma: {token.lemma_}")

# Named Entity Recognition
print("\nNamed Entities:")
for ent in doc.ents:
    print(f"  {ent.text:20} → {ent.label_}")
```

**Output:**

```
Token Analysis:
  Apple        | POS: PROPN  | Lemma: Apple
  Inc.         | POS: PROPN  | Lemma: Inc.
  was          | POS: AUX    | Lemma: be
  founded      | POS: VERB   | Lemma: found
  ...

Named Entities:
  Apple Inc.           → ORG
  Steve Jobs           → PERSON
  California           → GPE
  April 1, 1976        → DATE
```

---

## 📁 Data Files

This module includes sample data files for practice:

| File             | Description                                        |
| :--------------- | :------------------------------------------------- |
| `news_story.txt` | Sample news article for text processing            |
| `doj_press.json` | Department of Justice press releases (JSON format) |
| `students.txt`   | Sample student data for regex extraction           |

---

## 🎯 Learning Objectives

By completing this module, you will be able to:

1. **Tokenize** text using different strategies (word, sentence, subword)
2. **Normalize** text using stemming and lemmatization
3. **Filter** stop words while understanding when to keep them
4. **Tag** parts of speech and use them for downstream tasks
5. **Extract** named entities from unstructured text
6. **Write** regex patterns for custom text extraction
7. **Build** end-to-end preprocessing pipelines

---

## 📖 Further Reading

- [spaCy Documentation](https://spacy.io/usage)
- [NLTK Book](https://www.nltk.org/book/)
- [Regular Expressions 101](https://regex101.com/)
- [Stanford NLP Course](https://web.stanford.edu/class/cs224n/)
