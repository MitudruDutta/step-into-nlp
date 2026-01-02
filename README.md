# 🗣️ Step Into NLP

A structured, hands-on learning repository for mastering **Natural Language Processing** from fundamentals to production-ready applications. From tokenization to transformers, this project provides comprehensive documentation and practical implementations.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![spaCy](https://img.shields.io/badge/spaCy-3.5+-09A3D5.svg)](https://spacy.io)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 What You'll Learn

| Module                          | Topics                                            | Status         |
| ------------------------------- | ------------------------------------------------- | -------------- |
| Introduction to NLP             | NLP foundations, pipeline, tools overview         | ✅ Complete    |
| Text Preprocessing              | Tokenization, stemming, lemmatization, NER, regex | ✅ Complete    |
| Text Representation             | BoW, TF-IDF, Word2Vec, GloVe, embeddings          | ✅ Complete    |
| Practical NLP with Hugging Face | Transformers, fine-tuning, real-world apps        | ⏳ Coming Soon |

---

## 📁 Repository Structure

```
step-into-nlp/
│
├── 📘 Introduction to NLP/
│   ├── README.md                  # Module overview
│   ├── what-is-nlp.md             # What is NLP, history, applications
│   ├── nlp-pipeline.md            # End-to-end NLP pipeline guide
│   ├── nlp-tools.md               # spaCy vs NLTK vs Hugging Face vs Gensim
│   └── spacyvsnltk.ipynb          # 📓 Practical: spaCy vs NLTK comparison
│
├── 📗 Text Preprocessing/
│   ├── README.md                  # Module overview with quick reference
│   ├── docs/                      # 📖 Documentation guides
│   │   ├── tokenization.md
│   │   ├── stemming_lemmatization.md
│   │   ├── stop_words.md
│   │   ├── pos.md
│   │   ├── ner.md
│   │   ├── regex.md
│   │   └── pipeline.md
│   ├── notebooks/                 # 📓 Jupyter notebooks
│   │   ├── tokenization.ipynb
│   │   ├── stemming_lemmatization.ipynb
│   │   ├── stop_words.ipynb
│   │   ├── pos.ipynb
│   │   ├── ner.ipynb
│   │   ├── regex.ipynb
│   │   └── pipeline.ipynb
│   └── data/                      # 📊 Sample datasets
│       ├── doj_press.json
│       ├── news_story.txt
│       └── students.txt
│
├── 📙 Text Representation/
│   ├── README.md                  # Module overview
│   ├── docs/                      # 📖 Documentation guides
│   │   ├── bag_of_words.md
│   │   ├── ngrams.md
│   │   ├── tfidf.md
│   │   ├── word_embeddings.md
│   │   └── text_classification.md
│   ├── notebooks/                 # 📓 Jupyter notebooks
│   │   ├── bag_of_words.ipynb
│   │   ├── 10_bag_of_n_grams.ipynb
│   │   ├── tf_idf.ipynb
│   │   ├── spacy_word_vectors.ipynb
│   │   └── text_classification.ipynb
│   └── data/                      # 📊 Datasets
│       ├── Ecommerce_data.csv
│       ├── Emotion_classify_Data.csv
│       ├── Fake_Real_Data.csv
│       ├── movies_sentiment_data.csv
│       ├── news_dataset.json
│       └── spam.csv
│
├── README.md                      # You are here
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── LICENSE                        # MIT License
```

---

## 🛤️ Learning Path

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Introduction   │     │     Text        │     │      Text      │
│   to NLP ✅     │ ──► │  Preprocessing  │ ──► │ Representation │
│  (Foundations)  │     │       ✅        │     │       ✅       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                        ┌───────────────────────────────┘
                        ▼
                ┌─────────────────┐
                │  Practical NLP  │
                │  with Hugging   │
                │     Face 🤗     │
                │  (Coming Soon)  │
                └─────────────────┘
```

### Quick Start

→ Start with `Introduction to NLP/what-is-nlp.md`

---

## 📓 Notebooks

| Notebook                                                                                    | Module              | What You'll Learn                       |
| ------------------------------------------------------------------------------------------- | ------------------- | --------------------------------------- |
| [spacyvsnltk.ipynb](Introduction%20to%20NLP/spacyvsnltk.ipynb)                              | Introduction to NLP | Compare spaCy and NLTK tokenization     |
| [tokenization.ipynb](Text%20Preprocessing/notebooks/tokenization.ipynb)                     | Text Preprocessing  | Word, sentence, subword tokenization    |
| [stemming_lemmatization.ipynb](Text%20Preprocessing/notebooks/stemming_lemmatization.ipynb) | Text Preprocessing  | Stemming vs lemmatization techniques    |
| [stop_words.ipynb](Text%20Preprocessing/notebooks/stop_words.ipynb)                         | Text Preprocessing  | Stop word removal with pandas           |
| [pos.ipynb](Text%20Preprocessing/notebooks/pos.ipynb)                                       | Text Preprocessing  | Part-of-speech tagging with spaCy       |
| [ner.ipynb](Text%20Preprocessing/notebooks/ner.ipynb)                                       | Text Preprocessing  | Named Entity Recognition                |
| [regex.ipynb](Text%20Preprocessing/notebooks/regex.ipynb)                                   | Text Preprocessing  | Regular expressions for text extraction |
| [pipeline.ipynb](Text%20Preprocessing/notebooks/pipeline.ipynb)                             | Text Preprocessing  | Complete NLP preprocessing pipeline     |
| [bag_of_words.ipynb](Text%20Representation/notebooks/bag_of_words.ipynb)                    | Text Representation | Bag of Words implementation             |
| [10_bag_of_n_grams.ipynb](Text%20Representation/notebooks/10_bag_of_n_grams.ipynb)          | Text Representation | N-grams for capturing word sequences    |
| [tf_idf.ipynb](Text%20Representation/notebooks/tf_idf.ipynb)                                | Text Representation | TF-IDF vectorization                    |
| [spacy_word_vectors.ipynb](Text%20Representation/notebooks/spacy_word_vectors.ipynb)        | Text Representation | Word embeddings with spaCy              |
| [text_classification.ipynb](Text%20Representation/notebooks/text_classification.ipynb)      | Text Representation | End-to-end text classification          |

---

## 📚 Module: Introduction to NLP ✅

Foundational concepts for understanding NLP:

| File                                                           | Description                                              |
| -------------------------------------------------------------- | -------------------------------------------------------- |
| [README.md](Introduction%20to%20NLP/README.md)                 | Module overview and quick reference                      |
| [what-is-nlp.md](Introduction%20to%20NLP/what-is-nlp.md)       | What is NLP, its importance, applications, and history   |
| [nlp-pipeline.md](Introduction%20to%20NLP/nlp-pipeline.md)     | Complete guide to building NLP application pipelines     |
| [nlp-tools.md](Introduction%20to%20NLP/nlp-tools.md)           | In-depth comparison of Hugging Face, spaCy, NLTK, Gensim |
| [spacyvsnltk.ipynb](Introduction%20to%20NLP/spacyvsnltk.ipynb) | Practical notebook comparing spaCy and NLTK              |

**Key Topics:**

- **What is NLP?** — Definition, history, and real-world applications
- **NLP Pipeline** — Data acquisition → preprocessing → modeling → deployment
- **Tool Comparison** — When to use spaCy, NLTK, Gensim, or Hugging Face
- **Hands-on** — Practical comparison of spaCy vs NLTK

---

## � Module: Text Preprocessing ✅

Comprehensive text preprocessing techniques for NLP:

| File                                                                                  | Description                                      |
| ------------------------------------------------------------------------------------- | ------------------------------------------------ |
| [README.md](Text%20Preprocessing/README.md)                                           | Module overview and quick reference              |
| [docs/tokenization.md](Text%20Preprocessing/docs/tokenization.md)                     | Word, sentence, character & subword tokenization |
| [docs/stemming_lemmatization.md](Text%20Preprocessing/docs/stemming_lemmatization.md) | Stemming algorithms & spaCy lemmatization        |
| [docs/stop_words.md](Text%20Preprocessing/docs/stop_words.md)                         | Stop word removal strategies & when to keep them |
| [docs/pos.md](Text%20Preprocessing/docs/pos.md)                                       | Part-of-Speech tagging with fine-grained tags    |
| [docs/ner.md](Text%20Preprocessing/docs/ner.md)                                       | Named Entity Recognition & custom entities       |
| [docs/regex.md](Text%20Preprocessing/docs/regex.md)                                   | Regular expressions for text extraction          |
| [docs/pipeline.md](Text%20Preprocessing/docs/pipeline.md)                             | Complete end-to-end preprocessing pipeline       |

**Key Topics:**

- **Tokenization** — Word, sentence, subword splitting with spaCy
- **Normalization** — Stemming (NLTK) vs Lemmatization (spaCy)
- **Stop Words** — Removal strategies & when NOT to remove
- **POS Tagging** — Grammatical analysis & filtering
- **NER** — Entity extraction, visualization, custom entities
- **Regex** — Pattern matching for emails, phones, dates
- **Pipeline** — Combining all techniques efficiently

---

## 📙 Module: Text Representation ✅

Converting text into numerical representations for machine learning:

| File                                                                             | Description                               |
| -------------------------------------------------------------------------------- | ----------------------------------------- |
| [README.md](Text%20Representation/README.md)                                     | Module overview and quick reference       |
| [docs/bag_of_words.md](Text%20Representation/docs/bag_of_words.md)               | Count-based text representation           |
| [docs/ngrams.md](Text%20Representation/docs/ngrams.md)                           | Capturing word sequences with N-grams     |
| [docs/tfidf.md](Text%20Representation/docs/tfidf.md)                             | Term frequency-inverse document frequency |
| [docs/word_embeddings.md](Text%20Representation/docs/word_embeddings.md)         | Dense vector representations              |
| [docs/text_classification.md](Text%20Representation/docs/text_classification.md) | End-to-end classification pipeline        |

**Key Topics:**

- **Bag of Words** — Count-based vectorization with scikit-learn
- **N-Grams** — Capturing word context and sequences
- **TF-IDF** — Weighing term importance across documents
- **Word Embeddings** — Word2Vec, GloVe, spaCy vectors
- **Text Classification** — Complete ML pipeline with real datasets

---

## �🛠️ Setup

### Prerequisites

- Python 3.8+

### Installation

```bash
# Clone the repository
git clone https://github.com/MitudruDutta/step-into-nlp.git
cd step-into-nlp

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Verify Installation

```python
import spacy
import nltk

print(f"spaCy: {spacy.__version__}")
print(f"NLTK: {nltk.__version__}")

# Quick test
nlp = spacy.load("en_core_web_sm")
doc = nlp("NLP is amazing!")
print(f"Tokens: {[token.text for token in doc]}")
```

---

## 🛠️ Technologies Used

| Category          | Tools                       |
| ----------------- | --------------------------- |
| **Language**      | Python 3.8+                 |
| **Classical NLP** | spaCy, NLTK, Gensim         |
| **Deep Learning** | Hugging Face Transformers   |
| **Data Science**  | NumPy, Pandas, Scikit-learn |
| **Visualization** | Matplotlib, Seaborn         |
| **Notebooks**     | Jupyter                     |

---

## 📖 Recommended Resources

### Courses

- [Hugging Face NLP Course](https://huggingface.co/course) — Free, comprehensive
- [Stanford CS224N](https://web.stanford.edu/class/cs224n/) — NLP with Deep Learning
- [fast.ai NLP](https://www.fast.ai/) — Practical approach

### Books

- _Speech and Language Processing_ — Jurafsky & Martin
- _Natural Language Processing with Transformers_ — Tunstall et al.
- _Natural Language Processing with Python_ — NLTK Book (free online)

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Open issues for bugs or suggestions
- Submit PRs to improve documentation
- Add new topics or notebooks

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🔗 Connect

- **GitHub:** [MitudruDutta](https://github.com/MitudruDutta)
- **Repository:** [step-into-nlp](https://github.com/MitudruDutta/step-into-nlp)

---

<p align="center">
  <i>Language is the road map of a culture. Let's teach machines to read it.</i> 🗣️
</p>
