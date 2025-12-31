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
| Text Representation             | BoW, TF-IDF, Word2Vec, GloVe, embeddings          | ⏳ Coming Soon |
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
│   ├── tokenization.md            # Tokenization concepts & techniques
│   ├── stemming_lemmatization.md  # Stemming vs Lemmatization guide
│   ├── stop_words.md              # Stop words removal strategies
│   ├── pos.md                     # Part-of-Speech tagging
│   ├── ner.md                     # Named Entity Recognition
│   ├── regex.md                   # Regular expressions for NLP
│   ├── pipeline.md                # Complete preprocessing pipeline
│   ├── tokenization.ipynb         # 📓 Tokenization hands-on
│   ├── stemming_lemmatization.ipynb # 📓 Stemming & lemmatization
│   ├── stop_words.ipynb           # 📓 Stop words removal
│   ├── pos.ipynb                  # 📓 POS tagging
│   ├── ner.ipynb                  # 📓 Named Entity Recognition
│   ├── regex.ipynb                # 📓 Regex patterns
│   └── pipeline.ipynb             # 📓 Complete pipeline
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
│  (Foundations)  │     │       ✅        │     │  (Coming Soon) │
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

| Notebook                                                                          | Module              | What You'll Learn                       |
| --------------------------------------------------------------------------------- | ------------------- | --------------------------------------- |
| [spacyvsnltk.ipynb](Introduction%20to%20NLP/spacyvsnltk.ipynb)                    | Introduction to NLP | Compare spaCy and NLTK tokenization     |
| [tokenization.ipynb](Text%20Preprocessing/tokenization.ipynb)                     | Text Preprocessing  | Word, sentence, subword tokenization    |
| [stemming_lemmatization.ipynb](Text%20Preprocessing/stemming_lemmatization.ipynb) | Text Preprocessing  | Stemming vs lemmatization techniques    |
| [stop_words.ipynb](Text%20Preprocessing/stop_words.ipynb)                         | Text Preprocessing  | Stop word removal with pandas           |
| [pos.ipynb](Text%20Preprocessing/pos.ipynb)                                       | Text Preprocessing  | Part-of-speech tagging with spaCy       |
| [ner.ipynb](Text%20Preprocessing/ner.ipynb)                                       | Text Preprocessing  | Named Entity Recognition                |
| [regex.ipynb](Text%20Preprocessing/regex.ipynb)                                   | Text Preprocessing  | Regular expressions for text extraction |
| [pipeline.ipynb](Text%20Preprocessing/pipeline.ipynb)                             | Text Preprocessing  | Complete NLP preprocessing pipeline     |

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

| File                                                                        | Description                                      |
| --------------------------------------------------------------------------- | ------------------------------------------------ |
| [README.md](Text%20Preprocessing/README.md)                                 | Module overview and quick reference              |
| [tokenization.md](Text%20Preprocessing/tokenization.md)                     | Word, sentence, character & subword tokenization |
| [stemming_lemmatization.md](Text%20Preprocessing/stemming_lemmatization.md) | Stemming algorithms & spaCy lemmatization        |
| [stop_words.md](Text%20Preprocessing/stop_words.md)                         | Stop word removal strategies & when to keep them |
| [pos.md](Text%20Preprocessing/pos.md)                                       | Part-of-Speech tagging with fine-grained tags    |
| [ner.md](Text%20Preprocessing/ner.md)                                       | Named Entity Recognition & custom entities       |
| [regex.md](Text%20Preprocessing/regex.md)                                   | Regular expressions for text extraction          |
| [pipeline.md](Text%20Preprocessing/pipeline.md)                             | Complete end-to-end preprocessing pipeline       |

**Key Topics:**

- **Tokenization** — Word, sentence, subword splitting with spaCy
- **Normalization** — Stemming (NLTK) vs Lemmatization (spaCy)
- **Stop Words** — Removal strategies & when NOT to remove
- **POS Tagging** — Grammatical analysis & filtering
- **NER** — Entity extraction, visualization, custom entities
- **Regex** — Pattern matching for emails, phones, dates
- **Pipeline** — Combining all techniques efficiently

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
