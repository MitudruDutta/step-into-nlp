# Tokenization in NLP ✂️

Tokenization is the **first and most fundamental step** in any Natural Language Processing pipeline. It breaks down raw text into smaller, manageable pieces called **tokens** that can be processed by machine learning models.

---

## 📖 What is Tokenization?

**Tokenization** is the process of splitting a stream of text into individual units (tokens) such as words, sentences, subwords, or characters. These tokens serve as the basic building blocks for all downstream NLP tasks.

```
Input:  "The quick brown fox jumps over 13 lazy dogs."
Output: ["The", "quick", "brown", "fox", "jumps", "over", "13", "lazy", "dogs", "."]
```

---

## 🎯 Why is Tokenization Important?

1. **Machine Learning Requirement**: Models can't process raw text directly—they need numerical representations derived from tokens
2. **Vocabulary Building**: Creates a finite set of unique tokens for the model to learn
3. **Feature Extraction**: Enables bag-of-words, TF-IDF, and embedding generation
4. **Downstream Tasks**: All NLP tasks (NER, POS tagging, sentiment analysis) depend on proper tokenization

---

## 📊 Types of Tokenization

### 1. Word Tokenization

Splits text by whitespace and punctuation marks.

```python
import spacy
nlp = spacy.blank("en")

doc = nlp("The quick brown fox jumps over 13 lazy dogs.")
tokens = [token.text for token in doc]
# Output: ['The', 'quick', 'brown', 'fox', 'jumps', 'over', '13', 'lazy', 'dogs', '.']
```

**Pros:**

- Simple and intuitive
- Fast processing

**Cons:**

- Out-of-Vocabulary (OOV) problems with rare words
- Struggles with compound words and contractions

---

### 2. Sentence Tokenization

Divides text into complete sentences.

```python
nlp = spacy.blank("en")
nlp.add_pipe('sentencizer')

doc = nlp("Apple is looking at buying U.K. startup for $1 billion. Hydra is a dragon of India")
for sentence in doc.sents:
    print(sentence)
# Output:
# Apple is looking at buying U.K. startup for $1 billion.
# Hydra is a dragon of India
```

**Use Cases:**

- Document summarization
- Machine translation
- Question answering systems

---

### 3. Character Tokenization

Breaks text into individual characters.

```
Input:  "Hello"
Output: ["H", "e", "l", "l", "o"]
```

**Pros:**

- No OOV problems—any word can be represented
- Useful for morphologically rich languages

**Cons:**

- Very long sequences
- Harder for models to learn word-level semantics

---

### 4. Subword Tokenization

Splits rare words into frequent subword units while keeping common words intact.

```
Input:  "unhappiness"
Output: ["un", "happi", "ness"]

Input:  "the"
Output: ["the"]  (common word stays whole)
```

**Algorithms:**
| Algorithm | Used By | Description |
|:----------|:--------|:------------|
| **BPE** (Byte Pair Encoding) | GPT-2, GPT-3, RoBERTa | Iteratively merges most frequent character pairs |
| **WordPiece** | BERT, DistilBERT | Similar to BPE but uses likelihood instead of frequency |
| **SentencePiece** | T5, ALBERT, XLNet | Language-agnostic, works on raw text |
| **Unigram** | Used with SentencePiece | Probabilistic model selecting most likely tokenization |

---

## 🛠️ Tokenization with spaCy

### Basic Tokenization

```python
import spacy

# Create a blank English tokenizer (fast, no extra processing)
nlp = spacy.blank("en")

text = "I gave three $ to Peter."
doc = nlp(text)

for token in doc:
    print(f"Token: {token.text:10} | Index: {token.i}")
```

### Token Attributes

spaCy provides powerful built-in attributes for each token:

```python
doc = nlp("I gave 100$ to Peter.")

for token in doc:
    print(f"{token.text:8} | is_alpha: {token.is_alpha} | is_punct: {token.is_punct} | like_num: {token.like_num} | is_currency: {token.is_currency}")
```

| Attribute           | Description                         | Example    |
| :------------------ | :---------------------------------- | :--------- |
| `token.text`        | The token string                    | "Hello"    |
| `token.i`           | Token index in document             | 0, 1, 2... |
| `token.is_alpha`    | Contains only alphabetic characters | True/False |
| `token.is_punct`    | Is punctuation                      | True/False |
| `token.like_num`    | Looks like a number                 | True/False |
| `token.is_currency` | Is a currency symbol                | True/False |
| `token.like_email`  | Looks like an email                 | True/False |
| `token.is_stop`     | Is a stop word                      | True/False |

---

## 🔧 Practical Example: Extracting Emails

```python
nlp = spacy.blank("en")

with open("students.txt") as f:
    text = ' '.join(f.readlines())

doc = nlp(text)

emails = [token.text for token in doc if token.like_email]
print(emails)
# Output: ['john@example.com', 'jane@university.edu', ...]
```

---

## 🌐 Multi-language Tokenization

spaCy supports tokenization for many languages:

```python
# Hindi tokenization
nlp_hi = spacy.blank("hi")
doc = nlp_hi("राम ने सीता को एक पत्र लिखा।")

for token in doc:
    print(token.text)
```

---

## ⚙️ Custom Tokenization Rules

Sometimes you need to handle special cases like slang or domain-specific terms:

```python
from spacy.symbols import ORTH

nlp = spacy.blank("en")

# Add custom tokenization rule: "gimme" → "gim" + "me"
nlp.tokenizer.add_special_case("gimme", [{ORTH: "gim"}, {ORTH: "me"}])

doc = nlp("gimme double cheese extra large pizza")
tokens = [token.text for token in doc]
# Output: ['gim', 'me', 'double', 'cheese', 'extra', 'large', 'pizza']
```

---

## 📈 Tokenization Challenges

| Challenge            | Description                        | Solution                      |
| :------------------- | :--------------------------------- | :---------------------------- |
| **Contractions**     | "don't" → "do" + "n't" or "don't"? | Use language-specific rules   |
| **Hyphenated words** | "well-known" → one or two tokens?  | Depends on context            |
| **URLs and emails**  | Keep intact or split?              | Use special token detection   |
| **Emojis**           | 😀 → separate token?               | Modern tokenizers handle this |
| **Code-switching**   | "That's très cool"                 | Use multilingual models       |

---

## 🎓 Best Practices

1. **Choose the right tokenizer** for your task:

   - Word tokenization for traditional NLP
   - Subword for modern transformer models

2. **Handle edge cases** explicitly:

   - URLs, emails, phone numbers
   - Domain-specific abbreviations

3. **Consider your language**:

   - Some languages (Chinese, Japanese) don't use spaces
   - Morphologically rich languages benefit from subword tokenization

4. **Preserve important information**:
   - Case sensitivity may matter
   - Punctuation might be significant

---

## 📚 Further Reading

- [spaCy Tokenization Documentation](https://spacy.io/usage/linguistic-features#tokenization)
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/)
- [BPE Paper](https://arxiv.org/abs/1508.07909)
- [WordPiece Paper](https://arxiv.org/abs/1609.08144)

---

## ▶️ Next Steps

After tokenization, proceed to:

- [Stemming & Lemmatization](stemming_lemmatization.md) - Normalize tokens to their root forms
- [Stop Words Removal](stop_words.md) - Filter out low-information tokens
