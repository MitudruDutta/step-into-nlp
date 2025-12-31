# Part-of-Speech (POS) Tagging 🏷️

Part-of-Speech tagging is the process of assigning grammatical categories (noun, verb, adjective, etc.) to each word in a sentence. It's a fundamental NLP task that enables deeper linguistic analysis.

---

## 📖 What is POS Tagging?

**POS tagging** labels each token in a sentence with its grammatical category, revealing the syntactic structure and role of every word.

```
Input:  "Elon flew to mars yesterday"
Output: "Elon/PROPN flew/VERB to/ADP mars/PROPN yesterday/ADV"
```

---

## 🎯 Why is POS Tagging Important?

1. **Word Sense Disambiguation**: "bank" as a noun (river bank) vs. verb (bank on something)
2. **Information Extraction**: Extract all nouns as potential entities
3. **Syntax Analysis**: Understand sentence structure
4. **Text-to-Speech**: Proper pronunciation depends on word type
5. **Machine Translation**: Preserve grammatical structure across languages
6. **Grammar Checking**: Identify incorrect word usage

---

## 📊 Common POS Tags

### Universal POS Tags (spaCy)

| Tag     | Name                      | Description                 | Examples                   |
| :------ | :------------------------ | :-------------------------- | :------------------------- |
| `ADJ`   | Adjective                 | Describes nouns             | big, old, green, beautiful |
| `ADP`   | Adposition                | Preposition or postposition | in, on, at, to, from       |
| `ADV`   | Adverb                    | Modifies verbs/adjectives   | very, quickly, yesterday   |
| `AUX`   | Auxiliary                 | Helping verbs               | is, has, will, should, can |
| `CONJ`  | Conjunction               | Connects words/clauses      | and, but, or, if           |
| `DET`   | Determiner                | Articles, demonstratives    | the, a, this, that, my     |
| `INTJ`  | Interjection              | Exclamation                 | wow, oops, hello           |
| `NOUN`  | Noun                      | Person, place, thing        | cat, city, idea            |
| `NUM`   | Numeral                   | Number                      | one, 2, third, 100         |
| `PART`  | Particle                  | Function word               | not, 's, to (infinitive)   |
| `PRON`  | Pronoun                   | Replaces noun               | I, you, he, she, it        |
| `PROPN` | Proper Noun               | Specific names              | Google, Paris, John        |
| `PUNCT` | Punctuation               | Punctuation marks           | . , ! ? "                  |
| `SCONJ` | Subordinating Conjunction | Introduces clauses          | if, because, that          |
| `SYM`   | Symbol                    | Symbols                     | $, %, @                    |
| `VERB`  | Verb                      | Action or state             | run, eat, is, think        |
| `X`     | Other                     | Foreign words, typos        | asdf, xyzzy                |
| `SPACE` | Space                     | Whitespace                  | (spaces, tabs)             |

---

## 🛠️ POS Tagging with spaCy

### Basic POS Tagging

```python
import spacy

nlp = spacy.load("en_core_web_sm")

doc = nlp("Elon flew to mars yesterday. He carried biryani masala with him")

for token in doc:
    print(f"{token.text:10} | {token.pos_:6} | {spacy.explain(token.pos_)}")
```

**Output:**

```
Elon       | PROPN  | proper noun
flew       | VERB   | verb
to         | ADP    | adposition
mars       | PROPN  | proper noun
yesterday  | ADV    | adverb
.          | PUNCT  | punctuation
He         | PRON   | pronoun
carried    | VERB   | verb
biryani    | PROPN  | proper noun
masala     | NOUN   | noun
with       | ADP    | adposition
him        | PRON   | pronoun
```

### Coarse vs. Fine-Grained Tags

spaCy provides two levels of POS tags:

```python
doc = nlp("Wow! Dr. Strange made 265 million $ on the very first day")

for token in doc:
    print(f"{token.text:10} | {token.pos_:6} | {token.tag_:6} | {spacy.explain(token.tag_)}")
```

**Output:**

```
Wow        | INTJ   | UH     | interjection
!          | PUNCT  | .      | punctuation mark, sentence closer
Dr.        | PROPN  | NNP    | noun, proper singular
Strange    | PROPN  | NNP    | noun, proper singular
made       | VERB   | VBD    | verb, past tense
265        | NUM    | CD     | cardinal number
million    | NUM    | CD     | cardinal number
$          | SYM    | $      | symbol, currency
on         | ADP    | IN     | conjunction, subordinating or preposition
the        | DET    | DT     | determiner
very       | ADV    | RB     | adverb
first      | ADJ    | JJ     | adjective (English)
day        | NOUN   | NN     | noun, singular or mass
```

| Level                  | Attribute    | Example Tags      |
| :--------------------- | :----------- | :---------------- |
| **Coarse** (Universal) | `token.pos_` | NOUN, VERB, ADJ   |
| **Fine-grained**       | `token.tag_` | NN, NNS, VBD, VBG |

---

## 🎯 Understanding Tense with Fine-Grained Tags

spaCy can distinguish between verb tenses:

```python
# Present tense
doc = nlp("He quits the job")
print(f"{doc[1].text} | {doc[1].tag_} | {spacy.explain(doc[1].tag_)}")
# Output: quits | VBZ | verb, 3rd person singular present

# Past tense
doc = nlp("He quit the job")
print(f"{doc[1].text} | {doc[1].tag_} | {spacy.explain(doc[1].tag_)}")
# Output: quit | VBD | verb, past tense
```

---

## 💻 Practical Applications

### 1. Filtering by POS Tags

Extract only specific word types:

```python
doc = nlp("The quick brown fox jumps over the lazy dog")

# Get only nouns
nouns = [token.text for token in doc if token.pos_ == "NOUN"]
print(f"Nouns: {nouns}")  # ['fox', 'dog']

# Get only adjectives
adjectives = [token.text for token in doc if token.pos_ == "ADJ"]
print(f"Adjectives: {adjectives}")  # ['quick', 'brown', 'lazy']

# Get only verbs
verbs = [token.text for token in doc if token.pos_ == "VERB"]
print(f"Verbs: {verbs}")  # ['jumps']
```

### 2. Cleaning Text with POS

Remove spaces, punctuation, and unknown tokens:

```python
earnings_text = """Microsoft Corp. today announced the following results:
·   Revenue was $51.7 billion and increased 20%
·   Operating income was $22.2 billion
"""

doc = nlp(earnings_text)

# Remove SPACE, PUNCT, and X (unknown) tokens
filtered_tokens = [token for token in doc if token.pos_ not in ["SPACE", "PUNCT", "X"]]

print([t.text for t in filtered_tokens[:10]])
# Output: ['Microsoft', 'Corp.', 'today', 'announced', 'the', 'following', 'results', 'Revenue', 'was', '$']
```

### 3. Counting POS Distribution

```python
doc = nlp("The researchers are studying various studies about student habits")

# Count occurrences of each POS
count = doc.count_by(spacy.attrs.POS)

for pos_id, freq in count.items():
    pos_name = doc.vocab[pos_id].text
    print(f"{pos_name}: {freq}")
```

**Output:**

```
DET: 1
NOUN: 4
AUX: 1
VERB: 1
ADJ: 1
ADP: 1
```

### 4. Extracting Noun Phrases

```python
doc = nlp("The quick brown fox jumps over the lazy dog in the park")

# Get noun chunks (phrases where nouns are the head)
for chunk in doc.noun_chunks:
    print(f"'{chunk.text}' | Root: {chunk.root.text} | Root POS: {chunk.root.pos_}")
```

**Output:**

```
'The quick brown fox' | Root: fox | Root POS: NOUN
'the lazy dog' | Root: dog | Root POS: NOUN
'the park' | Root: park | Root POS: NOUN
```

---

## 📊 Fine-Grained Tag Reference

### Verb Tags

| Tag   | Description                       | Example         |
| :---- | :-------------------------------- | :-------------- |
| `VB`  | Verb, base form                   | eat, run        |
| `VBD` | Verb, past tense                  | ate, ran        |
| `VBG` | Verb, gerund/present participle   | eating, running |
| `VBN` | Verb, past participle             | eaten, run      |
| `VBP` | Verb, non-3rd person present      | eat, run        |
| `VBZ` | Verb, 3rd person singular present | eats, runs      |

### Noun Tags

| Tag    | Description           | Example       |
| :----- | :-------------------- | :------------ |
| `NN`   | Noun, singular        | cat, idea     |
| `NNS`  | Noun, plural          | cats, ideas   |
| `NNP`  | Proper noun, singular | Google, Paris |
| `NNPS` | Proper noun, plural   | Americans     |

### Adjective Tags

| Tag   | Description            | Example           |
| :---- | :--------------------- | :---------------- |
| `JJ`  | Adjective              | big, green        |
| `JJR` | Adjective, comparative | bigger, greener   |
| `JJS` | Adjective, superlative | biggest, greenest |

---

## 🔗 POS Tagging Resources

- [spaCy Annotation Specifications](https://v2.spacy.io/api/annotation)
- [Universal Dependencies POS Tags](https://universaldependencies.org/u/pos/)
- [Penn Treebank Tags](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)

---

## 🎓 Best Practices

1. **Use the right model**: Larger models (md, lg) have better accuracy

   ```python
   nlp = spacy.load("en_core_web_lg")  # Better accuracy than en_core_web_sm
   ```

2. **Consider context**: The same word can have different POS tags

   ```python
   doc1 = nlp("I need to book a flight")    # book = VERB
   doc2 = nlp("I'm reading a book")         # book = NOUN
   ```

3. **Handle unknown words**: Check for `X` tags in your output

4. **Combine with other features**: POS + NER + dependencies = richer analysis

---

## ⚠️ Common Challenges

| Challenge                  | Description                    | Solution                 |
| :------------------------- | :----------------------------- | :----------------------- |
| **Ambiguity**              | "lead" (noun) vs "lead" (verb) | Context-aware models     |
| **Unknown words**          | Slang, typos, domain terms     | Custom training or rules |
| **Multi-word expressions** | "New York" as single entity    | Use NER alongside POS    |
| **Code-switching**         | "That's muy bueno"             | Multilingual models      |

---

## 📚 Further Reading

- [spaCy Linguistic Features](https://spacy.io/usage/linguistic-features)
- [Part of Speech (Wikipedia)](https://en.wikipedia.org/wiki/Part_of_speech)
- [Penn Treebank Tagset](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)

---

## ▶️ Next Steps

After understanding POS tagging, proceed to:

- [Named Entity Recognition](ner.md) - Extract specific entities
- [Complete Pipeline](pipeline.md) - Build end-to-end preprocessing
