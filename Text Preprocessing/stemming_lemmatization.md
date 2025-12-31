# Stemming & Lemmatization ⚖️

Stemming and Lemmatization are text normalization techniques that reduce words to their base or root form. This helps NLP models treat different forms of the same word (like "running", "runs", "ran") as a single token.

---

## 📖 Why Normalize Words?

Consider these sentences:

- "I am **eating** an apple"
- "She **eats** apples daily"
- "He **ate** the apple yesterday"

Without normalization, a model sees three different words. With normalization, they all map to "**eat**" — the core meaning.

**Benefits:**

- Reduces vocabulary size
- Improves model generalization
- Enhances text matching and search
- Reduces data sparsity

---

## 🔨 Stemming

**Stemming** is a rule-based, heuristic approach that removes word suffixes to produce a "stem."

### How It Works

Stemming applies fixed rules to chop off common suffixes:

```
running  → runn  (remove "ing")
studies  → studi (remove "es", change "y" to "i")
carefully → care (remove "fully")
```

### Stemming with NLTK

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

words = ["eating", "eats", "eat", "ate", "adjustable", "rafting", "ability", "meeting"]

for word in words:
    print(f"{word:12} → {stemmer.stem(word)}")
```

**Output:**

```
eating       → eat
eats         → eat
eat          → eat
ate          → ate
adjustable   → adjust
rafting      → raft
ability      → abil
meeting      → meet
```

### Popular Stemming Algorithms

| Algorithm             | Aggressiveness | Description                                  |
| :-------------------- | :------------- | :------------------------------------------- |
| **Porter Stemmer**    | Moderate       | Most widely used, good balance               |
| **Snowball Stemmer**  | Moderate       | Improved Porter, supports multiple languages |
| **Lancaster Stemmer** | Aggressive     | Produces shorter stems, may over-stem        |
| **Regexp Stemmer**    | Custom         | Uses regex patterns for stemming             |

### Stemmer Comparison

```python
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer

porter = PorterStemmer()
snowball = SnowballStemmer("english")
lancaster = LancasterStemmer()

word = "generalization"

print(f"Porter:    {porter.stem(word)}")      # generaliz
print(f"Snowball:  {snowball.stem(word)}")    # general
print(f"Lancaster: {lancaster.stem(word)}")   # gen
```

### ⚠️ Stemming Limitations

1. **Produces non-words**: "studies" → "studi" (not a real word)
2. **Over-stemming**: Different words get same stem ("universal" and "university" → "univers")
3. **Under-stemming**: Same root words get different stems ("alumnus", "alumni" may not match)
4. **No context awareness**: Can't distinguish "meeting" (noun) from "meeting" (verb)

---

## 🔬 Lemmatization

**Lemmatization** uses vocabulary and morphological analysis to return the proper base form of a word, called the **lemma**.

### How It Works

Lemmatization:

1. Looks up the word in a dictionary
2. Considers the word's part of speech
3. Returns the actual root word

```
running  → run   (verb)
better   → good  (adjective, irregular form)
mice     → mouse (noun, irregular plural)
```

### Lemmatization with spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")

doc = nlp("eating eats eat ate adjustable rafting ability meeting better")

for token in doc:
    print(f"{token.text:12} → {token.lemma_}")
```

**Output:**

```
eating       → eat
eats         → eat
eat          → eat
ate          → eat
adjustable   → adjustable
rafting      → raft
ability      → ability
meeting      → meeting
better       → good
```

### The Power of Context

spaCy's lemmatizer considers context:

```python
doc = nlp("Mando talked for 3 hours although talking isn't his thing")

for token in doc:
    print(f"{token.text:10} → {token.lemma_}")
```

**Output:**

```
Mando      → Mando
talked     → talk
for        → for
3          → 3
hours      → hour
although   → although
talking    → talk
is         → be
n't        → not
his        → his
thing      → thing
```

---

## 🎨 Customizing the Lemmatizer

Sometimes you need custom lemmatization rules for slang, abbreviations, or domain-specific terms:

```python
import spacy

nlp = spacy.load("en_core_web_sm")

# Access the attribute ruler
ar = nlp.get_pipe('attribute_ruler')

# Add custom lemma rules
ar.add([[{"TEXT": "Bro"}], [{"TEXT": "Brah"}]], {"LEMMA": "Brother"})

doc = nlp("Bro, you wanna go? Brah, don't say no!")

for token in doc:
    print(f"{token.text:6} → {token.lemma_}")
```

**Output:**

```
Bro    → Brother
,      → ,
you    → you
wanna  → wanna
go     → go
?      → ?
Brah   → Brother
...
```

---

## 📊 Stemming vs Lemmatization: Comparison

| Aspect                | Stemming                    | Lemmatization                       |
| :-------------------- | :-------------------------- | :---------------------------------- |
| **Method**            | Rule-based suffix stripping | Dictionary + morphological analysis |
| **Speed**             | ⚡ Faster                   | 🐢 Slower                           |
| **Output Quality**    | May produce non-words       | Always produces valid words         |
| **Context Awareness** | ❌ No                       | ✅ Yes (uses POS)                   |
| **Irregular Forms**   | ❌ Often fails              | ✅ Handles correctly                |
| **Dependencies**      | Minimal                     | Requires language model             |
| **Memory**            | Low                         | Higher (dictionary lookup)          |

### When to Use Each

| Use Case                  | Recommended   | Reason                             |
| :------------------------ | :------------ | :--------------------------------- |
| **Search Engines**        | Stemming      | Speed matters, exact words don't   |
| **Information Retrieval** | Stemming      | Broader matching is beneficial     |
| **Sentiment Analysis**    | Lemmatization | Context and meaning matter         |
| **Chatbots**              | Lemmatization | Natural responses needed           |
| **Machine Translation**   | Lemmatization | Grammatical accuracy required      |
| **Document Indexing**     | Stemming      | Fast processing of large volumes   |
| **Text Classification**   | Either        | Depends on dataset characteristics |

---

## 💻 Practical Example: Processing Pipeline

```python
import spacy
from nltk.stem import PorterStemmer

nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()

text = "The researchers are studying various studies about student studying habits"

# Stemming approach
words = text.split()
stemmed = [stemmer.stem(word) for word in words]
print("Stemmed:", stemmed)

# Lemmatization approach
doc = nlp(text)
lemmatized = [token.lemma_ for token in doc]
print("Lemmatized:", lemmatized)
```

**Output:**

```
Stemmed:    ['the', 'research', 'are', 'studi', 'variou', 'studi', 'about', 'student', 'studi', 'habit']
Lemmatized: ['the', 'researcher', 'be', 'study', 'various', 'study', 'about', 'student', 'study', 'habit']
```

---

## 🎓 Best Practices

1. **Choose based on your task**:

   - Speed-critical applications → Stemming
   - Accuracy-critical applications → Lemmatization

2. **Consider your language**:

   - English: Both work well
   - Highly inflected languages (German, Russian): Lemmatization preferred

3. **Test both approaches**:

   - Run experiments to see which works better for your specific dataset

4. **Combine with other preprocessing**:

   - Apply after tokenization
   - Apply before or after stop word removal (test both)

5. **Handle edge cases**:
   - Proper nouns might need special handling
   - Domain-specific terms may require custom rules

---

## ⚠️ Common Pitfalls

| Pitfall                      | Description                              | Solution              |
| :--------------------------- | :--------------------------------------- | :-------------------- |
| **Named Entity Destruction** | "Microsoft" → "microsoft"                | Skip proper nouns     |
| **Semantic Loss**            | "better" and "best" → "good"             | May be intentional    |
| **Over-processing**          | Applying both stemming AND lemmatization | Choose one            |
| **Language Mismatch**        | Using English stemmer on French text     | Use appropriate tools |

---

## 📚 Further Reading

- [NLTK Stemming Documentation](https://www.nltk.org/howto/stem.html)
- [spaCy Lemmatization](https://spacy.io/usage/linguistic-features#lemmatization)
- [Porter Stemmer Algorithm](https://tartarus.org/martin/PorterStemmer/)

---

## ▶️ Next Steps

After normalization, proceed to:

- [Stop Words Removal](stop_words.md) - Filter out common words
- [Part-of-Speech Tagging](pos.md) - Label words with grammatical categories
