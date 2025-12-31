# Named Entity Recognition (NER) 🎯

Named Entity Recognition (NER) is the task of identifying and classifying named entities in text into predefined categories such as person names, organizations, locations, dates, monetary values, and more.

---

## 📖 What is Named Entity Recognition?

**NER** automatically detects and categorizes specific entities mentioned in text. It answers questions like:

- **Who** is mentioned? (People)
- **What organization** is involved? (Companies, institutions)
- **Where** did it happen? (Locations)
- **When** did it occur? (Dates, times)
- **How much** was involved? (Money, quantities)

```
Input:  "Tesla Inc is going to acquire Twitter for $45 billion"

Output:
  - Tesla Inc    → ORG (Organization)
  - Twitter      → ORG (Organization)
  - $45 billion  → MONEY (Monetary value)
```

---

## 🎯 Why is NER Important?

1. **Information Extraction**: Pull structured data from unstructured text
2. **Knowledge Graph Construction**: Build entity relationships
3. **Search Enhancement**: Improve search relevance
4. **Content Recommendation**: Understand article topics
5. **Customer Support**: Identify products, dates, order numbers
6. **Legal/Compliance**: Extract parties, amounts, dates from contracts

---

## 📊 Common Entity Types

### Standard spaCy Entity Labels

| Entity                                   | Label         | Description                                 | Examples                                |
| :--------------------------------------- | :------------ | :------------------------------------------ | :-------------------------------------- |
| Person                                   | `PERSON`      | Names of people                             | "Elon Musk", "Dr. Smith"                |
| Organization                             | `ORG`         | Companies, agencies, institutions           | "Google", "FBI", "Harvard"              |
| Geopolitical Entity                      | `GPE`         | Countries, cities, states                   | "France", "New York", "California"      |
| Location                                 | `LOC`         | Non-GPE locations                           | "Mount Everest", "Pacific Ocean"        |
| Date                                     | `DATE`        | Absolute or relative dates                  | "January 2024", "tomorrow", "next week" |
| Time                                     | `TIME`        | Times                                       | "3:00 PM", "noon", "midnight"           |
| Money                                    | `MONEY`       | Monetary values                             | "$500", "45 billion dollars"            |
| Percentage                               | `PERCENT`     | Percentages                                 | "25%", "fifty percent"                  |
| Quantity                                 | `QUANTITY`    | Measurements                                | "100 kg", "five miles"                  |
| Cardinal                                 | `CARDINAL`    | Numbers that are not covered by other types | "one", "100", "millions"                |
| Ordinal                                  | `ORDINAL`     | Ordinal numbers                             | "first", "2nd", "third"                 |
| Product                                  | `PRODUCT`     | Products, vehicles, etc.                    | "iPhone", "Boeing 747"                  |
| Event                                    | `EVENT`       | Named events                                | "World War II", "Olympics"              |
| Work of Art                              | `WORK_OF_ART` | Titles of creative works                    | "Mona Lisa", "The Matrix"               |
| Law                                      | `LAW`         | Named laws, regulations                     | "GDPR", "First Amendment"               |
| Language                                 | `LANGUAGE`    | Named languages                             | "English", "French"                     |
| Facility                                 | `FAC`         | Buildings, airports, etc.                   | "JFK Airport", "Eiffel Tower"           |
| Nationalities/Religious/Political Groups | `NORP`        | Group identities                            | "American", "Buddhist", "Democrat"      |

---

## 🛠️ NER with spaCy

### Basic NER

```python
import spacy

nlp = spacy.load("en_core_web_sm")

doc = nlp("Tesla Inc is going to acquire Twitter for $45 billion")

for ent in doc.ents:
    print(f"{ent.text:15} | {ent.label_:10} | {spacy.explain(ent.label_)}")
```

**Output:**

```
Tesla Inc       | ORG        | Companies, agencies, institutions, etc.
Twitter         | ORG        | Companies, agencies, institutions, etc.
$45 billion     | MONEY      | Monetary values, including unit
```

### Viewing All Available Entity Types

```python
# List all entity labels the model can recognize
print(nlp.pipe_labels['ner'])
```

**Output:**

```
['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC',
 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT',
 'QUANTITY', 'TIME', 'WORK_OF_ART']
```

---

## 🎨 Visualizing Entities

spaCy provides a built-in visualizer:

```python
from spacy import displacy

doc = nlp("Tesla Inc is going to acquire Twitter for $45 billion")

# Render in Jupyter notebook
displacy.render(doc, style="ent")

# Or save to HTML file
html = displacy.render(doc, style="ent", page=True)
with open("entities.html", "w") as f:
    f.write(html)
```

This produces a colorful visualization highlighting each entity in the text.

---

## 📍 Entity Positions

Get the character positions of entities:

```python
doc = nlp("Tesla Inc is going to acquire Twitter Inc for $45 billion")

for ent in doc.ents:
    print(f"{ent.text:15} | {ent.label_:8} | Start: {ent.start_char:3} | End: {ent.end_char}")
```

**Output:**

```
Tesla Inc       | ORG      | Start:   0 | End: 9
Twitter Inc     | ORG      | Start:  29 | End: 40
$45 billion     | MONEY    | Start:  45 | End: 56
```

---

## ⚙️ Setting Custom Entities

Sometimes the model misses entities or you need domain-specific ones:

```python
from spacy.tokens import Span

doc = nlp("Tesla is going to acquire Twitter for $45 billion")

# Check what was detected
for ent in doc.ents:
    print(f"{ent.text} | {ent.label_}")
# Output: $45 billion | MONEY (Tesla and Twitter might be missed!)

# Manually add entities
s1 = Span(doc, 0, 1, label="ORG")   # "Tesla"
s2 = Span(doc, 5, 6, label="ORG")   # "Twitter"

# Set entities (keep existing ones with "unmodified")
doc.set_ents([s1, s2], default="unmodified")

# Verify
for ent in doc.ents:
    print(f"{ent.text} | {ent.label_}")
```

**Output:**

```
Tesla | ORG
Twitter | ORG
$45 billion | MONEY
```

---

## 🌍 Multi-Language NER

spaCy models support NER in multiple languages:

```python
# French NER
nlp_fr = spacy.load("fr_core_news_sm")

doc = nlp_fr("Tesla Inc va racheter Twitter pour 45 milliards de dollars")

for ent in doc.ents:
    print(f"{ent.text} | {ent.label_} | {spacy.explain(ent.label_)}")
```

---

## ⚠️ NER Challenges and Limitations

### Common Issues

| Challenge              | Description                    | Example                         |
| :--------------------- | :----------------------------- | :------------------------------ |
| **Ambiguity**          | Same name, different types     | "Apple" (company vs. fruit)     |
| **Unknown Entities**   | New or rare names              | Startups, local places          |
| **Nested Entities**    | Entities within entities       | "Bank of America Tower"         |
| **Coreference**        | Pronouns referring to entities | "He" referring to "Elon Musk"   |
| **Context Dependency** | Meaning changes with context   | "Washington" (person vs. place) |

### Real Example: Bloomberg Confusion

```python
doc = nlp("Michael Bloomberg founded Bloomberg Inc in 1982")

for ent in doc.ents:
    print(f"{ent.text} | {ent.label_}")
```

**Output (may be incorrect):**

```
Michael Bloomberg | PERSON
Bloomberg Inc     | PERSON  ← Should be ORG!
1982              | DATE
```

**Solution**: Use larger models or specialized models:

- `en_core_web_md` or `en_core_web_lg` for better accuracy
- Hugging Face transformers (BERT-based NER) for state-of-the-art results

---

## 🚀 Using Transformer-Based NER

For higher accuracy, consider Hugging Face models:

```python
# Using transformers (example)
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER")

text = "Michael Bloomberg founded Bloomberg Inc in 1982"
results = ner(text)

for result in results:
    print(f"{result['word']} | {result['entity']} | {result['score']:.3f}")
```

---

## 💻 Practical Applications

### 1. News Article Analysis

```python
news = """
Apple announced today that CEO Tim Cook will visit the new
Apple Park facility in Cupertino, California next Monday.
The visit coincides with the launch of the iPhone 15,
which is expected to generate $100 billion in revenue.
"""

doc = nlp(news)

# Organize entities by type
entities_by_type = {}
for ent in doc.ents:
    if ent.label_ not in entities_by_type:
        entities_by_type[ent.label_] = []
    entities_by_type[ent.label_].append(ent.text)

for label, entities in entities_by_type.items():
    print(f"{label}: {entities}")
```

### 2. Resume Parsing

```python
resume = """
John Smith
Software Engineer at Google (2019-Present)
Previously at Microsoft (2015-2019)
Stanford University, Computer Science, 2015
Skills: Python, TensorFlow, AWS
Email: john.smith@gmail.com
"""

doc = nlp(resume)

print("Entities found:")
for ent in doc.ents:
    print(f"  {ent.text:25} → {ent.label_}")
```

### 3. Contract Analysis

```python
contract = """
This agreement between Acme Corporation and XYZ Industries,
dated January 15, 2024, involves a payment of $5,000,000
for services to be rendered over a period of 24 months.
"""

doc = nlp(contract)

# Extract key contract information
parties = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
amounts = [ent.text for ent in doc.ents if ent.label_ == "MONEY"]

print(f"Parties: {parties}")
print(f"Dates: {dates}")
print(f"Amounts: {amounts}")
```

---

## 🎓 Best Practices

1. **Choose the right model size**:

   - `sm` (small): Fast, lower accuracy
   - `md` (medium): Balanced
   - `lg` (large): Best accuracy, slower

2. **Validate entity types** for your domain:

   ```python
   # Check if expected entities are detected
   expected_orgs = ["Apple", "Google", "Microsoft"]
   detected_orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
   ```

3. **Combine with rule-based matching** for domain-specific entities

4. **Consider fine-tuning** for specialized domains (medical, legal, financial)

5. **Post-process results** to merge split entities or filter false positives

---

## 📚 Further Reading

- [spaCy NER Documentation](https://spacy.io/usage/linguistic-features#named-entities)
- [spaCy Entity Types](https://spacy.io/models/en)
- [Hugging Face NER Models](https://huggingface.co/models?pipeline_tag=token-classification)
- [OntoNotes 5 Entity Guidelines](https://catalog.ldc.upenn.edu/docs/LDC2013T19/OntoNotes-Release-5.0.pdf)

---

## ▶️ Next Steps

After NER, proceed to:

- [Regular Expressions](regex.md) - Pattern-based extraction
- [Complete Pipeline](pipeline.md) - Combine all techniques
