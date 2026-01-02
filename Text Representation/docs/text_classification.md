# Text Classification - Complete Guide

## Overview

Text classification is one of the most practical and widely-used applications of Natural Language Processing. It involves automatically assigning predefined categories or labels to text documents based on their content. This module brings together all the text representation techniques we've learned and applies them to build real-world classification systems.

---

## Table of Contents

1. [What is Text Classification](#what-is-text-classification)
2. [Classification Pipeline](#classification-pipeline)
3. [Feature Extraction Methods](#feature-extraction-methods)
4. [Classification Algorithms](#classification-algorithms)
5. [Complete Project: E-commerce Classification](#complete-project-e-commerce-classification)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Best Practices](#best-practices)
8. [Common Pitfalls](#common-pitfalls)
9. [Advanced Techniques](#advanced-techniques)

---

## What is Text Classification

Text classification (also called text categorization) is a supervised learning task where we train a model to assign labels to text documents.

### Types of Classification

| Type            | Description                         | Example                                  |
| :-------------- | :---------------------------------- | :--------------------------------------- |
| **Binary**      | Two classes                         | Spam vs Ham, Positive vs Negative        |
| **Multi-class** | Multiple mutually exclusive classes | News categories (Sports, Politics, Tech) |
| **Multi-label** | Multiple non-exclusive labels       | Movie tags (Action, Comedy, Romance)     |

### Real-World Applications

| Application              | Description                                   |
| :----------------------- | :-------------------------------------------- |
| **Spam Detection**       | Filter unwanted emails/messages               |
| **Sentiment Analysis**   | Determine opinion (positive/negative/neutral) |
| **Topic Classification** | Categorize news articles, documents           |
| **Intent Detection**     | Understand user intent in chatbots            |
| **Language Detection**   | Identify text language                        |
| **Fake News Detection**  | Identify misleading content                   |
| **Customer Support**     | Route tickets to appropriate departments      |
| **Content Moderation**   | Flag inappropriate content                    |

---

## Classification Pipeline

A typical text classification pipeline consists of these stages:

```
┌──────────────────────────────────────────────────────────────────┐
│                    TEXT CLASSIFICATION PIPELINE                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Raw Text  →  Preprocessing  →  Vectorization  →  Model  →  Label │
│                                                                   │
│  "Great      lowercase         [0.2, 0.5,      Naive     "Positive"│
│   movie!"    remove punct       0.1, ...]      Bayes               │
│              tokenize                                             │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Stage 1: Data Collection

```python
import pandas as pd

# Load your dataset
df = pd.read_csv("data.csv")

# Examine the data
print(df.head())
print(f"Total samples: {len(df)}")
print(f"Class distribution:\n{df['label'].value_counts()}")
```

### Stage 2: Text Preprocessing

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    # Lowercase
    text = text.lower()

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize
    tokens = text.split()

    # Remove stop words and stem
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]

    return ' '.join(tokens)

# Apply preprocessing
df['clean_text'] = df['text'].apply(preprocess_text)
```

### Stage 3: Feature Extraction (Vectorization)

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(df['clean_text'])
y = df['label']
```

### Stage 4: Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Maintain class distribution
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
```

### Stage 5: Model Training

```python
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)
```

### Stage 6: Evaluation

```python
from sklearn.metrics import classification_report, accuracy_score

predictions = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
print(classification_report(y_test, predictions))
```

---

## Feature Extraction Methods

### Method 1: Bag of Words

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.95,
    stop_words='english'
)

X_bow = vectorizer.fit_transform(texts)
```

**Best for:** Simple classification tasks, short texts

### Method 2: TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_tfidf = vectorizer.fit_transform(texts)
```

**Best for:** Document classification, search relevance

### Method 3: Word Embeddings Average

```python
import spacy
import numpy as np

nlp = spacy.load("en_core_web_lg")

def get_document_vector(text):
    doc = nlp(text)
    vectors = [token.vector for token in doc if token.has_vector]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(300)

X_embed = np.array([get_document_vector(text) for text in texts])
```

**Best for:** Semantic similarity, when training data is limited

### Comparison

| Method     | Dimensionality | Captures Semantics | Speed  |
| :--------- | :------------- | :----------------- | :----- |
| BoW        | High           | No                 | Fast   |
| TF-IDF     | High           | No                 | Fast   |
| Embeddings | Low (300)      | Yes                | Slower |

---

## Classification Algorithms

### 1. Naive Bayes

Probabilistic classifier based on Bayes' theorem with "naive" independence assumptions.

```python
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB(alpha=1.0)  # alpha is smoothing parameter
model.fit(X_train, y_train)
```

**Pros:** Fast, works well with small data, interpretable
**Cons:** Assumes feature independence
**Best for:** Text classification, spam detection

### 2. Support Vector Machine (SVM)

Finds the hyperplane that best separates classes.

```python
from sklearn.svm import LinearSVC

model = LinearSVC(C=1.0, max_iter=1000)
model.fit(X_train, y_train)
```

**Pros:** Effective in high dimensions, robust to overfitting
**Cons:** Slower to train, less interpretable
**Best for:** Medium-sized datasets, binary classification

### 3. Logistic Regression

Linear model for classification despite the name.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    class_weight='balanced'  # Handle imbalanced data
)
model.fit(X_train, y_train)
```

**Pros:** Fast, interpretable, probability outputs
**Cons:** Assumes linear separability
**Best for:** Binary classification, interpretability needed

### 4. Random Forest

Ensemble of decision trees.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    n_jobs=-1
)
model.fit(X_train, y_train)
```

**Pros:** Handles non-linearity, feature importance
**Cons:** Slower, can overfit
**Best for:** Complex patterns, feature analysis

### 5. Gradient Boosting

```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)
model.fit(X_train, y_train)
```

**Pros:** Often best accuracy
**Cons:** Slow, prone to overfitting
**Best for:** When accuracy is critical

---

## Complete Project: E-commerce Classification

### Problem Statement

Classify product descriptions into 4 categories:

- Electronics
- Household
- Books
- Clothing & Accessories

### Step 1: Load and Explore Data

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("Ecommerce_data.csv")

# Examine structure
print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nClass distribution:\n{df['label'].value_counts()}")

# Visualize distribution
plt.figure(figsize=(10, 6))
df['label'].value_counts().plot(kind='bar')
plt.title('Product Category Distribution')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### Step 2: Text Preprocessing

```python
import re
from sklearn.model_selection import train_test_split

def clean_text(text):
    """Preprocess text for classification"""
    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text

# Apply preprocessing
df['clean_text'] = df['Text'].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
```

### Step 3: Feature Extraction with TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=5000,      # Limit vocabulary
    ngram_range=(1, 2),     # Unigrams and bigrams
    min_df=2,               # Minimum document frequency
    max_df=0.95,            # Maximum document frequency
    stop_words='english'
)

# Fit and transform training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform test data (don't fit!)
X_test_tfidf = vectorizer.transform(X_test)

print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"Feature matrix shape: {X_train_tfidf.shape}")
```

### Step 4: Model Training and Comparison

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

# Define models
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Linear SVM': LinearSVC(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1)
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    start_time = time.time()

    # Train
    model.fit(X_train_tfidf, y_train)

    # Predict
    predictions = model.predict(X_test_tfidf)

    # Evaluate
    accuracy = accuracy_score(y_test, predictions)
    elapsed_time = time.time() - start_time

    results[name] = {
        'accuracy': accuracy,
        'time': elapsed_time
    }

    print(f"{name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Time: {elapsed_time:.2f}s\n")

# Compare results
results_df = pd.DataFrame(results).T
print("\n=== Model Comparison ===")
print(results_df.sort_values('accuracy', ascending=False))
```

### Step 5: Detailed Evaluation of Best Model

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Use Logistic Regression (often performs well)
best_model = LogisticRegression(max_iter=1000)
best_model.fit(X_train_tfidf, y_train)
predictions = best_model.predict(X_test_tfidf)

# Classification report
print("=== Classification Report ===")
print(classification_report(y_test, predictions))

# Confusion matrix
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=best_model.classes_,
            yticklabels=best_model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()
```

### Step 6: Making Predictions on New Data

```python
def predict_category(text):
    """Predict category for new text"""
    # Preprocess
    clean = clean_text(text)

    # Vectorize
    features = vectorizer.transform([clean])

    # Predict
    prediction = best_model.predict(features)[0]
    probabilities = best_model.predict_proba(features)[0]

    return {
        'prediction': prediction,
        'confidence': max(probabilities),
        'probabilities': dict(zip(best_model.classes_, probabilities))
    }

# Test with new products
test_products = [
    "Apple iPhone 13 Pro with A15 Bionic chip and 5G connectivity",
    "Comfortable cotton t-shirt for everyday wear",
    "Best-selling novel by Stephen King",
    "Stainless steel kitchen knife set"
]

for product in test_products:
    result = predict_category(product)
    print(f"\nProduct: {product[:50]}...")
    print(f"  Category: {result['prediction']}")
    print(f"  Confidence: {result['confidence']:.2%}")
```

### Step 7: Saving the Model

```python
import joblib

# Save model and vectorizer
joblib.dump(best_model, 'ecommerce_classifier.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

# Load later
loaded_model = joblib.load('ecommerce_classifier.joblib')
loaded_vectorizer = joblib.load('tfidf_vectorizer.joblib')
```

---

## Evaluation Metrics

### Accuracy

$$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}$$

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)
```

**When to use:** Balanced datasets
**Limitation:** Misleading for imbalanced data

### Precision

$$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives + False Positives}}$$

**Interpretation:** Of all predicted positives, how many are actually positive?
**When to use:** When false positives are costly (spam detection)

### Recall (Sensitivity)

$$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives + False Negatives}}$$

**Interpretation:** Of all actual positives, how many did we catch?
**When to use:** When false negatives are costly (disease detection)

### F1 Score

$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision + Recall}}$$

**Interpretation:** Harmonic mean of precision and recall
**When to use:** When you need balance between precision and recall

### Complete Metrics Example

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# For multi-class, specify averaging method
print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
print(f"Precision: {precision_score(y_test, predictions, average='weighted'):.4f}")
print(f"Recall:    {recall_score(y_test, predictions, average='weighted'):.4f}")
print(f"F1 Score:  {f1_score(y_test, predictions, average='weighted'):.4f}")

# Detailed per-class metrics
print("\n" + classification_report(y_test, predictions))
```

---

## Best Practices

### 1. Always Use Stratified Splits

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,  # Maintain class distribution
    random_state=42
)
```

### 2. Use Cross-Validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
```

### 3. Handle Imbalanced Data

```python
# Option 1: Class weights
model = LogisticRegression(class_weight='balanced')

# Option 2: Oversampling
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Option 3: Undersampling
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
```

### 4. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1.0, 10.0],
    'max_iter': [100, 500, 1000]
}

grid_search = GridSearchCV(
    LogisticRegression(),
    param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

### 5. Use Pipeline for Cleaner Code

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(max_features=5000)),
    ('classifier', MultinomialNB())
])

# Fit and predict
pipeline.fit(X_train_text, y_train)
predictions = pipeline.predict(X_test_text)
```

---

## Common Pitfalls

### 1. Data Leakage

```python
# WRONG - fitting vectorizer on all data
vectorizer.fit(all_text)
X_train = vectorizer.transform(train_text)
X_test = vectorizer.transform(test_text)

# CORRECT - fit only on training data
vectorizer.fit(train_text)
X_train = vectorizer.transform(train_text)
X_test = vectorizer.transform(test_text)
```

### 2. Ignoring Class Imbalance

```python
# Check class distribution
print(y_train.value_counts())

# If imbalanced, use appropriate metrics and techniques
```

### 3. Not Preprocessing Test Data

```python
# WRONG - different preprocessing for train vs test
X_train = preprocess(train_text)
X_test = test_text  # Forgot to preprocess!

# CORRECT - same preprocessing for both
X_train = preprocess(train_text)
X_test = preprocess(test_text)
```

### 4. Using Accuracy for Imbalanced Data

```python
# For imbalanced data, use F1, precision, recall instead
from sklearn.metrics import f1_score

f1 = f1_score(y_test, predictions, average='weighted')
```

---

## Advanced Techniques

### 1. Ensemble Methods

```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[
        ('nb', MultinomialNB()),
        ('lr', LogisticRegression(max_iter=1000)),
        ('svm', LinearSVC(max_iter=1000))
    ],
    voting='hard'
)

ensemble.fit(X_train, y_train)
```

### 2. Deep Learning with Neural Networks

```python
# Using Keras/TensorFlow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train.toarray(), y_train, epochs=10, batch_size=32, validation_split=0.2)
```

### 3. Transfer Learning with BERT

```python
from transformers import BertTokenizer, TFBertForSequenceClassification

# Load pre-trained BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)

# Fine-tune on your data
# ... (requires more setup)
```

---

## Datasets Included

| File                        | Description          | Use Case                   |
| :-------------------------- | :------------------- | :------------------------- |
| `spam.csv`                  | SMS spam collection  | Binary classification      |
| `Ecommerce_data.csv`        | Product descriptions | Multi-class classification |
| `Emotion_classify_Data.csv` | Text emotions        | Sentiment/emotion analysis |
| `Fake_Real_Data.csv`        | News articles        | Fake news detection        |
| `movies_sentiment_data.csv` | Movie reviews        | Sentiment analysis         |
| `news_dataset.json`         | News articles        | Topic classification       |

---

## Key Takeaways

1. **Pipeline matters:** Preprocessing → Vectorization → Training → Evaluation
2. **TF-IDF with Naive Bayes** is a strong baseline
3. **Always use stratified splits** for classification
4. **Handle imbalanced data** appropriately
5. **Use cross-validation** for reliable evaluation
6. **F1 score** is better than accuracy for imbalanced data
7. **Pipelines** keep code clean and prevent data leakage
8. **Save models** for deployment

---

## Practice Exercise

1. Load the spam dataset (`spam.csv`)
2. Build a complete classification pipeline
3. Compare at least 3 different algorithms
4. Tune hyperparameters using GridSearchCV
5. Evaluate using appropriate metrics
6. Make predictions on new messages

See [text_classification.ipynb](text_classification.ipynb) for a complete implementation.

---

## Further Reading

- [Scikit-learn Text Classification Tutorial](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
- [Text Classification with Transformers](https://huggingface.co/docs/transformers/tasks/sequence_classification)
- [Handling Imbalanced Data](https://imbalanced-learn.org/stable/)
- [Model Evaluation Guide](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

## Next Module

After completing Text Classification, you're ready for:

➡️ **Practical NLP with Hugging Face** — Transformer-based models
➡️ **Deep Learning for NLP** — Neural network architectures
