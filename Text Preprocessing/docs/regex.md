# Regular Expressions (Regex) in NLP 🧪

Regular expressions are powerful pattern-matching tools for finding, extracting, and manipulating text. They're essential for data cleaning and extracting structured information from unstructured text.

---

## 📖 What are Regular Expressions?

**Regular expressions** (regex) are sequences of characters that define search patterns. They allow you to:

- **Find** specific patterns in text
- **Extract** data like phone numbers, emails, dates
- **Replace** or **clean** unwanted characters
- **Validate** input formats

```python
import re

text = "Contact us at support@company.com or call 555-123-4567"

# Find email
email = re.search(r'\b[\w.%+-]+@[\w.-]+\.[a-zA-Z]{2,}\b', text)
print(email.group())  # support@company.com

# Find phone
phone = re.search(r'\d{3}-\d{3}-\d{4}', text)
print(phone.group())  # 555-123-4567
```

---

## 🎯 Why Regex in NLP?

1. **Data Cleaning**: Remove URLs, HTML tags, special characters
2. **Entity Extraction**: Phone numbers, emails, IDs with known formats
3. **Pre-processing**: Normalize text before ML models
4. **Pattern Matching**: Find specific formats in documents
5. **Validation**: Check if input matches expected format

---

## 📊 Regex Syntax Basics

### Character Classes

| Pattern | Matches                            | Example               |
| :------ | :--------------------------------- | :-------------------- |
| `\d`    | Any digit (0-9)                    | "7" in "abc7def"      |
| `\D`    | Any non-digit                      | "abc" in "abc7"       |
| `\w`    | Word character (a-z, A-Z, 0-9, \_) | "hello_123"           |
| `\W`    | Non-word character                 | spaces, punctuation   |
| `\s`    | Whitespace (space, tab, newline)   | " "                   |
| `\S`    | Non-whitespace                     | any visible character |
| `.`     | Any character except newline       | anything              |

### Quantifiers

| Pattern | Meaning         | Example                              |
| :------ | :-------------- | :----------------------------------- |
| `*`     | 0 or more       | `a*` matches "", "a", "aaa"          |
| `+`     | 1 or more       | `a+` matches "a", "aaa" (not "")     |
| `?`     | 0 or 1          | `a?` matches "", "a"                 |
| `{n}`   | Exactly n times | `a{3}` matches "aaa"                 |
| `{n,}`  | n or more       | `a{2,}` matches "aa", "aaa", ...     |
| `{n,m}` | Between n and m | `a{2,4}` matches "aa", "aaa", "aaaa" |

### Anchors

| Pattern | Meaning             |
| :------ | :------------------ |
| `^`     | Start of string     |
| `$`     | End of string       |
| `\b`    | Word boundary       |
| `\B`    | Not a word boundary |

### Character Sets

| Pattern       | Matches              |
| :------------ | :------------------- |
| `[abc]`       | a, b, or c           |
| `[^abc]`      | NOT a, b, or c       |
| `[a-z]`       | Any lowercase letter |
| `[A-Z]`       | Any uppercase letter |
| `[0-9]`       | Any digit            |
| `[a-zA-Z0-9]` | Any alphanumeric     |

### Groups and Alternation

| Pattern   | Meaning             |
| :-------- | :------------------ |
| `(abc)`   | Capture group       |
| `(?:abc)` | Non-capturing group |
| `a\|b`    | a OR b              |

---

## 🛠️ Python's `re` Module

### Core Functions

```python
import re

text = "Patient's phone is 7211059591. Bill amount is 120$"

# re.search() - Find first match
match = re.search(r'\d+', text)
print(match.group())  # 7211059591

# re.findall() - Find all matches
all_numbers = re.findall(r'\d+', text)
print(all_numbers)  # ['7211059591', '120']

# re.sub() - Replace matches
cleaned = re.sub(r'\d+', '[NUMBER]', text)
print(cleaned)  # Patient's phone is [NUMBER]. Bill amount is [NUMBER]$

# re.split() - Split by pattern
parts = re.split(r'\s+', text)
print(parts)  # ['Patient's', 'phone', 'is', '7211059591.', ...]
```

---

## 💻 Practical NLP Examples

### 1. Extracting Phone Numbers

```python
import re

text = "Patient's phone is (732)-111-9999, spouse phone number 7211059591. Bill amount is 120$"

# Pattern handles multiple formats
pattern = r'\(\d{3}\)-\d{3}-\d{4}|\d{10}'

phones = re.findall(pattern, text)
print(phones)  # ['(732)-111-9999', '7211059591']
```

### 2. Using Capture Groups

```python
text = "Patient's phone is 7211059591. Bill amount is 120$"

# Capture phone and amount separately
pattern = r'(\d{10})\D+(\d+)\$'

match = re.search(pattern, text)
if match:
    phone_number, bill_amount = match.groups()
    print(f"Phone: {phone_number}")    # 7211059591
    print(f"Bill: ${bill_amount}")     # $120
```

### 3. Extracting Email Addresses

```python
text = """
Contact John at john.doe@company.com or
Jane at jane_smith123@university.edu.
Support: help@support.co.uk
"""

pattern = r'\b[\w.%+-]+@[\w.-]+\.[a-zA-Z]{2,}\b'

emails = re.findall(pattern, text)
print(emails)
# ['john.doe@company.com', 'jane_smith123@university.edu', 'help@support.co.uk']
```

### 4. Cleaning URLs from Text

```python
text = "Check out https://example.com/page and http://test.org for more info"

# Remove URLs
pattern = r'https?://\S+'
cleaned = re.sub(pattern, '', text)
print(cleaned)  # "Check out  and  for more info"
```

### 5. Extracting Dates

```python
text = "Meeting on 12/25/2024, deadline is 2024-01-15, created on Jan 5, 2024"

# Multiple date formats
patterns = {
    'MM/DD/YYYY': r'\d{2}/\d{2}/\d{4}',
    'YYYY-MM-DD': r'\d{4}-\d{2}-\d{2}',
    'Month D, YYYY': r'[A-Z][a-z]{2}\s+\d{1,2},\s+\d{4}'
}

for format_name, pattern in patterns.items():
    matches = re.findall(pattern, text)
    if matches:
        print(f"{format_name}: {matches}")
```

### 6. Removing Special Characters

```python
text = "Hello!!! How are you??? I'm @great #feeling 100% happy!!!"

# Keep only alphanumeric and spaces
cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', text)
print(cleaned)  # "Hello How are you Im great feeling 100 happy"
```

### 7. Extracting Hashtags and Mentions

```python
text = "Great day @Python_Dev! #MachineLearning #NLP @DataScience"

hashtags = re.findall(r'#\w+', text)
mentions = re.findall(r'@\w+', text)

print(f"Hashtags: {hashtags}")   # ['#MachineLearning', '#NLP']
print(f"Mentions: {mentions}")   # ['@Python_Dev', '@DataScience']
```

### 8. Extracting Monetary Values

```python
text = "The product costs $19.99, tax is $2.50, total $22.49 or €20"

# Match various currency formats
pattern = r'[$€£¥]\d+(?:\.\d{2})?|\d+(?:\.\d{2})?\s*(?:dollars?|euros?|pounds?)'

amounts = re.findall(pattern, text, re.IGNORECASE)
print(amounts)  # ['$19.99', '$2.50', '$22.49', '€20']
```

---

## 📊 Common NLP Regex Patterns

| Use Case              | Pattern                                  | Example Match       |
| :-------------------- | :--------------------------------------- | :------------------ |
| **Email**             | `[\w.%+-]+@[\w.-]+\.[a-zA-Z]{2,}`        | user@example.com    |
| **URL**               | `https?://\S+`                           | https://example.com |
| **Phone (US)**        | `\d{3}[-.\s]?\d{3}[-.\s]?\d{4}`          | 555-123-4567        |
| **Date (MM/DD/YYYY)** | `\d{2}/\d{2}/\d{4}`                      | 12/25/2024          |
| **Time**              | `\d{1,2}:\d{2}(?:\s*[AP]M)?`             | 3:30 PM             |
| **IP Address**        | `\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}`     | 192.168.1.1         |
| **Credit Card**       | `\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}` | 1234-5678-9012-3456 |
| **ZIP Code (US)**     | `\d{5}(?:-\d{4})?`                       | 12345 or 12345-6789 |
| **HTML Tags**         | `<[^>]+>`                                | `<p>`, `</div>`     |
| **Hashtag**           | `#\w+`                                   | #Python             |
| **Mention**           | `@\w+`                                   | @username           |

---

## ⚙️ Regex Flags

| Flag                      | Description                       |
| :------------------------ | :-------------------------------- |
| `re.IGNORECASE` or `re.I` | Case-insensitive matching         |
| `re.MULTILINE` or `re.M`  | `^` and `$` match line boundaries |
| `re.DOTALL` or `re.S`     | `.` matches newlines too          |
| `re.VERBOSE` or `re.X`    | Allow comments in pattern         |

```python
# Case-insensitive search
pattern = r'python'
text = "I love Python and PYTHON"
matches = re.findall(pattern, text, re.IGNORECASE)
print(matches)  # ['Python', 'PYTHON']

# Verbose pattern with comments
pattern = r'''
    \d{3}     # Area code
    [-.\s]?   # Optional separator
    \d{3}     # First 3 digits
    [-.\s]?   # Optional separator
    \d{4}     # Last 4 digits
'''
match = re.search(pattern, "Call 555-123-4567", re.VERBOSE)
```

---

## ⚠️ Regex Pitfalls

| Pitfall                | Problem                          | Solution                 |
| :--------------------- | :------------------------------- | :----------------------- |
| **Greedy matching**    | `.*` matches too much            | Use `.*?` (non-greedy)   |
| **Special characters** | `.` matches any char             | Escape with `\.`         |
| **Unicode**            | `\w` might not match all letters | Use `[\w\u0080-\uFFFF]`  |
| **Performance**        | Complex patterns are slow        | Simplify or compile      |
| **Backtracking**       | Catastrophic backtracking        | Avoid nested quantifiers |

### Greedy vs Non-Greedy

```python
text = "<tag>content1</tag><tag>content2</tag>"

# Greedy (default) - matches everything between first < and last >
greedy = re.findall(r'<.*>', text)
print(greedy)  # ['<tag>content1</tag><tag>content2</tag>']

# Non-greedy - matches minimal content
non_greedy = re.findall(r'<.*?>', text)
print(non_greedy)  # ['<tag>', '</tag>', '<tag>', '</tag>']
```

---

## 🎓 Best Practices

1. **Start simple**: Build patterns incrementally

   ```python
   # Step 1: Match any digits
   # Step 2: Match exactly 10 digits
   # Step 3: Handle separators
   ```

2. **Test patterns**: Use tools like [regex101.com](https://regex101.com/)

3. **Compile frequently used patterns**:

   ```python
   email_pattern = re.compile(r'[\w.%+-]+@[\w.-]+\.[a-zA-Z]{2,}')
   # Faster when used multiple times
   emails = email_pattern.findall(text)
   ```

4. **Use raw strings**: Always prefix with `r` to avoid escape issues

   ```python
   pattern = r'\d+'  # Correct
   pattern = '\\d+'  # Works but harder to read
   ```

5. **Handle edge cases**: Test with empty strings, special characters

6. **Document complex patterns**: Use verbose mode with comments

---

## 📚 Further Reading

- [Python re Documentation](https://docs.python.org/3/library/re.html)
- [Regex101 - Online Tester](https://regex101.com/)
- [Regular-Expressions.info](https://www.regular-expressions.info/)
- [Regex Crossword](https://regexcrossword.com/) - Practice!

---

## ▶️ Next Steps

After mastering regex, proceed to:

- [Complete Pipeline](pipeline.md) - Combine all preprocessing techniques
- [Named Entity Recognition](ner.md) - ML-based entity extraction
