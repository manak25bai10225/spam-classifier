# spam-classifier
# Spam Email Classifier

A beginner-friendly machine learning project that classifies email messages as **spam** or **ham (not spam)** using the Naive Bayes algorithm.

Built for a first-year B.Tech AI/ML college portfolio — no GUI, no deep learning, just clean terminal-based Python.

---

## What it does

- Trais a Naive Bayes model on a set of labelled email messages
- Converts text to numbers using TF-IDF (Term Frequency–Inverse Document Frequency)
- Lets you type any message in the terminal and instantly classifies it
- Shows the result and a confidence percentage

---

## How it works

1. **Dataset** — 30 labelled messages (15 spam, 15 ham) built into the script
2. **TF-IDF Vectorizer** — turns each message into a numerical vector
3. **Multinomial Naive Bayes** — learns the probability of words appearing in spam vs ham
4. **Prediction** — takes your input, vectorizes it the same way, and predicts the class

---

## Requirements

```
pandas
scikit-learn
```

Install with:

```bash
pip install pandas scikit-learn
```

---

## Run

```bash
python spam_classifier.py
```

Example session:

```
 model trained successfully
 test accuracy: 100.0%
---------------------------------------------

 spam email classifier
 type an email message and press enter
 type 'quit' to exit
---------------------------------------------

 enter message: Congratulations you won a free prize click here now
 result   : SPAM
 confience: 99.2%
 this message looks like unwanted or suspicious content.

 enter message: Hey are we still on for lunch tomorrow?
 result   : NOT SPAM (ham)
 confidence: 97.8%
 this message looks like a normal, genuine email.
```

---

## Project structure

```
spam-classifier/
│
└── spam_classifier.py   # entire project in one file
```

---

## Concepts covered

- Text preprocessing with TF-IDF
- Supervised machine learning (classification)
- Train/test split and model evaluation
- Naive Bayes for text data

---

## Possible improvements

- Load a larger real-world dataset (e.g. the SMS Spam Collection from UCI)
- Add more features like email length, punctuation count
- Try other classifiers like Logistic Regression and compare accuracy
