# spam_classifier.py
# a simple spam email classifier using naive bayes
# libraries needed: pip install pandas scikit-learn

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------------------------------------
# step 1: our dataset (label, message)
# 'spam' = unwanted email, 'ham' = normal email
# -------------------------------------------------------

 data = {
    "label": [
        "spam", "spam", "spam", "spam", "spam",
        "spam", "spam", "spam", "spam", "spam",
        "spam",  "spam", "spam", "spam", "spam",
        "ham",  "ham",  "ham",  "ham",  "ham",
        "ham",  "ham",  "ham",  "ham",  "ham",
        "ham",   "ham",  "ham",  "ham",  "ham",
    ],
    "message": [
        "Congratulations! You have won a free iPhone. Click here to claim now.",
        "You are selected for a cash prize of 50000 rupees. Reply YES to claim.",
        "URGENT: Your bank account is at risk. Verify your details immediately.",
        "Free entry in our lucky draw. Win a luxury car today!",
        "Get cheap medicines online. No prescription needed. Order now.",
        "Make money fast from home. Earn 10000 per day. Limited slots!",
        "Your credit card has been compromised. Click the link to secure it.",
        "Buy followers and likes instantly. Boost your social media today.",
        "You have a pending refund. Provide your bank details to receive it.",
        "WINNER! You have been chosen. Call now to claim your reward.",
        "Hot singles near you are waiting. Click here to meet them.",
        "Lowest loan rates guaranteed. Apply now and get approved instantly.",
        "Lose weight in 7 days with this one simple trick. Buy now.",
        "Investment opportunity with 300 percent returns. Act fast!",
        "Your Netflix account will be cancelled. Update payment info now.",
        "Hey, are we still meeting for lunch tomorrow?",
        "Can you send me the notes from today's class?",
        "The assignment is due on Friday. Have you started yet?",
        "Mom said dinner is at 7. Are you coming?",
        "I finished the project. Let me know if you want to review it.",
        "Happy birthday! Hope you have a great day.",
        "The library is closed tomorrow due to maintenance.",
        "Can you help me understand the linear regression problem?",
        "Let's catch up this weekend. It has been a long time.",
        "The professor postponed the exam to next Monday.",
        "I will be late today. Please save me a seat.",
        "Did you watch the match last night? It was incredible.",
        "Thanks for helping me with the code. It works now.",
        "Our team meeting is scheduled for 3pm in room 204.",
        "I found a great article on machine learning. Sharing the link.",
    ]
}

df = pd.DataFrame(data)

# -------------------------------------------------------
# step 2: convert labels to numbers (spam=1, ham=0)
# -------------------------------------------------------

df["label_num"] = df["label"].map({"spam": 1, "ham": 0})

# -------------------------------------------------------
# step 3: convert email text into numbers using tfidf
# tfidf gives higher scores to words that are rare but
# important, and lower scores to very common words
# -------------------------------------------------------

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["message"])
y = df["label_num"]

# -------------------------------------------------------
# step 4: split data into training and testing sets
# we train on 80% and test on the remaining 20%
# -------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------------
# step 5: train the naive bayes model
# naive bayes works well for text classification tasks
# -------------------------------------------------------

model = MultinomialNB()
model.fit(X_train, y_train)

# -------------------------------------------------------
# step 6: check accuracy on the test set
# -------------------------------------------------------

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\n model trained successfully")
print(f" test accuracy: {accuracy * 100:.1f}%")
print("-" * 45)

# -------------------------------------------------------
# step 7: let the user type any message and classify it
# -------------------------------------------------------

print("\n spam email classifier")
print(" type an email message and press enter")
print(" type 'quit' to exit")
print("-" * 45)

while True:
    user_input = input("\n enter message: ").strip()

    if user_input.lower() == "quit":
        print(" goodbye!")
        break

    if not user_input:
        print(" please type something.")
        continue

    # convert the user message using the same vectorizer
    input_vector = vectorizer.transform([user_input])

    # predict: 1 = spam, 0 = ham
    result = model.predict(input_vector)[0]

    # get the confidence percentage for the predicted class
    probabilities = model.predict_proba(input_vector)[0]
    confidence = probabilities[result] * 100

    if result == 1:
        print(f" result   : SPAM")
        print(f" confidence: {confidence:.1f}%")
        print(" this message looks like unwanted or suspicious content.")
    else:
        print(f" result   : NOT SPAM (ham)")
        print(f" confidence: {confidence:.1f}%")
        print(" this message looks like a normal, genuine email.")
