# SMS Spam Detector (Machine Learning Project)

# Step 1: Import necessary libraries
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load dataset
# The dataset usually has two columns: v1 (label: ham/spam), v2 (SMS text)
data = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only useful columns
data = data[['v1', 'v2']]
data.columns = ['Category', 'Text']

# Step 3: Basic text cleaning function
def preprocess(msg):
    msg = msg.lower()                                # lowercase everything
    msg = re.sub(r"http\S+", " link ", msg)          # replace URLs with word 'link'
    msg = re.sub(r"\d+", " number ", msg)            # replace digits with word 'number'
    msg = msg.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    msg = re.sub(r"\s+", " ", msg).strip()           # remove extra spaces
    return msg

data['Clean_Text'] = data['Text'].apply(preprocess)

# Step 4: Encode labels (ham=0, spam=1)
data['Category_Num'] = data['Category'].map({'ham':0, 'spam':1})

# Step 5: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data['Clean_Text'], 
    data['Category_Num'], 
    test_size=0.25, 
    random_state=123
)

# Step 6: Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=4000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 7: Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 8: Predictions and evaluation
y_pred = model.predict(X_test_vec)

print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nDetailed Report:\n", classification_report(y_test, y_pred, target_names=['ham','spam']))

# Step 9: Test on your own messages
def check_message(msg):
    clean_msg = preprocess(msg)
    vec_msg = vectorizer.transform([clean_msg])
    result = model.predict(vec_msg)[0]
    return "Spam 🚫" if result == 1 else "Ham ✅"

# Example
print(check_message("Congrats! You won 5000 rupees, call now!"))
print(check_message("Hey, are you coming to the party tomorrow?"))
while True:
    user_msg = input("\nEnter an SMS (or type 'exit' to quit): ")
    if user_msg.lower() == "exit":
        print("Goodbye!")
        break
    print("Prediction:", check_message(user_msg))