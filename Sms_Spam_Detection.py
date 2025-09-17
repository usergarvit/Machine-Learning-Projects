
# Step 1: Import necessary libraries
import pandas as pd
import re
import string
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only useful columns
data = data[['v1', 'v2']]
data.columns = ['Category', 'Text']

# Step 3: Basic text cleaning function
def preprocess(msg):
    msg = msg.lower()
    msg = re.sub(r"http\S+", " link ", msg)
    msg = re.sub(r"\d+", " number ", msg)
    msg = msg.translate(str.maketrans("", "", string.punctuation))
    msg = re.sub(r"\s+", " ", msg).strip()
    return msg

data['Clean_Text'] = data['Text'].apply(preprocess)

# Step 4: Encode labels
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

# Step 8: Predictions and evaluation (for console output)
y_pred = model.predict(X_test_vec)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nDetailed Report:\n", classification_report(y_test, y_pred, target_names=['ham','spam']))

# Step 9: Function to test custom messages
def check_message(msg):
    clean_msg = preprocess(msg)
    vec_msg = vectorizer.transform([clean_msg])
    result = model.predict(vec_msg)[0]
    return "Spam 🚫" if result == 1 else "Ham ✅"


# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="centered")

st.title("📩 SMS Spam Detection App")
st.write("This app uses *Machine Learning (Naive Bayes + TF-IDF)* to classify SMS messages as Spam or Ham.")

# User input
user_msg = st.text_area("✉ Enter your SMS message:")

if st.button("Check Message"):
    if user_msg.strip() != "":
        prediction = check_message(user_msg)
        if "Spam" in prediction:
            st.error(f"🚨 {prediction}")
        else:
            st.success(f"✅ {prediction}")
    else:
        st.warning("⚠ Please enter a message before checking.")

# Show model accuracy
st.subheader("📊 Model Performance on Test Data")
st.write(f"*Accuracy:* {accuracy_score(y_test, y_pred):.2f}")
while True:
    user_msg = input("\nEnter an SMS (or type 'exit' to quit): ")
    if user_msg.lower() == "exit":
        break
    print("Prediction:", check_message(user_msg))