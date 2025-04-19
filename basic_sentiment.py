import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Function to preprocess text (convert to lowercase)
def preprocess_text(text):
    return text.lower()

# Sample Small Dataset (Improved)
data = {
    "review": [
        "I love this movie, it was amazing!",
        "Absolutely fantastic! A must-watch.",
        "Worst movie ever, I hated it.",
        "Terrible film, waste of time.",
        "It was okay, not great but not bad either.",
        "The plot was boring and slow.",
        "Amazing cinematography and great acting!",
        "Horrible experience, I regret watching it.",
        "Not my favorite, but it had some good moments.",
        "I really enjoyed the story, well done!",
        "movie was good",
        "movie has bad direction",
    ],
    "sentiment": [1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0]  # 1 = Positive, 0 = Negative
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Apply Preprocessing to Dataset
df["review"] = df["review"].apply(preprocess_text)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["review"])
y = df["sentiment"]

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X, y)

# Streamlit UI
st.title("🎬 Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict sentiment:")

# User Input
user_input = st.text_area("Enter your review:")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Apply Preprocessing to Input
        processed_input = preprocess_text(user_input)

        # Transform Input
        input_transformed = vectorizer.transform([processed_input])
        prediction = model.predict(input_transformed)[0]
        
        # Show Result
        sentiment = "Positive 😃" if prediction == 1 else "Negative 😞"
        st.subheader(f"Sentiment: {sentiment}")
    else:
        st.warning("Please enter a review.")
