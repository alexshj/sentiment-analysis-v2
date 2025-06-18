import streamlit as st
import joblib
import re
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary NLTK data
nltk.download('stopwords')

# Load improved model & vectorizer
model = joblib.load("sentiment_model_v2.pkl")
vectorizer = joblib.load("vectorizer_v2.pkl")

# Initialize tokenizer & stemmer
tokenizer = TweetTokenizer()
stemmer = PorterStemmer()

# Preprocessing function
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = tokenizer.tokenize(text)
    filtered = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(filtered)

# Streamlit UI
st.title("🧠 Improved Sentiment Analysis (v2)")
st.subheader("Enter a sentence or tweet to analyze its sentiment")

user_input = st.text_area("Your Input")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        processed = preprocess(user_input)
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)[0]
        label = {
            1: "🟢 Positive",
            0: "🟡 Neutral",
            -1: "🔴 Negative"
        }
        st.success(f"Predicted Sentiment: {label.get(prediction, 'Unknown')}")
