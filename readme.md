# 🧠 Sentiment Analysis App (v2)

This is an improved sentiment analysis web app built using **Scikit-learn**, **NLTK**, and **Streamlit**.

It uses a custom-trained `LogisticRegression` model on real-world tweets (Sentiment140 dataset) to predict whether a sentence is:

- 🔴 Negative  
- 🟡 Neutral  
- 🟢 Positive

---

## 🚀 Try It Live

👉 [Click here to run the app on Streamlit](https://your-username.streamlit.app/)  
*(replace with your actual app link after deployment)*

---

## 🧠 Features

- Preprocessing using NLTK and TweetTokenizer
- Custom-trained model on 20k+ tweets
- Real-time prediction via web interface
- Fully deployable with Streamlit Cloud

---

## 📦 Requirements

All dependencies are listed in `requirements.txt`. Main packages:

- `streamlit`
- `scikit-learn`
- `nltk`
- `joblib`

---

## 🛠️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app_v2.py
