import streamlit as st
from transformers import pipeline

# -----------------------------
# 감정 분석 모델 로드 (Streamlit 캐시)
# -----------------------------
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

sentiment_model = load_model()

# -----------------------------
# 감정별 음악 추천 데이터
# -----------------------------
music_recommendations = {
    "positive": [
        ("🎵 Happy – Pharrell Williams", "https://www.youtube.com/watch?v=ZbZSe6N_BXs"),
        ("Good as Hell – Lizzo", "https://www.youtube.com/watch?v=vuq-VAiW9kw"),
        ("Uptown Funk – Bruno Mars", "https://www.youtube.com/watch?v=OPf0YbXqDm0"),
    ],
    "negative": [
        ("💔 Someone Like You – Adele", "https://www.youtube.com/watch?v=hLQl3WQQoQ0"),
        ("Fix You – Coldplay", "https://www.youtube.com/watch?v
