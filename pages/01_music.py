import streamlit as st
import pipeline

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
        ("😊 Happy – Pharrell Williams", "https://www.youtube.com/watch?v=ZbZSe6N_BXs"),
        ("Good as Hell – Lizzo", "https://www.youtube.com/watch?v=vuq-VAiW9kw"),
        ("Uptown Funk – Bruno Mars", "https://www.youtube.com/watch?v=OPf0YbXqDm0"),
    ],
    "negative": [
        ("😔 Someone Like You – Adele", "https://www.youtube.com/watch?v=hLQl3WQQoQ0"),
        ("Fix You – Coldplay", "https://www.youtube.com/watch?v=k4V3Mo61fJM"),
        ("Let Her Go – Passenger", "https://www.youtube.com/watch?v=RBumgq5yVrA"),
    ],
    "neutral": [
        ("😐 Shallow – Lady Gaga & Bradley Cooper", "https://www.youtube.com/watch?v=bo_efYhYU2A"),
        ("Counting Stars – OneRepublic", "https://www.youtube.com/watch?v=hT_nvWreIhg"),
        ("Lost Stars – Adam Levine", "https://www.youtube.com/watch?v=cL4uhaQ58Rk"),
    ],
}

# -----------------------------
# UI
# -----------------------------
st.title("🎧 감정 기반 음악 추천")
st.write("당신의 감정을 적어주세요. 어울리는 음악을 추천해드릴게요!")

user_input = st.text_area("🗣 오늘의 기분이나 상황을 입력하세요:", height=120)

if st.button("음악 추천 받기 🎵"):
    if not user_input.strip():
        st.warning("먼저 감정을 입력해주세요!")
    else:
        with st.spinner("감정을 분석 중입니다..."):
            result = sentiment_model(user_input)[0]

        label = result["label"].lower()
        score = result["score"]

        if "positive" in label:
            emotion = "positive"
            emoji = "😊"
        elif "negative" in label:
            emotion = "negative"
            emoji = "😔"
        else:
            emotion = "neutral"
            emoji = "😐"

        st.subheader(f"당신의 감정은: {emoji} **{emotion.upper()}** ({score:.2f})")
        st.write("🎵 추천 음악 리스트:")

        for title, link in music_recommendations[emotion]:
            st.markdown(f"- [{title}]({link})")

st.markdown("---")
st.caption("💡 감정 분석 모델: CardiffNLP Twitter RoBERTa")
