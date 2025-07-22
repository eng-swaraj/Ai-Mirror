import streamlit as st
from transformers import pipeline
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import speech_recognition as sr
from collections import Counter
import re
from wordcloud import WordCloud
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

st.set_page_config(page_title="🧠 Mood Mirror", layout="centered")
st.title("🪞 AI Mood Mirror")
st.write("Speak or type how your day went. Let AI reflect your emotions 🌤️")

# Load sentiment & emotion model
@st.cache_resource
def load_pipelines():
    sentiment_analyzer = pipeline("sentiment-analysis")
    emotion_analyzer = pipeline("text-classification",
                                model="j-hartmann/emotion-english-distilroberta-base",
                                top_k=1)
    summarizer = pipeline("summarization")
    return sentiment_analyzer, emotion_analyzer, summarizer

sentiment_analyzer, emotion_analyzer, summarizer = load_pipelines()

# --- Voice input function ---
def transcribe_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... Speak now")
        try:
            audio = recognizer.listen(source, timeout=5)
            st.success("✅ Got it! Transcribing...")
            return recognizer.recognize_google(audio)
        except sr.WaitTimeoutError:
            st.warning("⏰ You were silent for too long.")
        except sr.UnknownValueError:
            st.warning("🤔 Could not understand your voice.")
        except Exception as e:
            st.error(f"Error: {e}")
    return ""

# --- Input area ---
col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_area("📝 How was your day?", height=150)
with col2:
    if st.button("🎤 Speak Instead"):
        user_input = transcribe_audio()

# --- Mood analysis ---
if st.button("Analyze My Mood"):
    if user_input.strip() == "":
        st.warning("Please write or speak something about your day.")
    else:
        sentiment = sentiment_analyzer(user_input)[0]
        emotion = emotion_analyzer(user_input)[0][0]  # fix for top_k=1

        st.subheader("🔍 AI Mood Reflection:")
        st.markdown(f"**Primary Emotion**: `{emotion['label']}` with {emotion['score']:.2f} confidence")
        st.markdown(f"**Sentiment**: `{sentiment['label']}` with {sentiment['score']:.2f} confidence")

        # Reflective message
        st.markdown("**🪞 Reflection:**")
        reflections = {
            "joy": "Sounds like a good day! 🌈 Keep the positive energy flowing.",
            "sadness": "It's okay to feel down. Tomorrow is a new chance. 🌱",
            "anger": "Rough day? Maybe a deep breath and music can help. 🎧",
            "fear": "Anxiety is valid. Try grounding yourself — you're safe now.",
            "surprise": "Something unexpected? Embrace the unknown. 🌠",
            "neutral": "A calm day can be beautiful in its own way. ☕",
        }
        main_emotion = emotion['label'].lower()
        reflection = reflections.get(main_emotion, "Emotions make us human. Embrace yours today.")
        st.info(reflection)

        # Save entry
        mood_data = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "text": user_input,
            "emotion": emotion["label"],
            "emotion_score": round(emotion["score"], 2),
            "sentiment": sentiment["label"],
            "sentiment_score": round(sentiment["score"], 2)
        }

        if not os.path.exists("mood_log.csv"):
            pd.DataFrame([mood_data]).to_csv("mood_log.csv", index=False)
        else:
            pd.concat([pd.read_csv("mood_log.csv"), pd.DataFrame([mood_data])], ignore_index=True).to_csv("mood_log.csv", index=False)

# --- Show mood history ---
if os.path.exists("mood_log.csv"):
    st.markdown("---")
    st.subheader("📈 Mood Over Time")
    df = pd.read_csv("mood_log.csv")
    df['date'] = pd.to_datetime(df['date'])

    mood_counts = df.groupby(['date', 'emotion']).size().unstack(fill_value=0)
    st.line_chart(mood_counts)

    with st.expander("📋 View Your Entries"):
        st.dataframe(df[::-1], use_container_width=True)

    st.download_button("📥 Download Mood Log", data=df.to_csv(index=False), file_name="mood_log.csv")

    st.markdown("---")
    st.subheader("📝 Daily Summary")
    today = datetime.now().strftime("%Y-%m-%d")
    today_entries = df[df['date'] == today]['text'].tolist()

    if today_entries:
        full_text = " ".join(today_entries)
        if len(full_text.split()) > 50:
            summary = summarizer(full_text[:1024])[0]['summary_text']
            st.success(f"📌 Summary for {today}:\n{summary}")
        else:
            st.info("Not enough text for summary today. Try writing a bit more.")
    else:
        st.info("No entries found for today.")

    # --- Weekly Summary ---
    st.markdown("---")
    st.subheader("🗓️ Weekly Summary")
    week_ago = datetime.now() - timedelta(days=7)
    weekly_df = df[df['date'] >= week_ago]
    weekly_entries = weekly_df['text'].tolist()

    if weekly_entries:
        full_week_text = " ".join(weekly_entries)
        if len(full_week_text.split()) > 50:
            summary = summarizer(full_week_text[:2048])[0]['summary_text']
            st.success("📅 Weekly Summary:")
            st.write(summary)
        else:
            st.info("Not enough content this week to generate a summary.")

        # Most common words
        words = re.findall(r'\b\w+\b', full_week_text.lower())
        common_words = Counter(words).most_common(10)
        st.subheader("🔠 Most Used Words This Week:")
        for word, count in common_words:
            st.write(f"- **{word}**: {count} times")

        # Word Cloud
        st.subheader("☁️ Word Cloud")
        wc = WordCloud(width=600, height=300, background_color='white').generate(full_week_text)
        st.image(wc.to_array(), use_column_width=True)

        # Weekly graph
        st.subheader("📊 Weekly Emotion Distribution")
        emotion_week = weekly_df.groupby(['date', 'emotion']).size().unstack(fill_value=0)
        st.bar_chart(emotion_week)

        # Journal Page
        st.markdown("---")
        st.subheader("📘 Download Weekly Journal Page")
        journal_text = f"# Weekly Mood Journal\n\n## Summary:\n{summary}\n\n## Most Used Words:\n" + "\n".join([f"- {word}: {count}" for word, count in common_words]) + "\n\n## Entries:\n" + "\n---\n".join(weekly_entries)
        st.download_button("📄 Download Weekly Journal", data=journal_text, file_name="weekly_journal.txt")

        # Email option
        st.markdown("---")
        st.subheader("📨 Email Your Weekly Journal")
        with st.form("email_form"):
            recipient = st.text_input("Enter your email")
            submitted = st.form_submit_button("Send Email")
        if submitted and recipient:
            try:
                msg = MIMEMultipart()
                msg['From'] = 'your_email@example.com'  # replace with real sender
                msg['To'] = recipient
                msg['Subject'] = 'Your Weekly Mood Journal'
                msg.attach(MIMEText(journal_text, 'plain'))

                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.starttls()
                server.login('your_email@example.com', 'your_password')  # use app password or token
                server.send_message(msg)
                server.quit()
                st.success("✅ Email sent successfully!")
            except Exception as e:
                st.error(f"Failed to send email: {e}")
