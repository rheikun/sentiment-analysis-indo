import streamlit as st
import pandas as pd
import altair as alt
import torch
from transformers import pipeline

device = 0 if torch.cuda.is_available() else -1

# Load fine-tuned sentiment analysis model
model_path = "./trained_model1"  # Path model yang telah dilatih
sentiment_model = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path, device=device)

def main():
    st.title("Sentiment Analysis NLP App - Bahasa Indonesia")
    st.subheader("Streamlit Projects - Bahasa Indonesia Support")
    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.subheader("Home")
        with st.form("nlpForm"):
            raw_text = st.text_area("Masukkan Teks Di Sini")
            submit_button = st.form_submit_button(label='Analisis')
        
        # Layout
        col1, col2 = st.columns(2)
        
        if submit_button:
            with col1:
                st.info("Hasil")
                sentiment = analyze_sentiment_indonesian(raw_text)

                # Pemetaan label
                label_mapping = {
                    "LABEL_0": "Negatif",
                    "LABEL_1": "Netral",
                    "LABEL_2": "Positif"
                }

                # Ambil label dari hasil sentimen
                sentiment_label = sentiment[0]['label']
                sentiment_result = label_mapping[sentiment_label]  # Mendapatkan label deskriptif

                # Tampilkan hasil
                st.write(f"Label: {sentiment_label}")
                st.write(f"Hasil Sentimen: {sentiment_result}")

                # Emoji berdasarkan label hasil
                if sentiment_label == "LABEL_2":  # Positif
                    st.markdown("Sentimen: Positif :smiley:")
                elif sentiment_label == "LABEL_0":  # Negatif
                    st.markdown("Sentimen: Negatif :angry:")
                elif sentiment_label == "LABEL_1":  # Netral
                    st.markdown("Sentimen: Netral :neutral_face:")

                # Convert sentiment to DataFrame
                sentiment_dict = {'metric': 'sentiment', 'value': sentiment[0]['score']}
                result_df = pd.DataFrame([sentiment_dict])
                st.dataframe(result_df)
                
                # Visualization
                c = alt.Chart(result_df).mark_bar().encode(
                    x='metric',
                    y='value',
                    color='metric'
                )
                st.altair_chart(c, use_container_width=True)
            
            with col2:
                st.info("Sentimen Token")
                token_sentiments = analyze_token_sentiment_indonesian(raw_text)
                st.write(token_sentiments)
    
    else:
        st.subheader("Tentang Aplikasi")

def analyze_sentiment_indonesian(text):
    result = sentiment_model(text)
    return result

def analyze_token_sentiment_indonesian(docx):
    tokens = docx.split()
    token_sentiments = {}
    
    for token in tokens:
        sentiment = sentiment_model(token)
        token_sentiments[token] = sentiment[0]['label']
    
    return token_sentiments

if __name__ == '__main__':
    main()
