import streamlit as st
import pickle
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# ── Load Model & Vectorizer ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('fake_news_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    return model, tfidf

model, tfidf = load_model()

# ── Text Cleaner ──────────────────────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_news(text):
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])
    pred = model.predict(vec)[0]
    return pred

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📰 Fake News Detection System")
st.markdown("*Classify news articles as Real or Fake using NLP & Machine Learning*")
st.markdown("---")

# ── Session State Initialization ──────────────────────────────────────────────
if "news_text" not in st.session_state:
    st.session_state["news_text"] = ""

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Detect News", "📊 Model Performance", "📈 Dataset Insights"])

# ── TAB 1: Detection ──────────────────────────────────────────────────────────
with tab1:
    st.header("Enter a News Article to Classify")
    
    news_input = st.text_area(
        "Paste the news article title and/or content below:",
        height=200,
        value=st.session_state["news_text"],
        placeholder="e.g. Scientists confirm new vaccine 100% effective against all diseases...",
        key="news_input_box"
    )

    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        detect_btn = st.button("🔍 Detect", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state["news_text"] = ""
            st.rerun()

    if detect_btn and news_input.strip():
        prediction = predict_news(news_input)
        st.markdown("---")
        if prediction == 1:
            st.success("## ✅ REAL NEWS")
            st.markdown("This article is classified as **Real / Authentic** news.")
        else:
            st.error("## 🚨 FAKE NEWS")
            st.markdown("This article is classified as **Fake / Misinformation**.")

        with st.expander("🔎 What was analyzed?"):
            cleaned = clean_text(news_input)
            st.write("**Cleaned Text (first 300 chars):**")
            st.write(cleaned[:300] + "...")
            st.write(f"**Word count:** {len(cleaned.split())}")

    elif detect_btn:
        st.warning("Please enter some news text before clicking Detect.")

    st.markdown("---")
    st.subheader("📌 Try These Examples")
    examples = {
        "✅ Real Example": "NEW DELHI (Reuters) - Prime Minister Narendra Modi spoke today...",
        "🚨 Fake Example": "BREAKING: Obama secretly a lizard person confirmed by CIA whistleblower! Deep state exposed!!",
        "✅ Real Example 2": "CAPE CANAVERAL, Fla. (Reuters) - NASA's James Webb Space Telescope has captured the deepest infrared image of the universe, revealing thousands of galaxies some over 13 billion years old.",
    }
    
    for label, text in examples.items():
        if st.button(f"Load: {label}"):
            st.session_state["news_text"] = text
            st.rerun()

# ── TAB 2: Model Performance ──────────────────────────────────────────────────
with tab2:
    st.header("Model Performance Metrics")
    st.markdown("Trained and evaluated on **44,898 articles** (80/20 split)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",  "98%+",  "Best Model")
    col2.metric("Precision", "0.98+", "Best Model")
    col3.metric("Recall",    "0.98+", "Best Model")
    col4.metric("F1-Score",  "0.98+", "Best Model")

    st.markdown("---")
    st.subheader("All Models Comparison")

    perf_data = {
        'Model'     : ['Logistic Regression', 'Naive Bayes', 'Decision Tree', 'Random Forest', 'Linear SVM'],
        'Accuracy'  : [0.986, 0.944, 0.997, 0.991, 0.995],
        'Precision' : [0.985, 0.942, 0.997, 0.991, 0.995],
        'Recall'    : [0.987, 0.947, 0.997, 0.991, 0.995],
        'F1-Score'  : [0.986, 0.944, 0.997, 0.991, 0.995],
    }
    perf_df = pd.DataFrame(perf_data)
    
    st.dataframe(perf_df, use_container_width=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(perf_df))
    w = 0.2
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    for i, m in enumerate(metrics):
        ax.bar(x + i * w, perf_df[m], w, label=m, color=colors[i], alpha=0.85)
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(perf_df['Model'], rotation=15, ha='right')
    ax.set_ylim(0.85, 1.02)
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

# ── TAB 3: Dataset Insights ───────────────────────────────────────────────────
with tab3:
    st.header("Dataset Insights")

    @st.cache_data
    def load_data():
        fake = pd.read_csv('Fake.csv')
        fake['label'] = 0
        true = pd.read_csv('True.csv') 
        true['label'] = 1
        return pd.concat([fake, true], ignore_index=True)

    df = load_data()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Articles", f"{len(df):,}")
    col2.metric("Fake Articles",  f"{(df['label']==0).sum():,}")
    col3.metric("Real Articles",  f"{(df['label']==1).sum():,}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    label_counts = df['label'].value_counts()
    axes[0].pie(label_counts, labels=['Fake', 'Real'],
                autopct='%1.1f%%', colors=['#e74c3c', '#2ecc71'],
                startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[0].set_title('Class Distribution', fontweight='bold')

    fake_subj = df[df['label']==0]['subject'].value_counts().head(6)
    real_subj = df[df['label']==1]['subject'].value_counts().head(6)
    all_subj = pd.concat([fake_subj.rename('Fake'), real_subj.rename('Real')], axis=1).fillna(0)
    all_subj.plot(kind='barh', ax=axes[1], color=['#e74c3c', '#2ecc71'], alpha=0.85)
    axes[1].set_title('Subject Distribution', fontweight='bold')
    axes[1].set_xlabel('Count')

    plt.tight_layout()
    st.pyplot(fig)