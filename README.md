# 📰 Fake News Detection System

## Objective
Automatically classify news articles as **Real** or **Fake** using Natural Language Processing (NLP) and Machine Learning — helping prevent the spread of misinformation.

## Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit
- **NLP:** TF-IDF Vectorization (50K features, unigrams + bigrams)
- **Environment:** VS Code / Jupyter Notebook

## Dataset
| File | Records | Label |
|------|---------|-------|
| `Fake.csv` | 23,481 | 0 (Fake) |
| `True.xlsx` | 21,417 | 1 (Real) |
| **Total** | **44,898** | — |

**Features used:** `title` + `text` combined into a single content field.

## Methodology
1. **Data Loading** — Merged Fake and Real datasets with labels
2. **Text Cleaning** — Lowercasing, URL removal, punctuation removal, stopword handling
3. **EDA** — Class distribution, article length, subject analysis
4. **Feature Extraction** — TF-IDF Vectorizer (50,000 features, bigrams)
5. **Model Training** — 5 models compared: Logistic Regression, Naive Bayes, Decision Tree, Random Forest, Linear SVM
6. **Evaluation** — Accuracy, Precision, Recall, F1-Score, Confusion Matrix
7. **Deployment** — Streamlit web app for real-time prediction

## Models Trained
| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~98.6% |
| Naive Bayes | ~94.4% |
| Decision Tree | ~99.7% |
| Random Forest | ~99.1% |
| Linear SVM | ~99.5% |

## Project Structure
```
fake_news_detection/
│
├── fake_news_detection.ipynb   # Main Jupyter Notebook
├── app.py                      # Streamlit Web App
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── Fake.csv                    # Fake news dataset
├── True.xlsx                   # Real news dataset
├── fake_news_model.pkl         # Saved best model (generated after running notebook)
└── tfidf_vectorizer.pkl        # Saved TF-IDF vectorizer (generated after running notebook)
```

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Jupyter Notebook
```bash
jupyter notebook fake_news_detection.ipynb
```
> Run all cells top to bottom. This will train the models and generate `fake_news_model.pkl` and `tfidf_vectorizer.pkl`.

### 3. Launch the Streamlit App
```bash
streamlit run app.py
```

## Expected Output
- Trained model files (`.pkl`)
- Classification report with 98%+ accuracy
- Visualizations: class distribution, model comparison, confusion matrix
- Live Streamlit web app for news classification

## Future Enhancements
- Deep Learning models (LSTM, BERT)
- Real-time browser extension
- Multilingual fake news detection
- Integration with social media APIs
