# Predicting Creativity from Text Using NLP and RAG

## Overview
This project investigates whether creativity can be predicted from text alone, without using explicit personality labels or psychological test outputs at inference time. The core idea is to model writing style and linguistic patterns, then combine classical machine learning with retrieval-augmented context.

The pipeline analyzes how a person writes and classifies the text as **Creative** or **Not Creative**, together with a confidence score.

## Project Goal
The main research question is:

Can AI detect creativity from natural language writing behavior?

To answer this, the notebook builds a complete end-to-end workflow that includes:

- Data loading and preprocessing
- Text cleaning and NLP feature engineering
- Exploratory data analysis (EDA)
- Model training and evaluation
- Retrieval-Augmented Generation (RAG) support using vector search
- Live prediction on user-provided text

## Data Sources
This project uses two real-world datasets:

- **MBTI personality posts dataset** (social-media-style text posts)
- **Big Five personality test dataset** (950,000+ responses)

The MBTI text data is used to extract linguistic features from writing. The Big Five data is used to derive openness-related creativity signals and support large-scale behavioral grounding.

## RAG Component
The retrieval pipeline is built with:

- **Sentence Transformers** for text embeddings
- **ChromaDB** for vector storage and nearest-neighbor retrieval

Given an input text, the RAG layer retrieves similar real personality profiles. This adds contextual evidence that complements pure feature-based classification.

## Machine Learning Models
The notebook trains and compares five models:

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

In this project version, Logistic Regression provides the best overall balance between predictive performance and generalization on unseen text.

## Repository Structure
- `NLP_RAG.ipynb`: Main notebook with the full pipeline
- `mbti_1/mbti_1.csv`: MBTI dataset
- `data-final/data-final.csv`: Big Five dataset (tab-separated)
- `data-final/codebook.txt`: Data dictionary / codebook

## How to Run
1. Create and activate a Python environment.
2. Install required packages used in the notebook (for example: numpy, pandas, matplotlib, seaborn, scikit-learn, xgboost, lightgbm, sentence-transformers, chromadb).
3. Open and run `NLP_RAG.ipynb` from top to bottom.
4. Use the final prediction cell to test your own text.

## Output
For each input text sample, the system returns:

- Predicted class: **Creative** or **Not Creative**
- Confidence score
- Retrieval-backed context from similar profiles (RAG)

## Notes
- This project is for research and educational purposes.
- Creativity is a complex human trait; results should be interpreted as probabilistic signals, not absolute judgments.
