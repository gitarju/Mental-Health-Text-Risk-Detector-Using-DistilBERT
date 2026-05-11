# Mental Health Text Risk Detector

## Project Overview
The Mental Health Text Risk Detector is a Natural Language Processing (NLP) and Machine Learning (ML) application designed to identify signs of emotional distress, depression, anxiety, and other mental health risks from text. By computationally analyzing linguistic patterns, the system detects potential risk indicators within written communication, particularly social media posts.

This project sits at the intersection of Artificial Intelligence, Data Science, and Computational Linguistics. It functions as an early-risk detection and awareness tool to assist in research and moderation systems.

## Project Team
*   Arjun
*   Athul
*   Sruthi
*   Vishnu

## Dataset
The model is trained on dataset obtained from Kaggle ([Dataset Link](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)). Reddit provides a rich source of anonymous, highly expressive text where users openly discuss psychological struggles. The dataset contains posts categorized by conditions such as depression, anxiety, stress, and neutral control groups. Given the unstructured nature of social media text, the data undergoes rigorous preprocessing to handle slang, abbreviations, and noise.

## System Architecture

### 1. Data Preprocessing
Raw text is cleaned and normalized. This includes removing URLs, punctuation, and special characters, converting text to lowercase, and filtering out non-informative stopwords. Lemmatization is applied to reduce words to their base forms, enabling the model to capture semantic similarities.

### 2. Natural Language Processing
The system utilizes advanced NLP techniques to analyze sentence structure and meaning. This includes tokenization, part-of-speech tagging, and dependency parsing to move beyond simple keyword matching and understand deeper linguistic behaviors.

### 3. Feature Extraction
Behavioral feature extraction is crucial to the system's intelligence. The application looks for self-referencing pronouns and emotionally intense vocabulary. Text is converted into numerical representations using TF-IDF (Term Frequency-Inverse Document Frequency), which assigns mathematical weight to significant distress markers.

### 4. Machine Learning Classification
The project employs supervised machine learning models to classify the text into risk categories. During training, the models learn the complex relationships between linguistic patterns and specific emotional states. The system evaluates these models using strict performance metrics, prioritizing high recall to ensure genuine distress signals are not missed.

### 5. Application Interface
The final predictive model is integrated into a Streamlit web application. Users can input text to receive an instant analysis, including a categorized risk profile, confidence breakdown, and automated detection of severe crisis indicators.

## Ethical Considerations & Disclaimer
This system is designed strictly for educational, research, and awareness purposes. **It is not a medical diagnostic tool.** Machine learning models can produce false positives and false negatives, and they struggle with complex nuances such as sarcasm or cultural variations in emotional expression. Predictions made by this software should never replace professional psychological assessment or clinical judgment.
