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
The model is trained on dataset obtained from Kaggle ([Dataset Link](https://www.kaggle.com/datasets/entenam/reddit-mental-health-dataset)). Reddit provides a rich source of anonymous, highly expressive text where users openly discuss psychological struggles. The dataset contains posts categorized by conditions such as depression, anxiety, stress, and neutral control groups. Given the unstructured nature of social media text, the data undergoes rigorous preprocessing to handle slang, abbreviations, and noise.

## System Architecture

### 1. Data Preprocessing
Raw text is cleaned and normalized. This includes removing URLs, punctuation, and special characters, converting text to lowercase, and filtering out non-informative stopwords. Lemmatization is applied to reduce words to their base forms, enabling the model to capture semantic similarities.

### 2. Natural Language Processing
The system utilizes advanced NLP techniques to analyze sentence structure and meaning. This includes tokenization, part-of-speech tagging, and dependency parsing to move beyond simple keyword matching and understand deeper linguistic behaviors.

### 3. Transformer-based NLP Architecture
Instead of traditional bag-of-words or TF-IDF approaches, the system utilizes a **Transformer-based deep learning architecture**. Specifically, it uses a fine-tuned `DistilBERT` model provided by Hugging Face. The self-attention mechanisms within DistilBERT allow the model to understand the deep contextual meaning of the text, capturing nuances that simpler models might miss.

### 4. Deep Learning Classification & Hugging Face Integration
The DistilBERT model was fine-tuned on the Reddit dataset to classify text into distinct emotional risk categories. To ensure the application remains lightweight and easy to deploy (such as on Streamlit Community Cloud), the 267MB+ model weights are hosted externally on the **Hugging Face Model Hub**. At runtime, the Streamlit app securely fetches and caches the model directly from Hugging Face via the `transformers` pipeline.

### 5. Application Interface
The final predictive model is integrated into a Streamlit web application. Users can input text to receive an instant analysis, including a categorized risk profile, confidence breakdown, and automated detection of severe crisis indicators.

## How to Run Locally

To run the application on your own machine, ensure you have Python 3.8+ installed.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gitarju/Mental-Health-Text-Risk-Detector-Using-DistilBERT.git
   cd Mental-Health-Text-Risk-Detector-Using-DistilBERT
   ```

2. **Install the required Python packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the SpaCy English language model:**
   This is required for the NLP syntactic analysis step.
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Launch the Streamlit web application:**
   ```bash
   streamlit run app.py
   ```
   *Note: On the first run, the app will automatically download the 267MB DistilBERT model weights from the Hugging Face Hub. This may take a minute depending on your internet connection.*

## Ethical Considerations & Disclaimer
This system is designed strictly for educational, research, and awareness purposes. **It is not a medical diagnostic tool.** Machine learning models can produce false positives and false negatives, and they struggle with complex nuances such as sarcasm or cultural variations in emotional expression. Predictions made by this software should never replace professional psychological assessment or clinical judgment.
