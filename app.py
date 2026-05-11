import os
import sys
import subprocess

# Auto-install dependencies if they are missing
try:
    import streamlit as st
    import joblib
    import spacy
    import re
    import html
    import nltk
    import plotly.graph_objects as go
except ImportError as e:
    print(f"Missing dependency detected: {e}. Auto-installing from requirements.txt...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully! Reloading imports...")
        import streamlit as st
        import joblib
        import spacy
        import re
        import nltk
        import plotly.graph_objects as go
    except Exception as install_error:
        print(f"Failed to install dependencies automatically: {install_error}")
        print("Please install them manually using: pip install -r requirements.txt")
        sys.exit(1)

try:
    from nltk.corpus import stopwords
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    stop_words = set(stopwords.words('english'))
    sia = SentimentIntensityAnalyzer()
except Exception:
    nltk.download('stopwords')
    nltk.download('vader_lexicon')
    from nltk.corpus import stopwords
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    stop_words = set(stopwords.words('english'))
    sia = SentimentIntensityAnalyzer()

nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="Mental Health Text Risk Detector", layout="wide")

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    .stApp {
        background-color: #f4f6f8;
        font-family: 'Inter', sans-serif;
    }
    
    /* Input visual separation */
    .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02) !important;
    }
    
    /* Centralize main container */
    .block-container {
        max-width: 900px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        color: #1a1a1a;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        font-size: 2.2rem;
        font-weight: 600;
    }
    .sub-header {
        text-align: center;
        color: #6c757d;
        margin-bottom: 3rem;
        font-size: 1rem;
    }

    /* Divider */
    .divider-container {
        display: flex;
        align-items: center;
        text-align: center;
        color: #adb5bd;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        margin-bottom: 2rem;
        margin-top: 2rem;
    }
    .divider-container::before, .divider-container::after {
        content: '';
        flex: 1;
        border-bottom: 1px solid #dee2e6;
    }
    .divider-container::before { margin-right: 1.5em; }
    .divider-container::after { margin-left: 1.5em; }

    /* Result Main Card */
    .result-main-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 2.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .result-label {
        color: #5c7c9a;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
    }
    .result-title {
        color: #0b2136;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 1rem;
        letter-spacing: -0.5px;
    }
    .result-desc {
        color: #495057;
        font-size: 1.05rem;
        line-height: 1.6;
        max-width: 85%;
    }
    
    /* Custom Progress Bars */
    .progress-label {
        font-size: 0.9rem;
        color: #212529;
        font-weight: 500;
        margin-bottom: 0.3rem;
        display: flex;
        justify-content: space-between;
    }
    .progress-bar-bg {
        background-color: #e2e8f0;
        border-radius: 10px;
        height: 10px;
        width: 100%;
        margin-bottom: 1.5rem;
        overflow: hidden;
    }
    .progress-bar-fill {
        background: linear-gradient(90deg, #38bdf8 0%, #0ea5e9 100%);
        height: 100%;
        border-radius: 10px;
    }

    /* Support Card */
    .support-card {
        background: #fff5f5;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #ffe3e3;
    }
    .support-title {
        color: #e03131;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 1px;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        text-transform: uppercase;
    }
    .support-box {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid #ffe3e3;
    }
    .support-box-label {
        font-size: 0.75rem;
        color: #868e96;
        margin-bottom: 0.2rem;
    }
    .support-box-val {
        font-size: 1.3rem;
        font-weight: 700;
        color: #212529;
    }
    .support-link {
        font-size: 0.8rem;
        color: #0056b3;
        text-decoration: none;
        font-weight: 600;
        display: block;
        margin-top: 1rem;
    }
    
    /* About Card */
    .about-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e9ecef;
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
    }
    .about-text h4 {
        margin: 0 0 0.5rem 0;
        font-size: 0.95rem;
        color: #212529;
    }
    .about-text p {
        margin: 0;
        font-size: 0.8rem;
        color: #6c757d;
        line-height: 1.5;
    }
    
    /* Hide Streamlit stuff */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("<div style='font-size: 0.85rem; color: #6c757d; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 0.5rem;'>Mental Health Text Risk Detector</div>", unsafe_allow_html=True)
st.markdown("<h1 class='main-header' style='margin-top: 0;'>How are you feeling today?</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Paste text for instant AI emotional analysis.</p>", unsafe_allow_html=True)



@st.cache_resource
def load_model():
    if os.path.exists("models/model.joblib"):
        return joblib.load("models/model.joblib")
    return None

model = load_model()

if model is None:
    st.error("Model file not found at `models/model.joblib`. Please run the training script before using this app.")

def clean_text(text):
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    

        
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Use session state to control view
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "user_region" not in st.session_state:
    st.session_state.user_region = "India"

def perform_analysis(text):
    if not model:
        st.error("Model not found. Please run the training script first.")
        return False
    
    cleaned = clean_text(text)
    if not cleaned.strip():
        st.warning("Please enter valid text for analysis. Numbers and URLs alone cannot be analyzed.")
        return False

    raw_prediction = model.predict([cleaned])[0]
    probabilities = model.predict_proba([cleaned])[0]
    prob_dict = dict(zip(model.classes_, probabilities))
    
    # 1. Extreme Crisis Keyword Bypass (Immediate Escalation)
    # This dictionary is enriched by the top statistical feature weights natively learned by the core TF-IDF model
    CRISIS_KEYWORDS = {
        "suicide", "suicidal", "kill myself", "end my life", "don't want to live",
        "can't go on", "no reason to live", "end it all", "hurt myself",
        "overdose", "gun", "rope", "wrists", "painless", "im gone", "want end",
        "raped", "molested", "sodomized", "abused", "sexually assault",
        "bomb", "detonate", "kill someone", "murder", "take my life", "taking my life"
    }
    
    has_crisis_keyword = any(kw in text.lower() for kw in CRISIS_KEYWORDS)
    
    # Custom Confidence Thresholds to reduce false positives
    # Demands a higher statistical burden of proof for severe labels
    if prob_dict.get("High Risk (SW)", 0) >= 0.65:
        prediction = "High Risk (SW)"
    elif prob_dict.get("Depression", 0) >= 0.55:
        prediction = "Depression"
    elif prob_dict.get("Anxiety", 0) >= 0.50:
        prediction = "Anxiety"
    elif prob_dict.get("Mental Health", 0) >= 0.45:
        prediction = "Mental Health"
    elif prob_dict.get("Loneliness", 0) >= 0.40:
        prediction = "Loneliness"
    else:
        prediction = "Other"
    
    # Syntactic Analysis using SpaCy
    doc = nlp(text)
    themes = list(set([chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) < 4]))[:5]
    actions = list(set([token.lemma_.lower() for token in doc if token.pos_ == "VERB" and not token.is_stop and token.is_alpha]))[:5]
    
    st.session_state.analysis_results = {
        "original_text": text,
        "prediction": prediction,
        "probabilities": dict(zip(model.classes_, probabilities)),
        "syntax": {"themes": themes, "actions": actions},
        "has_crisis_keyword": has_crisis_keyword
    }
    return True

if st.session_state.analysis_results is None:
    # Input View
    _, col_reg = st.columns([3, 1])
    with col_reg:
        selected_region = st.selectbox(
            "Region (For Support Lines)",
            ["United States", "United Kingdom", "Canada", "Australia", "India", "International"],
            index=["United States", "United Kingdom", "Canada", "Australia", "India", "International"].index(st.session_state.user_region),
            label_visibility="visible"
        )
        st.session_state.user_region = selected_region
    
    st.write("")

    st.markdown("**Text Analysis**")
    user_input = st.text_area("Input", height=200, label_visibility="collapsed", placeholder="Describe your thoughts or paste content here...", max_chars=2000, help="Maximum 2000 characters.")
        
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        st.write("") # Spacing
        if st.button("Analyze Mood", use_container_width=True, type="primary"):
            with st.spinner("Analyzing emotional markers..."):
                import time
                time.sleep(0.5) # Provide slight visual feedback
                success = False
                if user_input:
                    success = perform_analysis(user_input)
                else:
                    st.warning("Please provide text for analysis.")
                
                if success:
                    st.rerun()
                
else:
    # Results View
    st.markdown("<div class='divider-container'>LATEST RESULTS</div>", unsafe_allow_html=True)
    
    res = st.session_state.analysis_results
    
    # Display Original Text
    st.markdown("### Analyzed Text")
    safe_text = html.escape(res.get('original_text', ''))
    st.markdown(f"<div style='background-color: #ffffff; padding: 1rem; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 1.5rem; color: #475569; font-style: italic;'>\"{safe_text}\"</div>", unsafe_allow_html=True)

    pred = res["prediction"]
    # Normalize labels from the new model
    if pred == "High Risk (SW)":
        pred = "High Risk"
    
    probs = {k.replace("High Risk (SW)", "High Risk"): v for k, v in res["probabilities"].items()}
    max_prob = max(probs.values()) * 100
    has_crisis_keyword = res.get("has_crisis_keyword", False)
    
    # Supportive Messages Dictionary
    import random
    support_msgs = {
        "High Risk": [
            "We hear how much pain you're in. Please know that there are people who want to support you through this dark moment.",
            "You don't have to carry this burden alone. Reaching out to a crisis line can provide immediate, confidential support.",
            "Your life has value, even when it feels overwhelmingly difficult. Please connect with a professional who can help right now."
        ],
        "Depression": [
            "It takes courage to acknowledge these heavy feelings. Remember that depression is treatable, and brighter days are possible.",
            "Please remember to be gentle with yourself. Taking small steps and seeking professional guidance can make a big difference.",
            "You are not your thoughts. Acknowledging this emotional fatigue is the first step toward healing and finding support."
        ],
        "Anxiety": [
            "It's completely okay to feel overwhelmed. Remember to take deep breaths and focus on what you can control right now.",
            "Anxiety can feel isolating, but you are not alone. Consider reaching out to someone you trust or exploring grounding techniques.",
            "Your feelings are valid. Take things one moment at a time, and consider talking to a professional about managing this stress."
        ],
        "Loneliness": [
            "Feeling disconnected is a very human experience. There are communities and professionals ready to listen when you're ready to reach out.",
            "Isolation can be incredibly heavy. Please consider connecting with a support group or a friend, even a small connection helps.",
            "You matter, and your presence is valued. Taking the step to connect with others, even virtually, can help bridge the gap."
        ],
        "Mental Health": [
            "It takes strength to reflect on your well-being. Remember that reaching out for support is a sign of courage.",
            "Masking your true feelings can be exhausting. Finding a safe space to talk openly can help lift that weight.",
            "Your emotional health is deeply important. Please consider connecting with someone you trust."
        ],
        "Mixed": [
            "Sometimes emotions are hard to untangle. Taking a step back to reflect on how you're feeling can be incredibly helpful.",
            "You don't need a specific label to validate your feelings. If you're feeling weighed down, it's always okay to seek support.",
            "Mixed feelings often mean you are carrying a lot at once. Make sure to prioritize balance and self-care right now."
        ],
        "Mild": [
            "It's normal to experience ups and downs. Make sure to prioritize self-care and take time to recharge.",
            "Everyone has difficult moments. Taking a step back and practicing self-compassion can help you navigate this stress.",
            "Your feelings are completely valid. Don't hesitate to lean on your support system when everyday stress feels a bit too heavy."
        ],
        "Normal": [
            "Continuing to check in with yourself is a great habit for long-term emotional well-being.",
            "Remember that maintaining mental health is an ongoing journey. Keep prioritizing your self-care routines.",
            "Your emotional awareness is a great strength. Continue to prioritize balance and well-being in your daily life."
        ]
    }

    # Main Card
    is_mild = False
    is_mixed = False

    if pred in ["High Risk", "Depression", "Anxiety", "Loneliness", "Mental Health"]:
        if max_prob < 35:
            is_mild = True
            title = "Mild Emotional Indicators"
            desc = (
                f"It looks like there may be some {pred.lower()} in what you've shared, "
                f"though the signal is faint ({max_prob:.1f}% confidence). "
                f"Moments like these are a natural part of life. "
                f"If things feel heavier than usual, talking to someone can help."
            )
        else:
            title = "We Hear You" if pred != "High Risk" else "You Don't Have to Face This Alone"
            desc = (
                f"What you've shared reflects some feelings associated with {pred.lower()}. "
                f"That's okay — recognizing it is already a meaningful step. "
                f"Reaching out to someone you trust, or a professional, "
                f"can make a real difference."
            )
    else:
        risk_mass = sum(probs.get(k, 0) for k in ["High Risk", "Depression", "Anxiety", "Loneliness", "Mental Health"])
        if risk_mass > 0.65:
            is_mixed = True
            title = "Something Worth Acknowledging"
            desc = (
                "There's a quiet heaviness in what you've shared — nothing alarming, "
                "but enough to be worth sitting with. "
                "If it's been weighing on you, speaking with someone "
                "you trust can bring some relief."
            )
        else:
            title = "You Seem to Be Doing Okay"
            desc = (
                "Nothing in what you've shared points to significant distress. "
                "That said, it's always good to check in with yourself regularly. "
                "You know your own feelings best."
            )
    
    # Select randomized supportive message based on category
    if is_mild:
        msg_category = "Mild"
    elif is_mixed:
        msg_category = "Mixed"
    elif pred in support_msgs:
        msg_category = pred
    else:
        msg_category = "Normal"
        
    support_quote = random.choice(support_msgs[msg_category])
    
    banner_html = ""
    if has_crisis_keyword:
        banner_html = (
            "<div style='background-color: #fef2f2; border-left: 4px solid #ef4444; padding: 1.5rem; margin-bottom: 2rem; border-radius: 8px;'>"
            "<strong style='color: #991b1b; font-size: 1.1rem; display: block; margin-bottom: 0.5rem;'>Emergency Lifeline Alert</strong>"
            "<span style='color: #7f1d1d; font-size: 0.95rem; line-height: 1.5;'>Our system detected language associated with extreme clinical distress or danger. If you or someone you know is in immediate danger, please step away from this application and dial emergency services or a dedicated crisis lifeline immediately.</span>"
            "</div>"
        )
    
    st.markdown(
        f"<div class='result-main-card'>\n"
        f"{banner_html}\n"
        f"<div class='result-label'>Final Analysis Result</div>\n"
        f"<div class='result-title'>{title}</div>\n"
        f"<div class='result-desc'>{desc}</div>\n"
        f"<div style='margin-top: 1.5rem; padding: 1rem; background-color: #f8f9fa; border-left: 4px solid #4cc9f0; border-radius: 4px; font-style: italic; color: #495057;'>\n"
        f"\"{support_quote}\"\n"
        f"</div>\n"
        f"</div>", 
        unsafe_allow_html=True
    )
    
    # Secondary Row
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("### Confidence Breakdown")
        st.write("")
        
        # Dynamically show all probabilities sorted by confidence
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        for raw_label, val_dec in sorted_probs:
            val = val_dec * 100
            
            # Map labels to friendly UI text
            if raw_label in ["Normal", "Other"]:
                label = "Normal / No Concern"
            elif raw_label == "High Risk":
                label = "High Risk / Severity"
            elif raw_label == "Mild":
                label = "Mild Stress Indicators"
            else:
                label = f"{raw_label} Indicators"
                
            st.markdown(f"""
            <div class='progress-label'>
                <span>{label}</span>
                <span>{int(val)}%</span>
            </div>
            <div class='progress-bar-bg'>
                <div class='progress-bar-fill' style='width: {val}%'></div>
            </div>
            """, unsafe_allow_html=True)
            
    with col2:
        st.markdown("### Risk Profile")
        
        # Risk Severity Gauge using Plotly
        risk_prob = max(probs.get("Anxiety", 0), probs.get("Depression", 0), probs.get("High Risk", 0), probs.get("Loneliness", 0), probs.get("Mental Health", 0)) * 100
        
        if risk_prob > 75:
            severity_text = "Critical"
            color = "#dc2626" # Deep Red
        elif risk_prob > 50:
            severity_text = "High"
            color = "#ea580c" # Dark Orange
        elif risk_prob > 30:
            severity_text = "Medium"
            color = "#eab308" # Amber/Yellow
        else:
            severity_text = "Low"
            color = "#10b981" # Emerald Green
            
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_prob,
            number = {'suffix': "%", 'font': {'size': 24, 'color': '#1a1a1a'}},
            title = {'text': severity_text, 'font': {'size': 28, 'color': color, 'weight': 'bold'}},
            gauge = {
                'axis': {'range': [0, 100], 'visible': False},
                'bar': {'color': color, 'thickness': 0.25},
                'bgcolor': "#e2e8f0",
                'borderwidth': 0,
            }
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'family': 'Inter'})
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        if severity_text in ["Critical", "High"]:
            gauge_subtext = "Detection of urgency markers and severe emotional fatigue suggests a need for <b>immediate supportive intervention</b>."
        elif severity_text == "Medium":
            gauge_subtext = "Presence of distress indicators suggests a need to <b>monitor emotional well-being</b> and consider reaching out to a support system."
        else:
            gauge_subtext = "No immediate urgency markers detected. Always remember to prioritize <b>routine self-care</b> and mental wellness."
            
        if is_mixed:
            display_pred = "Mixed Emotional Distress"
        elif pred == "Other":
            display_pred = "Normal / No Concern"
        else:
            display_pred = pred
            
        st.markdown(f"""
        <div style='text-align: center; margin-bottom: 0.75rem;'>
            <span style='background-color: #f1f5f9; border: 1px solid #e2e8f0; padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; color: #475569; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase;'>
                Primary Indicator: {display_pred}
            </span>
        </div>
        <div style='text-align: center; font-size: 0.85rem; color: #495057; padding: 0 1rem;'>
        {gauge_subtext}
        </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    
    # Bottom Row Actions
    col3, col4 = st.columns([1.5, 1])
    
    with col3:
        if st.button("Start New Analysis", use_container_width=True, type="primary"):
            st.session_state.analysis_results = None
            st.rerun()
            
        st.markdown("""
        <div class='about-card'>
            <div class='about-text'>
                <h4>About our AI Analysis</h4>
                <p>These results are generated using natural language processing models. While highly indicative, they do not constitute a clinical diagnosis. We recommend discussing these findings with a qualified mental health professional.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        support_numbers = {
            "United States": {"call": "988", "text_label": "Text 'HOME' to", "text_val": "741741"},
            "United Kingdom": {"call": "111", "text_label": "Text 'SHOUT' to", "text_val": "85258"},
            "Canada": {"call": "988", "text_label": "Text 'HOME' to", "text_val": "686868"},
            "Australia": {"call": "13 11 14", "text_label": "Text Lifeline to", "text_val": "0477 13 11 14"},
            "India": {"call": "9152987821", "text_label": "AASRA Helpline", "text_val": "24x7"},
            "International": {"call": "Emergency", "text_label": "Online Support", "text_val": "FindAHelpline"}
        }
        region_data = support_numbers.get(st.session_state.user_region, support_numbers["India"])
        
        st.markdown(f"""
        <div class='support-card'>
            <div class='support-title'>IMMEDIATE SUPPORT</div>
            <div class='support-box'>
                <div>
                    <div class='support-box-label'>Crisis Line ({st.session_state.user_region})</div>
                    <div class='support-box-val'>{region_data['call']}</div>
                </div>
            </div>
            <div class='support-box'>
                <div>
                    <div class='support-box-label'>{region_data['text_label']}</div>
                    <div class='support-box-val'>{region_data['text_val']}</div>
                </div>
            </div>
            <a href='https://findahelpline.com/' target='_blank' class='support-link'>Find local clinical teams →</a>
        </div>
        """, unsafe_allow_html=True)

# Footer / About Project section at the bottom of the page
st.write("---")
st.markdown("<h3 style='text-align: center; color: #495057; font-size: 1.2rem; margin-bottom: 1.5rem;'>About the Project</h3>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("#### The Mission")
    st.markdown("<p style='font-size: 0.85rem; color: #6c757d;'>An AI-powered tool designed to identify emotional markers and potential indicators of anxiety, depression, or high-risk thoughts from text and images.</p>", unsafe_allow_html=True)
with col2:
    st.markdown("#### Key Features")
    st.markdown("<p style='font-size: 0.85rem; color: #6c757d;'>• NLP Text Analysis<br>• Real-time Assessment<br>• Local Support Links</p>", unsafe_allow_html=True)
with col3:
    st.markdown("#### Technology Stack")
    st.markdown("<p style='font-size: 0.85rem; color: #6c757d;'>`Python` `Streamlit` `Scikit-Learn`<br>`SpaCy` `NLTK` `Plotly`</p>", unsafe_allow_html=True)

st.markdown("<div style='text-align: center; margin-top: 2rem; color: #adb5bd; font-size: 0.75rem;'>Disclaimer: This tool is for educational purposes only and is not a substitute for professional medical advice.</div>", unsafe_allow_html=True)
