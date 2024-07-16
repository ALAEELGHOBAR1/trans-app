import base64
import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os
import nltk
import langdetect

# Enable wide mode
st.set_page_config(layout="wide")

# Caching encoded images to avoid loading them multiple times
@st.cache_data
def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Paths to your local images
image_path_left = "images/logo.png"
image_path_right = "images/logo11.png"
background_image_path = "images/back.png"
encoded_image_left = get_base64_encoded_image(image_path_left)
encoded_image_right = get_base64_encoded_image(image_path_right)
encoded_background_image = get_base64_encoded_image(background_image_path)

# Custom CSS to modify text area, background, button, select box colors, and title style
st.markdown(
    f"""
    <style>
    .stTextArea textarea {{
        background-color: #FFFFFF;  /* Set text area background to white */
        color: #0766AD;  /* Adjust text color if needed */
    }}
    .stApp {{
        background: url("data:image/jpg;base64,{encoded_background_image}") no-repeat center center fixed;
        background-size: cover;
    }}
    .stButton button {{
        background-color: #FFFFFF;
        color: #0766AD;  /* Adjust text color if needed */
        border: 2px solid #0766AD;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        transition: background-color 0.3s, color 0.3s, transform 0.3s;
    }}
    .stButton button:hover {{
        background-color: #0766AD;
        color: #FFFFFF;  /* Inverse colors on hover */
        transform: scale(1.05);
    }}
    .stSelectbox select {{
        background-color: #FFFFFF;  /* Set select box background to white */
        color: #0766AD;  /* Adjust text color if needed */
    }}
    .title {{
        text-align: center;
        color: #0766AD;
        font-size: 2.5em; /* Adjust the font size if needed */
        margin-bottom: 230px; /* Add space below the title */
    }}
    .label {{
        font-weight: bold;
        font-size: 1.2em; /* Increase font size */
        color: #0766AD; /* Set the same blue color as the title */
        margin-bottom: 0px;
    }}
    .logo-container {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 30px;
    }}
    .logo-left {{
        width: 300px; /* Adjust the width as needed */
        height: auto;
    }}
    .logo-right {{
        width: 200px; /* Adjust the width as needed */
        height: auto;
    }}
    .spacer {{
        height: 20px;  /* Adjust the height to create space */
    }}
    @media (max-width: 768px) {{
        .logo-container {{
            flex-direction: row; /* Ensure logos are horizontal */
            justify-content: space-between; /* Keep logos separated */
        }}
        .logo-left {{
            width: 100px;  /* Adjust the width for smaller screens */
        }}
        .logo-right {{
            width: 100px;  /* Adjust the width for smaller screens */
        }}
        .title {{
            font-size: 1.5em; /* Adjust the font size for smaller screens */
        }}
        .stTextArea textarea, .stSelectbox select {{
            font-size: 0.9em; /* Adjust the font size for smaller screens */
        }}
        .stButton button {{
            font-size: 0.9em; /* Adjust the font size for smaller screens */
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Add the logos in a container
st.markdown(
    f"""
    <div class="logo-container">
        <img class="logo-left" src="data:image/png;base64,{encoded_image_left}">
        <h1 class="title">Translation App</h1>
        <img class="logo-right" src="data:image/png;base64,{encoded_image_right}">
    </div>
    """,
    unsafe_allow_html=True
)

# Labels with the new style
st.markdown('<p class="label">Enter text to translate</p>', unsafe_allow_html=True)
user_text = st.text_area("", value=st.session_state.get('user_text', ''))

st.markdown('<p class="label">Select destination language</p>', unsafe_allow_html=True)
dest_lang = st.selectbox("", ["en", "fr", "ar"])

# Adjust margin for the "Traduire" button
st.markdown("<style>.stButton {margin-top: 20px;}</style>", unsafe_allow_html=True)

# Download nltk punkt tokenizer
nltk.download('punkt')

# Language codes for NLLB and MarianMT
nllb_language_codes = {
    'en': 'eng_Latn',
    'fr': 'fra_Latn',
    'ar': 'arb_Arab'
}

marian_language_codes = {
    'en': 'en',
    'fr': 'fr',
    'ar': 'ar'
}

# Load the NLLB model
@st.cache_resource(show_spinner=False)
def load_nllb_model():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    translation_pipeline = pipeline("translation", model=model, tokenizer=tokenizer)
    return translation_pipeline

# Load the MarianMT model
@st.cache_resource(show_spinner=False)
def load_marian_model(src_lang, dest_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{dest_lang}"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    translation_pipeline = pipeline("translation", model=model, tokenizer=tokenizer)
    return translation_pipeline

# Initialize the NLLB model
nllb_pipeline = load_nllb_model()

# Initialize the CSV file
csv_file = 'translations.csv'

# Function to save translations to CSV
def save_to_csv(data, file):
    try:
        if not os.path.isfile(file):
            df = pd.DataFrame(data)
            df.to_csv(file, index=False)
        else:
            df = pd.DataFrame(data)
            df.to_csv(file, mode='a', header=False, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving to CSV: {e}")
        return False

# Function to load and display CSV content
def load_csv(file):
    if os.path.isfile(file):
        df = pd.read_csv(file)
        return df
    else:
        return pd.DataFrame()

# Initialize session state variables
if 'user_text' not in st.session_state:
    st.session_state.user_text = ''
if 'translated_text_nllb' not in st.session_state:
    st.session_state.translated_text_nllb = ''
if 'translated_text_marian' not in st.session_state:
    st.session_state.translated_text_marian = ''
if 'src_lang' not in st.session_state:
    st.session_state.src_lang = 'en'
if 'dest_lang_iso' not in st.session_state:
    st.session_state.dest_lang_iso = ''

# Center the "Traduire" button
if st.button("Traduire"):
    if user_text:
        st.session_state.user_text = user_text  # Store user text in session state
        src_lang = langdetect.detect(user_text)
        st.session_state.src_lang = src_lang
        st.write(f"Detected source language: {src_lang}")

        src_lang_nllb = nllb_language_codes.get(src_lang)
        dest_lang_nllb = nllb_language_codes.get(dest_lang)
        src_lang_marian = marian_language_codes.get(src_lang)
        dest_lang_marian = marian_language_codes.get(dest_lang)

        if not src_lang_nllb or not dest_lang_nllb:
            st.write("Unsupported language code.")
        else:
            # Perform translations based on source and destination language pairs
            if src_lang == 'ar' and dest_lang == 'fr':
                marian_pipeline = load_marian_model(src_lang_marian, dest_lang_marian)
                translation_marian = marian_pipeline(user_text)
                st.session_state.translated_text_marian = translation_marian[0]['translation_text']
                st.write(f"MarianMT Translation: {st.session_state.translated_text_marian}")

            elif src_lang == 'ar' and dest_lang == 'en':
                translation_nllb = nllb_pipeline(user_text, src_lang=src_lang_nllb, tgt_lang=dest_lang_nllb)
                st.session_state.translated_text_nllb = translation_nllb[0]['translation_text']
                st.write(f"NLLB Translation: {st.session_state.translated_text_nllb}")

            elif src_lang == 'fr' and dest_lang == 'ar':
                marian_pipeline = load_marian_model(src_lang_marian, dest_lang_marian)
                translation_marian = marian_pipeline(user_text)
                st.session_state.translated_text_marian = translation_marian[0]['translation_text']
                st.write(f"MarianMT Translation: {st.session_state.translated_text_marian}")

                translation_nllb = nllb_pipeline(user_text, src_lang=src_lang_nllb, tgt_lang=dest_lang_nllb)
                st.session_state.translated_text_nllb = translation_nllb[0]['translation_text']
                st.write(f"NLLB Translation: {st.session_state.translated_text_nllb}")

            elif src_lang == 'fr' and dest_lang == 'en':
                translation_nllb = nllb_pipeline(user_text, src_lang=src_lang_nllb, tgt_lang=dest_lang_nllb)
                st.session_state.translated_text_nllb = translation_nllb[0]['translation_text']
                st.write(f"NLLB Translation: {st.session_state.translated_text_nllb}")

            elif src_lang == 'en' and dest_lang == 'fr':
                marian_pipeline = load_marian_model(src_lang_marian, dest_lang_marian)
                translation_marian = marian_pipeline(user_text)
                st.session_state.translated_text_marian = translation_marian[0]['translation_text']
                st.write(f"MarianMT Translation: {st.session_state.translated_text_marian}")

                translation_nllb = nllb_pipeline(user_text, src_lang=src_lang_nllb, tgt_lang=dest_lang_nllb)
                st.session_state.translated_text_nllb = translation_nllb[0]['translation_text']
                st.write(f"NLLB Translation: {st.session_state.translated_text_nllb}")

            elif src_lang == 'en' and dest_lang == 'ar':
                marian_pipeline = load_marian_model(src_lang_marian, dest_lang_marian)
                translation_marian = marian_pipeline(user_text)
                st.session_state.translated_text_marian = translation_marian[0]['translation_text']
                st.write(f"MarianMT Translation: {st.session_state.translated_text_marian}")
    else:
        st.write("Please enter text to translate.")

# Display options to approve the translation
if st.session_state.translated_text_nllb and st.session_state.translated_text_marian:
    translation_choice = st.radio(
        "Choose the translation to save:",
        ('MarianMT', 'NLLB')
    )
    if st.button("Approve Translation"):
        chosen_translation = st.session_state.translated_text_nllb if translation_choice == 'NLLB' else st.session_state.translated_text_marian
        data = [{
            'Source Language': st.session_state.src_lang,
            'Target Language': dest_lang,
            'Original Text': st.session_state.user_text,
            'Translated Text': chosen_translation,
            'Model': translation_choice
        }]
        if save_to_csv(data, csv_file):
            st.write("Translation saved successfully!")
            st.write(f"File saved to: {os.path.abspath(csv_file)}")
            # Clear session state after saving
            st.session_state.user_text = ''
            st.session_state.translated_text_nllb = ''
            st.session_state.translated_text_marian = ''
            st.session_state.src_lang = 'en'
            st.session_state.dest_lang_iso = ''
elif st.session_state.translated_text_nllb or st.session_state.translated_text_marian:
    translation_choice = 'NLLB' if st.session_state.translated_text_nllb else 'MarianMT'
    chosen_translation = st.session_state.translated_text_nllb or st.session_state.translated_text_marian
    if st.button("Approve Translation"):
        data = [{
            'Source Language': st.session_state.src_lang,
            'Target Language': dest_lang,
            'Original Text': st.session_state.user_text,
            'Translated Text': chosen_translation,
            'Model': translation_choice
        }]
        if save_to_csv(data, csv_file):
            st.write("Translation saved successfully!")
            st.write(f"File saved to: {os.path.abspath(csv_file)}")
            # Clear session state after saving
            st.session_state.user_text = ''
            st.session_state.translated_text_nllb = ''
            st.session_state.translated_text_marian = ''
            st.session_state.src_lang = 'en'
            st.session_state.dest_lang_iso = ''

# Display CSV content with buttons
if st.button("Show CSV content", key="show_csv"):
    csv_data = load_csv(csv_file)
    if not csv_data.empty:
        st.write(csv_data)
    else:
        st.write("No data found.")

if st.button("Close CSV content", key="close_csv"):
    st.empty()
