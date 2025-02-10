from flask import Flask, render_template, request, jsonify
import requests
from gtts import gTTS
from deep_translator import GoogleTranslator
from transformers import pipeline
from IPython.display import Audio, display  # Used for testing in notebooks

app = Flask(__name__)

# Set Groq API key (replace with your actual key)
GROQ_API_KEY = ""

# Language mapping
LANGUAGE_CODES = {
    "hindi": "hi",
    "telugu": "te",
    "marathi": "mr",
    "gujarati": "gu"
}

# Hugging Face Translation Pipeline
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-mul")

def translate_with_google(text, target_language):
    """Translates using Google Translator API."""
    try:
        return GoogleTranslator(source="en", target=LANGUAGE_CODES[target_language]).translate(text)
    except Exception as e:
        print(f"Google Translation failed: {e}")
        return None

def translate_with_huggingface(text):
    """Translates using Hugging Face Transformer model."""
    try:
        return translator(text)[0]['translation_text']
    except Exception as e:
        print(f"Hugging Face Translation failed: {e}")
        return None

def translate_with_groq(text, target_language):
    """Translates English text using Groq API."""
    url = "https://api.groq.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"Translate this English sentence to {target_language}: {text}"

    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are a professional translator."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        print("Groq API Error:", response.json())
        return None

def get_translation(text, target_language):
    """Gets translation using Google Translator, Hugging Face, and Groq API."""
    # Try Google Translator first
    translation = translate_with_google(text, target_language)
    if translation:
        return translation



@app.route("/")
def home():
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate():
    data = request.json
    text = data.get("text", "")
    target_language = data.get("language", "").lower()

    if target_language not in LANGUAGE_CODES:
        return jsonify({"error": "Invalid language selected"}), 400

    # Perform translation using available methods
    translated_text = get_translation(text, target_language)

    if translated_text:
        audio_path = text_to_speech(translated_text, LANGUAGE_CODES[target_language])
        return jsonify({"translated_text": translated_text, "audio": audio_path})
    else:
        return jsonify({"error": "Translation failed"}), 500

if __name__ == "__main__":
    app.run(debug=True)



