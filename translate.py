from transformers import MarianMTModel, MarianTokenizer
import os

# Path to the directory where cached models are stored
CACHE_DIR = "./model_cache"
DEFAULT_MODEL_NAME = "Helsinki-NLP/opus-mt-ga-en"

# Load pre-trained MarianMT model for Irish to English translation
model_name = 'Helsinki-NLP/opus-mt-ga-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def load_model_and_tokenizer():
    """
    Load the MarianMT model and tokenizer.
    If a cached model exists in `CACHE_DIR`, load it; otherwise, load the default model.

    Returns:
        MarianMTModel: The loaded translation model.
        MarianTokenizer: The tokenizer for the loaded model.
    """
    if os.path.exists(os.path.join(CACHE_DIR, "config.json")):
        # Load the cached model
        model = MarianMTModel.from_pretrained(CACHE_DIR)
        tokenizer = MarianTokenizer.from_pretrained(CACHE_DIR)
    else:
        # Load the default model
        model = MarianMTModel.from_pretrained(DEFAULT_MODEL_NAME)
        tokenizer = MarianTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    return model, tokenizer

def translate_text(text):
    """
    Translate Irish text to English using the MarianMT model.

    Args:
        text (str): The input text in Irish.

    Returns:
        str: The translated text in English.
    """
    model, tokenizer = load_model_and_tokenizer()
    # Tokenize input text
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    # Translate the text
    translated_tokens = model.generate(**inputs)
    # Decode the translated text
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text