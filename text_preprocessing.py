
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

# Initialize NLTK tools
stop_words = set(stopwords.words("english"))

# Preserve important negation words
stop_words.discard("not")
stop_words.discard("no")
stop_words.discard("nor")

lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r"\w+")

def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    # Convert text to lowercase
    text = text.lower()

    # Remove URLs and special characters
    text = re.sub(r"http\S+|www\S+", "", text)

    # Tokenize text
    words = tokenizer.tokenize(text)

    # Remove stopwords and apply lemmatization
    processed_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(processed_words)
