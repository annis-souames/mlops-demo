import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re


# Builds some features from the text
def build_features(df):
    pass

# Clean the text from stop words and punctuation using nltk
def clean_text(text):
    # Lower
    text = text.lower()
    # Remove stopwords
    pattern = re.compile(r'\b(' + r"|".join(stopwords) + r")\b\s*")
    text = pattern.sub('', text)
    # Spacing and filters
    text = re.sub(r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text)  # add spacing
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = text.strip()  # strip white space at the ends
    text = re.sub(r"http\S+", "", text)  #  remove links

    return text

def preprocess(df, index):
    pass
