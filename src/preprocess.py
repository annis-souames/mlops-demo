import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import re


# Builds some features from the text
def build_features(df):
    # DataFrame cleanup
    df["text"] = df.title + " " + df.description  # feature engineering
    df["text"] = df.text.apply(clean_text)  # clean text
    df = df.drop(columns=["id", "created_on", "title", "description"], errors="ignore")  # drop cols
    df = df.dropna(subset=["tag"])  # drop nulls
    df = df[["text", "tag"]]  # rearrange cols
    print("Cleaned dataframe, only these columns are left: ", df.columns)
    return df

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

# Tokenize the text, we will us ethe BERT tokenizer
def tokenize(batch):
    tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False)
    encoded_inputs = tokenizer(batch["text"].tolist(), return_tensors="np", padding="longest")
    return dict(ids=encoded_inputs["input_ids"], masks=encoded_inputs["attention_mask"], targets=np.array(batch["tag"]))


def preprocess(df, class_to_index):
    df = build_features(df)
    df = df[["text", "tag"]]  # rearrange columns
    df["tag"] = df["tag"].map(class_to_index)  # label encoding
    outputs = tokenize(df)
    return outputs


# Split data into train and validation and save them to disk if save param is True
def split_save_data(df, save=False, val_size=0.2):
    train, val = train_test_split(df, test_size=val_size)
    if save:
        train.to_csv("data/train.csv", index=False)
        val.to_csv("data/val.csv", index=False)
    return train, val
