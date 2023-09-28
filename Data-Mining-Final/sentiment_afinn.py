# Import libraries and files
import pandas as p
import re, string, unicodedata
import contractions
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from afinn import Afinn
from string import punctuation
import json

# Create Global Objects
tokenizer = ToktokTokenizer()
ps = nltk.porter.PorterStemmer()
pattern = r"[^a-zA-z0-9\s]"
stopword_list = set(stopwords.words("english"))

# Remove Punctuation
def remove_punct(text):
    for p in punctuation:
        text = text.replace(p, "")
    return text


# Remove Special characters
def remove_special_chars(text, remove_digits=True):
    text = re.sub(pattern, "", text)
    return text


# Remove Accented Characters
def remove_accented_chars(text):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")
    return text


#  Contractions
def expand_contractions(con_text):
    con_text = contractions.fix(con_text)
    return con_text


# Remove English Stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = " ".join(filtered_tokens)
    return filtered_text


# Stem the text
def simple_stemmer(text):
    text = " ".join([ps.stem(word) for word in text.split()])
    return text


# Assign polarity score using Afinn
def afinn_sent_analysis(text1, af):
    score = af.score(text1)
    return score


# Categorize Sentiments
def afinn_sent_category(score):
    categories = ["positive", "negative", "neutral"]
    if score > 0:
        return categories[0]
    elif score < 0:
        return categories[1]
    else:
        return categories[2]


# Function for preprocessing text
def preprocess(text):
    text = remove_punct(text)
    text = remove_special_chars(text)
    text = remove_accented_chars(text)
    text = expand_contractions(text)
    return text


# Sentiment Analysis using Afinn
def analyze_sentiment(commentListWithDate, sentimentFile):
    af = Afinn()
    data = p.DataFrame(commentListWithDate, columns=["comment", "date", "polarity_vader"])
    data["word_count"] = data["comment"].apply(lambda x: len(str(x).split(" ")))
    data_clean = data.copy()
    data_clean["Comments"] = data_clean["comment"].str.lower().str.strip()

    data_clean["Comments"] = data_clean["Comments"].apply(preprocess)

    data_clean_bckup = data_clean.copy()
    data_clean["Comments_Clean"] = data_clean["Comments"].apply(remove_stopwords)
    data_clean["Normalized_Comments"] = data_clean["Comments_Clean"].apply(simple_stemmer)
    data_clean = data_clean.drop(columns=data_clean[["Comments_Clean"]], axis=1)
    data_clean = data_clean[["Comments", "Normalized_Comments", "word_count", "comment", "date", "polarity_vader"]]
    data_clean_bckup_norm = data_clean.copy()
    # data_clean.head()
    data_clean["afinn_score"] = [afinn_sent_analysis(comm, af) for comm in data_clean["Normalized_Comments"]]
    data_clean["afinn_sent_category"] = [afinn_sent_category(scr) for scr in data_clean["afinn_score"]]
    positive = len(data_clean[data_clean["afinn_sent_category"] == "positive"])
    negative = len(data_clean[data_clean["afinn_sent_category"] == "negative"])
    neutral = len(data_clean[data_clean["afinn_sent_category"] == "neutral"])
    count = len(data_clean.index)
    # Write sentiment score into file
    sentimentFile.write("Sentiment Afinn" + "\n")
    sentimentFile.write("Positive sentiment : " + str(positive / count * 100) + "\n")
    sentimentFile.write("Negative sentiment : " + str(negative / count * 100) + "\n")
    sentimentFile.write("Neutral sentiment : " + str(neutral / count * 100) + "\n")
    print("Sentiment Afinn")
    print("Positive sentiment : ", positive / count * 100)
    print("Negative sentiment : ", negative / count * 100)
    print("Neutral sentiment : ", neutral / count * 100)
    newDF = data_clean[["comment", "date", "polarity_vader", "afinn_score"]]
    return (positive, negative, neutral), json.loads(newDF.to_json(orient="records"))
