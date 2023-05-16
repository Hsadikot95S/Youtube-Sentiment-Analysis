"""
This file is for to determine accuracy of all the sentiment algorithms on the baseline data
"""
# Import libraries and files
import pandas as p

import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# function for Vader Sentiment Analysis
def getVaderSentiment(row):
    commentbot = SentimentIntensityAnalyzer()
    vs = commentbot.polarity_scores(row["Comments"])
    if vs["compound"] >= 0.05:
        return 1.0
    elif vs["compound"] <= -0.05:
        return -1.0
    else:
        return 0


# function to calculate sentiment score using Vader
def analyze_sentiment_vader(df):
    df["vader_score"] = df.apply(lambda row: getVaderSentiment(row), axis=1)
    df["accuracy"] = np.where(df["vader_score"] == df["overallSentiment"], "True", "False")
    print(len(df[df["accuracy"] == "True"]) / len(df.index))


# Calling function to get accuracy for both Vader and Afinn
def performBaselineAnalysis():
    dataframe = p.read_csv("data/labeled_comments.csv", encoding="ISO-8859-1")
    dataframe["overallSentiment"] = (dataframe["Positive"] * 1) + (dataframe["Neutral"] * 0) + (dataframe["Negative"] * -1)

    analyze_sentiment_vader(dataframe)


performBaselineAnalysis()
