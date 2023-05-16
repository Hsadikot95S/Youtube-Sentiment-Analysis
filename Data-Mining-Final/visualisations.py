import pandas as pd
import matplotlib.pyplot as plt
import seaborn as seabornInstance
import os
import json
import numpy as np
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud



# Visualizing sentiment percentage from Vader and Afinn using Pie chart
def vader_pie(dataframe, channelName):
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    lis1 = []
    tot_pos = dataframe["positive_vader"].sum()
    tot_neg = dataframe["negative_vader"].sum()
    tot_ntr = dataframe["neutral_vader"].sum()
    lis1.append(tot_pos)
    lis1.append(tot_neg)
    lis1.append(tot_ntr)
    senti = ["Positive", "Negative", "Neutral"]
    my_colors = ["green", "red", "orange"]
    line1 = plt.pie(lis1, labels=senti, startangle=90, colors=my_colors, autopct="%.1f%%")
    # plt.title("Vader Sentiment for " + channelName)
    plt.title("Vader")
    plt.axis("equal")
    ax1 = fig.add_subplot(224)
    plt.savefig("images/" + channelName + "_pie1.png")
    plt.show()



# Craete a word clous using 100 most frequent used words
def fancySentiment(comments, channelName):
    stopword = set(stopwords.words("english") + list(string.punctuation) + ["n't"])
    filtered_comments = []
    for i in comments:
        words = word_tokenize(i)
        temp_filter = ""
        for w in words:
            if w not in stopword:
                temp_filter += str(w)
                temp_filter += " "
        filtered_comments.append(temp_filter)
    filtered_comments_str = " ".join(filtered_comments)
    sentiment = WordCloud(width=800, height=500, random_state=21, max_font_size=110, max_words=100).generate(filtered_comments_str)
    # sentiment.generate(filtered_comments_str)
    plt.figure(figsize=(10, 7))
    plt.imshow(sentiment, interpolation="bilinear")
    plt.axis("off")
    plt.savefig("images/" + channelName + "_wordcloud.png")
    plt.show()


# Calling function
def performVisualisations(channelName):
    dataframe = pd.read_json("comments/" + channelName + "_stats.json")
    with open("constants.json") as json_file:
        constants = json.load(json_file)

    vader_pie(dataframe, channelName)


# Calling function
def makeWordCloud(channelName, comments):
    fancySentiment(comments, channelName)
