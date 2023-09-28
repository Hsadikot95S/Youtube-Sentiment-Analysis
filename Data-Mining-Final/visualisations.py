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

# Visualizing graph results from NRC Lexicon
def nrc_visualisation(dataframe, channelName):
    emotions = [
        "anger_NRC",
        "anticipation_NRC",
        "disgust_NRC",
        "fear_NRC",
        "joy_NRC",
        "negative_NRC",
        "positive_NRC",
        "sadness_NRC",
        "surprise_NRC",
        "trust_NRC",
    ]
    cm = ["r", "coral", "maroon", "tomato", "cyan", "chocolate", "green", "magenta", "black", "yellowgreen"]
    fig, ax = plt.subplots(4, 3, facecolor="w", edgecolor="k")
    fig.subplots_adjust(hspace=0.2, wspace=0.1)
    # fig.suptitle("Sentiment of " + channelName, fontsize=20, x=0.7, y=0.2)
    fig.subplots_adjust(top=0.88)
    ax = ax.ravel()

    for i, emotion in enumerate(emotions):
        emotionName = emotion.split("_")
        y = dataframe[emotion].to_numpy()
        ax[i].plot(y, linewidth=2, color=cm[i], label=emotionName[0].capitalize())
        ax[i].set_xticks([])
        ax[i].legend()
        ax[i].set_ylim([0, 0.5])

    fig.tight_layout()
    fig.delaxes(ax[-1])
    fig.delaxes(ax[-2])
    plt.tight_layout()
    plt.savefig("images/" + channelName + "_nrc.png")
    plt.show()


# Visualizing graph results from Vader and Afinn sentiment analysis
def vader_afinn_vis(dataframe, constants, channelName):
    zero = 0
    dataframe["score_vader"] = dataframe["positive_vader"] / 100 * 1 + dataframe["negative_vader"] / 100 * -1 + dataframe["neutral_vader"] / 100 * 0
    score_array_vader = dataframe["score_vader"].to_numpy()
    score_masked_vader = np.ma.masked_less_equal(score_array_vader, zero)
    dataframe["score_afinn"] = dataframe["positive_afinn"] / 100 * 1 + dataframe["negative_afinn"] / 100 * -1 + dataframe["neutral_afinn"] / 100 * 0
    score_array_afinn = dataframe["score_afinn"].to_numpy()
    score_masked_afinn = np.ma.masked_less_equal(score_array_afinn, zero)
    ax = plt.subplot(111)
    ax.plot(score_array_vader, "r", label="Vader ")
    # ax.plot(score_masked_vader, 'g',label='Vader positive')
    ax.plot(score_array_afinn, "k", ls="dashed", linewidth=2, label="Afinn ")
    # ax.plot(score_masked_afinn, 'blue',ls='dashed',linewidth=2,label='Afinn positive',)
    plt.axhline(zero, color="k", linestyle="--")
    # plt.title("Polarity Score in Vader vs Afinn for " + channelName, fontsize=20)
    plt.xlabel("Video Number", fontsize=10)
    plt.ylabel("Sentiment score", fontsize=10)
    plt.ylim(-1, 1)
    plt.xlim(0, constants["VideoCount"])
    ax.legend()
    plt.savefig("images/" + channelName + "_vader_afinn.png")
    plt.show()


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
    lis1 = []
    tot_pos = dataframe["positive_afinn"].sum()
    tot_neg = dataframe["negative_afinn"].sum()
    tot_ntr = dataframe["neutral_afinn"].sum()
    lis1.append(tot_pos)
    lis1.append(tot_neg)
    lis1.append(tot_ntr)
    senti = ["Positive", "Negative", "Neutral"]
    my_colors = ["green", "red", "orange"]
    line2 = plt.pie(lis1, labels=senti, startangle=90, colors=my_colors, autopct="%.1f%%")
    # plt.title("Afinn Sentiment for " + channelName)
    plt.title("Afinn")
    plt.axis("equal")
    plt.savefig("images/" + channelName + "_pie1.png")
    plt.show()


# Visualizing sentiment percentage from NRC Lexicon using Pie chart
def NRC_pie(dataframe, channelName):
    # ax1 = fig.add_subplot(339)
    lis1 = []
    tot_pos = dataframe["positive_NRC"].sum()
    tot_neg = dataframe["negative_NRC"].sum()
    tot_angr = dataframe["anger_NRC"].sum()
    tot_ant = dataframe["anticipation_NRC"].sum()
    tot_dis = dataframe["disgust_NRC"].sum()
    tot_fear = dataframe["fear_NRC"].sum()
    tot_joy = dataframe["joy_NRC"].sum()
    tot_sad = dataframe["sadness_NRC"].sum()
    tot_sur = dataframe["surprise_NRC"].sum()
    tot_trs = dataframe["trust_NRC"].sum()

    lis1.append(tot_pos)
    lis1.append(tot_neg)
    lis1.append(tot_angr)
    lis1.append(tot_ant)
    lis1.append(tot_dis)
    lis1.append(tot_fear)
    lis1.append(tot_joy)
    lis1.append(tot_sad)
    lis1.append(tot_sur)
    lis1.append(tot_trs)

    senti = ["positive", "negative", "Anger", "Anticipation", "Disgust", "Fear", "Joy", "Sad", "Surprise", "Trust"]
    my_colors = ["green", "red", "orange", "silver", "white", "pink", "purple", "magenta", "yellow", "cyan"]
    plt.pie(lis1, labels=senti, startangle=90, colors=my_colors, autopct="%.1f%%")
    # plt.title("NRC Sentiment for " + channelName, x=0, y=0.0005)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("images/" + channelName + "_pie2.png")
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
    nrc_visualisation(dataframe, channelName)
    vader_afinn_vis(dataframe, constants, channelName)
    vader_pie(dataframe, channelName)
    NRC_pie(dataframe, channelName)


# Calling function
def makeWordCloud(channelName, comments):
    fancySentiment(comments, channelName)
