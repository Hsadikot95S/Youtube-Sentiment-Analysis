"""
This peice of code groups the data by date
"""
# Import libraries and files
import json
from statistics import mean
from itertools import groupby

# function to group the data by date
def getDateWiseGrouped(channelName):
    with open("comments/" + channelName + "_comment_scores.json") as json_file:
        contents = json.load(json_file)

    contents.sort(key=lambda content: content["date"])

    groups = groupby(contents, lambda content: content["date"])

    customJSON = []

    for key, value in groups:
        listOfvals = list(value)
        vaderval = [c["polarity_vader"] for c in listOfvals]
        afinnval = [c["afinn_score"] for c in listOfvals]
        customJSON.append({"date": key, "polarity_vader_avg": mean(vaderval), "no_comments": len(listOfvals), "afinn_score_avg": mean(afinnval)})

    return customJSON
