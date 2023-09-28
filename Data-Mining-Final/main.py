"""
This is the Driver Module and Entry Point
"""
# Import libraries and files
import os
import json
import extractComments as ec
import sentiment_vader as sv
import visualisations as vis
import getVideoIds as fid
import googleapiclient.discovery
import google_auth_oauthlib
import getVideoStatistics as vs
import pandas as p
import sentiment_afinn as sa
import sentiment_NRC as snrc
import mapper
import predictionModels as pred
import createTimeSeriesData as ctsd
import predictionTimeSeriesModels as ptsm

with open("constants.json") as json_file:
    constants = json.load(json_file)

with open("auth/keys.json") as json_file:
    keys = json.load(json_file)

total_comments = []
commentsWithDate = []
total_sentiment = [(0, 0, 0)]
# Accessing YouTube API with credentials
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(constants["OAuthFile"], constants["Scopes"])
credentials = flow.run_console()
youtube = googleapiclient.discovery.build(constants["ApiServiceName"], constants["ApiVersion"], developerKey=keys["APIKey"])
channelName = fid.getIds(youtube, constants["VideoCount"])

with open("comments/" + channelName + "_vidlist.json") as json_file:
    vlist = json.load(json_file)


videoIds = [x["id"]["videoId"] for x in vlist]
stats = vs.getStatistics(youtube, videoIds)


filePath = "sentimentAnalysis/" + str(channelName) + ".txt"
sentimentFile = open(filePath, "w", encoding="utf-8")
commentsInfo = []
# Loop over the videos to extract comments and perform sentiment analysis
for index, v in enumerate(vlist):
    title = v["snippet"]["title"]
    sentimentFile.write("Video Number : " + str(index + 1) + " --> " + title + "\n")
    print("Downloading comments of Video Number : " + str(index + 1) + " --> ", title)
    vid = v["id"]["videoId"]
    comments, commentListWithDate = ec.commentExtract(vid, youtube, constants["CommentCount"])
    total_comments.extend(comments)
    if len(comments) > 0:
        sent_vader, commentListWithDate = sv.analyze_sentiment(commentListWithDate, sentimentFile)
        sent_afinn, commentListWithDate = sa.analyze_sentiment(commentListWithDate, sentimentFile)
        sent_NRC = snrc.sentimentNRC(comments, sentimentFile)
        stats[index]["title"] = "Video Number : " + str(index + 1) + " --> " + title + "\n"
        stats[index] = mapper.mapObject(sent_vader, sent_afinn, sent_NRC, stats[index], comments)
        total_sentiment.append(sent_vader)
        commentsWithDate.extend(commentListWithDate)

# Write Channel stats into file
fdata = json.dumps(stats)
filePtr = open("comments/" + channelName + "_stats.json", "w")
filePtr.write(fdata)
filePtr.close()

# Write Polarity Scores of Comments into file
fdata = json.dumps(commentsWithDate)
filePtr = open("comments/" + channelName + "_comment_scores.json", "w")
filePtr.write(fdata)
filePtr.close()

sentimentFile.close()
print("Total Comments Scraped " + str(len(total_comments)))

# Getting geouped data from Time Series Data file
groupedData = ctsd.getDateWiseGrouped(channelName)
ptsm.performPredictions(groupedData, channelName)
pred.performPredictions(channelName)
vis.performVisualisations(channelName)
vis.makeWordCloud(channelName, total_comments)
