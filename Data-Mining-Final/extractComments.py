"""
This file is to fetch the comments for the videos
"""
# Import libraries and files
import time
import json
from datetime import datetime
from googleapiclient.errors import HttpError


with open("auth/keys.json") as json_file:
    keys = json.load(json_file)
key = keys["APIKey"]

# function to fetch comments along with date
def commentExtract(videoId, youtube, count=-1):
    page_info = makeRequest(youtube, videoId)
    comments = []
    commentsWithDate = []
    comments = [x["snippet"]["topLevelComment"]["snippet"]["textOriginal"] for x in page_info["items"]]
    commentsWithDate = [
        {"comment": x["snippet"]["topLevelComment"]["snippet"]["textOriginal"], "date": x["snippet"]["topLevelComment"]["snippet"]["updatedAt"].split("T")[0]}
        for x in page_info["items"]
    ]
    while ("nextPageToken" in page_info) and (len(comments) < count):
        temp = page_info
        page_info = getNextPage(youtube, videoId, page_info["nextPageToken"])
        commentList = [x["snippet"]["topLevelComment"]["snippet"]["textOriginal"] for x in page_info["items"]]
        commentListWithDate = [
            {
                "comment": x["snippet"]["topLevelComment"]["snippet"]["textOriginal"],
                "date": x["snippet"]["topLevelComment"]["snippet"]["updatedAt"].split("T")[0],
            }
            for x in page_info["items"]
        ]
        comments.extend(commentList)
        commentsWithDate.extend(commentListWithDate)
    return comments, commentsWithDate


#  Call API and handle exceptions
def makeRequest(youtube, videoId, retryCount=3):
    try:
        request = youtube.commentThreads().list(part="snippet", videoId=videoId, maxResults=100)
        return request.execute()
    except HttpError as ex:
        if retryCount - 1 == 0:
            return {"items": []}
        if ex.resp.status == 403:
            time.sleep(60)
        return makeRequest(youtube, videoId, retryCount - 1)


# Fetch next 100 comments
def getNextPage(youtube, videoId, pageToken, retryCount=3):
    try:
        request = youtube.commentThreads().list(part="snippet", videoId=videoId, maxResults=100, pageToken=pageToken)
        return request.execute()
    except HttpError as ex:
        if retryCount - 1 == 0:
            return {"items": []}
        if ex.resp.status == 403:
            time.sleep(60)
        return getNextPage(youtube, videoId, pageToken, retryCount - 1)
