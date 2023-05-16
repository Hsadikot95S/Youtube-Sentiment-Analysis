"""
This file is to get the Video Statistics
"""
# Import libraries and files
import time
from googleapiclient.errors import HttpError

# fetch the video statistics for 50 videos at a time
def getStatistics(youtube, videoIds):
    if len(videoIds) <= 50:
        response = requestStats(youtube, videoIds)
        return response["items"]
    else:
        chunkedVideos = make_chunks(videoIds, 50)
        stats = []
        for vids in chunkedVideos:
            response = requestStats(youtube, vids)
            stats.extend(response["items"])
        return stats


# Slice data into chunks
def make_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


# fetch video statistics using Video ID
def requestStats(youtube, videoIds, retryCount=3):
    try:
        request = youtube.videos().list(part="statistics", maxResults=50, id=",".join(videoIds))
        return request.execute()
    except HttpError as ex:
        if retryCount - 1 == 0:
            return {"items": 0}
        if ex.resp.status == 403:
            time.sleep(60)
        return requestStats(youtube, videoIds, retryCount - 1)
