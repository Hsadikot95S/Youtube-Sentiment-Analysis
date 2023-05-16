# combine the sentiment score for all algorithms and video stats such as view count, like/dislike count,commentcount

def mapObject(vader, stats, comments):
    positive, negative, neutral = vader
    stats["positive_vader"] = (positive / len(comments)) * 100
    stats["negative_vader"] = (negative / len(comments)) * 100
    stats["neutral_vader"] = (neutral / len(comments)) * 100

    stats["viewCount"] = int(stats["statistics"]["viewCount"])
    stats["likeCount"] = int(stats["statistics"]["likeCount"])
    stats["dislikeCount"] = int(stats["statistics"].get("dislikeCount", 0))
    stats["commentCount"] = int(stats["statistics"]["commentCount"])
    # stats["likedislikeratio"] = (stats["likeCount"]) / (stats["dislikeCount"])
    if stats["dislikeCount"] == 0:
        stats["likedislikeratio"] = None
    else:
        stats["likedislikeratio"] = (stats["likeCount"]) / (stats["dislikeCount"])
    del stats["statistics"]
    return stats
