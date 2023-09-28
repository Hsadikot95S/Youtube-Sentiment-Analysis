# comnbine the sentiment score for all algorithms and video stats such as view count, like/dislike count,commentcount
def mapObject(vader, afinn, NRC, stats, comments):
    positive, negative, neutral = vader
    stats["positive_vader"] = (positive / len(comments)) * 100
    stats["negative_vader"] = (negative / len(comments)) * 100
    stats["neutral_vader"] = (neutral / len(comments)) * 100

    positive, negative, neutral = afinn
    stats["positive_afinn"] = (positive / len(comments)) * 100
    stats["negative_afinn"] = (negative / len(comments)) * 100
    stats["neutral_afinn"] = (neutral / len(comments)) * 100

    stats["anger_NRC"] = NRC.iloc[0]["anger"]
    stats["anticipation_NRC"] = NRC.iloc[0]["anticipation"]
    stats["disgust_NRC"] = NRC.iloc[0]["disgust"]
    stats["fear_NRC"] = NRC.iloc[0]["fear"]
    stats["joy_NRC"] = NRC.iloc[0]["joy"]
    stats["negative_NRC"] = NRC.iloc[0]["negative"]
    stats["positive_NRC"] = NRC.iloc[0]["positive"]
    stats["sadness_NRC"] = NRC.iloc[0]["sadness"]
    stats["surprise_NRC"] = NRC.iloc[0]["surprise"]
    stats["trust_NRC"] = NRC.iloc[0]["trust"]

    stats["viewCount"] = int(stats["statistics"]["viewCount"])
    stats["likeCount"] = int(stats["statistics"]["likeCount"])
    if "dislikeCount" in stats["statistics"]:
        stats["dislikeCount"] = int(stats["statistics"]["dislikeCount"])
    else:
        stats["dislikeCount"] = 1  # or some other default value
    stats["commentCount"] = int(stats["statistics"]["commentCount"])
    stats["likedislikeratio"] = (stats["likeCount"]) / (stats["dislikeCount"])
    del stats["statistics"]
    return stats
