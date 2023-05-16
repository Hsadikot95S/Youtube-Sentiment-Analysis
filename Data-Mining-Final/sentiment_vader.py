# Import libraries and files
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Sentiment analysis using Vader
def analyze_sentiment(comments, sentimentFile):
    commentbot = SentimentIntensityAnalyzer()
    fresult = {"positivenum": 0, "negativenum": 0, "neutralnum": 0}
    count = 0
    for obj in comments:
        comment = obj["comment"]
        vs = commentbot.polarity_scores(comment)
        count += 1
        # Assign sentiment score based on compound value
        if vs["compound"] >= 0.05:
            fresult["positivenum"] += 1
        elif vs["compound"] <= -0.05:
            fresult["negativenum"] += 1
        else:
            fresult["neutralnum"] += 1
        obj["polarity_vader"] = vs["compound"]

    # Write Sentiment score into file
    sentimentFile.write("Sentiment Vader" + "\n")
    sentimentFile.write("Positive sentiment : " + str(fresult["positivenum"] / count * 100) + "\n")
    sentimentFile.write("Negative sentiment : " + str(fresult["negativenum"] / count * 100) + "\n")
    sentimentFile.write("Neutral sentiment : " + str(fresult["neutralnum"] / count * 100) + "\n")
    print("Sentiment Vader")
    print("Positive sentiment : ", fresult["positivenum"] / count * 100)
    print("Negative sentiment : ", fresult["negativenum"] / count * 100)
    print("Neutral sentiment : ", fresult["neutralnum"] / count * 100)
    return (fresult["positivenum"], fresult["negativenum"], fresult["neutralnum"]), comments
