import json

key_file = open("keywords.json")
feed_file = open("feedback.json")
KEYWORDS = json.load(key_file)
FEEDBACK = json.load(feed_file)


def find_keywords(keywords, feedback):
    sentiment = []
    for i in feedback:
        comment = i["comment"].lower()
        for j in keywords:
            if j["keyword"] in comment:
                sentiment.append(j)

    return sentiment


def find_sentiment(sentiments):
    sentiment_val = 0
    for i in sentiments:
        if i["is_negative"] == True:
            sentiment_val += int(i["emphasis"]) * -1
        else:
            sentiment_val += int(i["emphasis"])
    return sentiment_val


print(f"Your sentiment is: {find_sentiment(find_keywords(KEYWORDS, FEEDBACK))}")
