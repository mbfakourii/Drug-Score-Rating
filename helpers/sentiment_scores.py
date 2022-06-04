from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def sentiment_scores(sentiment):
    sentiment_dict = SentimentIntensityAnalyzer().polarity_scores(sentiment)

    # print(" ", sentiment_dict['compound'] * 100, "% compound")

    if sentiment_dict['compound'] >= 0:
        # Positive
        return 1

    elif sentiment_dict['compound'] <= - 0.01:
        # Negative
        return 0
