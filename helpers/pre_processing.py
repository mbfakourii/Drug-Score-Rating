import re
from time import sleep

import pandas
from pandas import read_csv
from tqdm import tqdm

from helpers.sentiment_scores import sentiment_scores
from helpers.remove_stop_words import remove_stop_words


def pre_processing(df):
    print("\n> Pre-processing on data:")

    # remove useless columns
    df.drop(["drugName", "condition", "date", "usefulCount"], axis=1, inplace=True)

    # cheng first name columns to id
    df.columns = ['id', 'review', 'rating']

    # added new columns
    df['score_rating'] = '1'
    df['score_sentiment'] = '2'
    df['clean_review'] = '3'

    # pandas.set_option('display.max_columns', None)

    # rearrange columns
    df = df[['id', 'review', 'clean_review', 'rating', 'score_rating', 'score_sentiment']]

    reviews = []
    y = []
    pbar = tqdm(total=len(df))

    for i, row in df.iterrows():
        # # Remove anything that is not a word
        # review = re.sub(r"[^a-z\s]", " ", row.review.lower())
        # review = re.sub(r"\s+", " ", review)
        pbar.update(1)
        # remove stop words
        review = remove_stop_words(row.review)
        reviews.append(review)
        df.at[i, "clean_review"] = review

        if sentiment_scores(review) == 0:
            df.at[i, "score_sentiment"] = 0
        else:
            df.at[i, "score_sentiment"] = 1

        if row.rating < 5:
            y.append(0)
            df.at[i, "score_rating"] = 0
        else:
            y.append(1)
            df.at[i, "score_rating"] = 1

    pbar.close()
    return df
    # return reviews, y
