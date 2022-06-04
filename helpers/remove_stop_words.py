import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer


def remove_stop_words(value):
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    stop_words = set(stopwords.words('english'))

    word_tokens = word_tokenize(value)

    filtered_sentence = [word for word in word_tokens if word not in stopwords.words('english')]

    filtered_sentence = []

    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    return TreebankWordDetokenizer().detokenize(filtered_sentence).replace('\\', '')
