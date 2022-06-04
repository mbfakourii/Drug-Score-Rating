from builtins import print

from pandas import read_csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from models.linearsvc import linearsvc
from models.multinomialnb import multinomialnb
from models.bernoullinb import bernoullinb
from models.logistic_regression import logistic_regression
from models.perceptron import perceptron
from models.mlpclassifier import mlpclassifier

# numpy.set_printoptions(threshold=sys.maxsize)

if __name__ == "__main__":
    # print("\n> Concatenate datasets:")
    # a = read_csv("files/drugsComTrain_raw.tsv", delimiter="\t")
    # print("\trows data train = " + len(a).__str__())

    # b = read_csv("files/drugsComTest_raw.tsv", delimiter="\t")
    # print("\trows data test = " + len(b).__str__())

    # out = a.append(b)
    # print("\trows all data = " + len(out).__str__())

    # ------------- Save pre processing
    # aa = pre_processing(out)
    # aa.to_csv("output/output.csv")
    # exit(0)

    # ------------- Load File Pre processing
    print("\n> Concatenate datasets:")
    out = read_csv("output/output.csv")

    print("\n> Train Test Split:")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(out["clean_review"], out["score_rating"], shuffle=False,
                                                                test_size=0.25)
    # ------------- Vectorization
    print("\n> X Train Vectorizing :")
    vectorizer = CountVectorizer()
    vectorizer.fit(X_train_raw.tolist())
    X_train = vectorizer.transform(X_train_raw.tolist())

    print("\n> X Test Vectorizing :")
    X_test = vectorizer.transform(X_test_raw.tolist())

    # ------------- Normalize
    print("\n> X_train Normalizing:")
    transformer_train = Normalizer().fit(X_train)
    X_train = transformer_train.transform(X_train)

    print("\n> X_test Normalizing:")
    transformer_test = Normalizer().fit(X_test)
    X_test = transformer_test.transform(X_test)

    # ------------- Models
    print("\n> LinearSVC:")
    linearsvc(X_train, X_test, y_test, y_train)

    print("\n> BernoulliNB:")
    bernoullinb(X_train, X_test, y_test, y_train)

    print("\n> MultinomialNB:")
    multinomialnb(X_train, X_test, y_test, y_train)

    print("\n> LogisticRegression:")
    logistic_regression(X_train, X_test, y_test, y_train)

    print("\n> Perceptron:")
    perceptron(X_train, X_test, y_test, y_train)

    print("\n> MLPClassifier:")
    mlpclassifier(X_train, X_test, y_test, y_train)



