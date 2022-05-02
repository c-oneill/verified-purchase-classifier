"""
NOTE: adapted from assignment 2: spam vs. ham

potential improvements?:
   tfidf weighting --> use TfidfVectorizer?
               --> decreases performance significantly
   adjust decision threshold --> lower threshold more lenient on labelling spam?
   adjust tokenization --> stop words, different regex?
       split on individual digits
       adjust to include more punctutation
   set threshold to include in vocab.
       help with overfitting?
       significantly improved performance
"""

from typing import Iterator, Iterable, Tuple, Text, Union

import numpy as np
from scipy.sparse import spmatrix

from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Binarizer
from sklearn.linear_model import LogisticRegression

NDArray = Union[np.ndarray, spmatrix]


def read_smsspam(smsspam_path: str) -> Iterator[Tuple[Text, Text]]:
    """Generates (label, text) tuples from the lines in an SMSSpam file.

    FROMAT: https://s3.amazonaws.com/amazon-reviews-pds/tsv/index.txt

    :param smsspam_path: The path of an SMSSpam file, formatted as above.
    :return: An iterator over (label, text) tuples.
    """
    unverifiedCount = 0;
    verifiedCount = 0

    lst = list()
    for line in open(smsspam_path):
        line_split = line.split("\t")

        text = line_split[13]
        label = line_split[11]
        if label == "N":
            unverifiedCount += 1
        if label == "Y":
            verifiedCount += 1

        lst.append((label, text))

    print("unverified count:", unverifiedCount)
    print("verifiedCount count:", verifiedCount)

    return iter(lst)


class TextToFeatures:
    def __init__(self, texts: Iterable[Text]):
        """Initializes an object for converting texts to features.

        During initialization, the provided training texts are analyzed to
        determine the vocabulary, i.e., all feature values that the converter
        will support. Each such feature value will be associated with a unique
        integer index that may later be accessed via the .index() method.

        It is up to the implementer exactly what features to produce from a
        text, but the features will always include some single words and some
        multi-word expressions (e.g., "need" and "to you").

        :param texts: The training texts.
        """
        self.__vectorizer = CountVectorizer(ngram_range=(1, 2),
                                            token_pattern= #r"(?u)\b\w\w +\b")
                                            r"(?u)\b[a-zA-Z]+\b|\d|!|\?|\"|\'|\.|\(|\)|\-|;|/|&|\$|\+|£")

        self.__vectorizer.fit(texts)
        # Notes:
        # default token pattern: r”(?u)\b\w\w+\b”
        #   (?u) --> activates unicode matching
        #   \b --> word boundary
        #   \w\w+ --> [a-zA-Z0-9_][a-zA-Z0-9_]+  (ie. 2+ alphanumeric chars)
        #
        # my tokens: words with 1+ alpha chars, single digits, selected punctuation
        #
        #
        # self.__vectorizer = TfidfVectorizer(ngram_range=(1, 2),
        #                                     stop_words={'english'},
        #                                     min_df=2,
        #                                     token_pattern=r"(?u)\b\w\w+\b|!|\?|\"|\'|\.|\(|\)|\-|;|/|&")



    def index(self, feature: Text):
        """Returns the index in the vocabulary of the given feature value.

        :param feature: A feature
        :return: The unique integer index associated with the feature.
        """
        return self.__vectorizer.vocabulary_.get(feature)

    def __call__(self, texts: Iterable[Text]) -> NDArray:
        """Creates a feature matrix from a sequence of texts.

        Each row of the matrix corresponds to one of the input texts. The value
        at index j of row i is the value in the ith text of the feature
        associated with the unique integer j.

        It is up to the implementer what the value of a feature that is present
        in a text should be, though a common choice is 1. Features that are
        absent from a text will have the value 0.

        :param texts: A sequence of texts.
        :return: A matrix, with one row of feature values for each text.
        """
        return self.__vectorizer.transform(texts)


class TextToLabels:
    def __init__(self, labels: Iterable[Text]):
        """Initializes an object for converting texts to labels.

        During initialization, the provided training labels are analyzed to
        determine the vocabulary, i.e., all labels that the converter will
        support. Each such label will be associated with a unique integer index
        that may later be accessed via the .index() method.

        :param labels: The training labels.
        """
        self.__labelEncoder = LabelEncoder()
        self.__labelEncoder.fit(labels)

    def index(self, label: Text) -> int:
        """Returns the index in the vocabulary of the given label.

        :param label: A label
        :return: The unique integer index associated with the label.
        """
        return self.__labelEncoder.transform((label,))[0]

    def __call__(self, labels: Iterable[Text]) -> NDArray:
        """Creates a label vector from a sequence of labels.

        Each entry in the vector corresponds to one of the input labels. The
        value at index j is the unique integer associated with the jth label.

        :param labels: A sequence of labels.
        :return: A vector, with one entry for each label.
        """
        return self.__labelEncoder.transform(labels)


class Classifier:
    def __init__(self):
        """Initalizes a logistic regression classifier.
        """
        THRESHOLD = 0.5
        self.__binarizer = Binarizer(threshold=THRESHOLD)
        self.__model = LogisticRegression()

    def train(self, features: NDArray, labels: NDArray) -> None:
        """Trains the classifier using the given training examples.

        :param features: A feature matrix, where each row represents a text.
        Such matrices will typically be generated via TextToFeatures.
        :param labels: A label vector, where each entry represents a label.
        Such vectors will typically be generated via TextToLabels.
        """
        self.__model.fit(features, labels)

    def predict(self, features: NDArray) -> NDArray:
        """Makes predictions for each of the given examples.

        :param features: A feature matrix, where each row represents a text.
        Such matrices will typically be generated via TextToFeatures.
        :return: A prediction vector, where each entry represents a label.
        """
        return self.__binarizer.transform(self.__model.predict(features).reshape(-1, 1))


def main():
    # get texts and labels from the training data
    train_examples = read_smsspam("dataSplits/verifiedPurchase.train")
    train_labels, train_texts = zip(*train_examples)

    # get texts and labels from the development data
    devel_examples = read_smsspam("dataSplits/verifiedPurchase.devel")
    devel_labels, devel_texts = zip(*devel_examples)

    # create the feature extractor and label encoder
    to_features = TextToFeatures(train_texts)
    to_labels = TextToLabels(train_labels)

    # train the classifier on the training data
    classifier = Classifier()
    classifier.train(to_features(train_texts), to_labels(train_labels))

    # make predictions on the development data
    predicted_indices = classifier.predict(to_features(devel_texts))
    assert np.array_equal(predicted_indices, predicted_indices.astype(bool))

    # measure performance of predictions
    devel_indices = to_labels(devel_labels)
    spam_label = to_labels.index("N")  # as in "no, not a verified purchase"
    f1 = f1_score(devel_indices, predicted_indices, pos_label=spam_label)
    accuracy = accuracy_score(devel_indices, predicted_indices)

    print("F1:", f1)
    print("Accuracy:", accuracy)


if __name__ == '__main__':
    main()