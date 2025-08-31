# Assignment 1 Fall 2025
# Note: ChatGPT used to understand concepts and as a coding assistant
# - Michael Velez

# models.py
import re
from collections import Counter
import numpy as np
import random

from sentiment_data import *
from utils import *

from collections import Counter

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence.
    :param indexer: the provided feature extractor indexer.
    """
    def __init__(self, indexer):
        self._indexer = indexer

    def get_indexer(self):
        return self._indexer

    def _normalize_token(self, tok: str) -> str:
        # lowercase and strip leading/trailing punctuation
        if tok is None:
            return ""
        t = tok.lower()
        # remove leading/trailing non-alphanumeric characters
        t = re.sub(r'^\W+|\W+$', '', t)
        return t

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Returns a Counter mapping feature_index (int) -> value (float).
        :param sentence: list of stirngs to extract features from
        :param add_to_indexer: Optional boolean used to grow the dimensionality of the featurizer
        """
        feats = Counter()
        # Using presence-level features (1 or 0)
        seen = set()
        for tok in sentence:
            t = self._normalize_token(tok)
            if not t:
                continue
            if t in seen:
                #skip remaining loop since presence already accounted for
                continue
            seen.add(t)

            feat_name = f"UNI={t}"
            # New features to be added
            if add_to_indexer:
                try:
                    idx = self._indexer.add_and_get_index(feat_name)
                except Exception as e:
                    raise RuntimeError("Indexer does not implement add_and_get_index in utils.py") from e
            else:
                # Get the existing index
                try:
                    idx = self._indexer.index_of(feat_name)
                except Exception as e:
                    raise RuntimeError("Indexer does not implement get_index; inspect utils.py and update this call") from e

                # if indexer returns -1, None, or a negative value for missing features, skip
                if idx is None or (isinstance(idx, int) and idx < 0):
                    continue
            # 1.0 to signify presence
            feats[idx] += 1.0

        return feats


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Wraps a weight vector (numpy array) and a feature extractor.
    Predicts 1 if score > 0 else 0.
    """
    def __init__(self, weights: np.ndarray, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        # extract features w/o growing the indexer
        feats = self.feat_extractor.extract_features(sentence, add_to_indexer=False)
        score = 0.0
        # Perform dot product and make prediction
        for idx, val in feats.items():
            if idx < 0 or idx >= self.weights.shape[0]:
                continue
            score += self.weights[idx] * val
        return 1 if score > 0.0 else 0


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Logistic regression classifier wraps weight vector and featurizer.
    :param weights: 1D numpy array of size = number of features
    :param feat_extractor: the FeatureExtractor for producing feature indices
    """
    def __init__(self, weights: np.ndarray, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def _score(self, sentence: List[str]) -> float:
        """
        Compute the linear score w · x for a sentence.
        :param sentence: List of strings to be scored
        :return: float: sparse dot procut score
        """
        feats = self.feat_extractor.extract_features(sentence, add_to_indexer=False)
        s = 0.0
        # Perform sparse dot product
        for idx, val in feats.items():
            # skip any feature indices that are out of bounds
            if 0 <= idx < self.weights.shape[0]:
                s += self.weights[idx] * val
        return s

    def _sigmoid(self, x: float) -> float:
        """
        Numerically stable sigmoid function. 
        :param x: raw score value (w · x).
        :return: float: Sigmoid output representing probability (between 0 and 1).
        """
        if x >= 0:
          return 1.0 / (1.0 + np.exp(-x))
        else:
          return np.exp(x) / (np.exp(x) + 1.0)

    def predict_scores(self, sentence: List[str]) -> float:
        """
        Compute probability that sentence belongs to the positive class.
        :param sentence: sentence tokenized into words.
        :return: float: Probability between 0 and 1.
        """
        s = self._score(sentence)
        return self._sigmoid(s)

    def predict(self, sentence: List[str]) -> int:
        """
        Returns binary prediction of 0 or 1 with a threshold of 0.5.
        :param sentence: sentence tokenized into words.
        :return: int: Predicted class label (0 = negative, 1 = positive).
        """
        p = self.predict_scores(sentence)
        return 1 if p >= 0.5 else 0


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    # Hyperparameters
    NUM_EPOCHS = 10
    INITIAL_LEARNING_RATE = 1.0

    # Pre-extract features
    features_cache = []
    max_index = -1
    for ex in train_exs:
        feats = feat_extractor.extract_features(ex.words, add_to_indexer=True)
        features_cache.append(feats)
        if len(feats) > 0:
            max_index = max(max_index, max(feats.keys()))

    dim = max_index + 1 if max_index >= 0 else 0
    # initialize weights
    weights = np.zeros(dim, dtype=float)

    # Uncomment for fixed randomness during dev
    # random.seed(0)

    for epoch in range(NUM_EPOCHS):
        # adjust learning rate (slight decay) 
        eta = INITIAL_LEARNING_RATE / (1.0 + epoch * 0.1)
        # shuffle indices for random order each epoch
        indices = list(range(len(train_exs)))
        random.shuffle(indices)

        for i in indices:
            # feats maps idx -> val.
            feats = features_cache[i]
            # compute score using sparse dot product
            score = 0.0
            for idx, val in feats.items():
                if 0 <= idx < weights.shape[0]:
                    score += weights[idx] * val
            # Map gold label 0/1 to perceptron targets y = -1 or +1
            gold_label = train_exs[i].label
            y = 1 if gold_label == 1 else -1
            # Predicted label
            pred_y = 1 if score > 0.0 else -1

            if pred_y != y:
                # Update weights: w <- w + eta * y * x
                for idx, val in feats.items():
                    if 0 <= idx < weights.shape[0]:
                        weights[idx] += eta * y * val

        # compute training accuracy
        correct = 0
        for j, ex in enumerate(train_exs):
            feats = features_cache[j]
            s = sum(weights[idx] * val for idx, val in feats.items() if 0 <= idx < weights.shape[0])
            pred = 1 if s > 0.0 else 0
            if pred == ex.label:
                correct += 1
        # Print epoch, train accuracy and eta
        train_acc = correct / len(train_exs)
        print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] train_acc={train_acc:.4f} eta={eta:.4f}")

    # wrap in classifier and return
    return PerceptronClassifier(weights, feat_extractor)


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    # Hyperparameters
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.1

    # Pre-extract features for all training examples
    features_cache = []
    max_index = -1
    for ex in train_exs:
        # add new features to indexer
        feats = feat_extractor.extract_features(ex.words, add_to_indexer=True)
        features_cache.append(feats)
        if len(feats) > 0:
            max_index = max(max_index, max(feats.keys()))
    
    # set weight vector dimensionality and intialize weights
    dim = max_index + 1 if max_index >= 0 else 0
    weights = np.zeros(dim, dtype=float)

    # Wrap weights + feature extractor in classifier 
    clf = LogisticRegressionClassifier(weights, feat_extractor)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        indices = list(range(len(train_exs)))
        # shuffle training order
        random.shuffle(indices)

        for i in indices:
            feats = features_cache[i]
            # gold label (0 or 1)
            y = train_exs[i].label

            # Compute probability
            score = clf._score(train_exs[i].words)
            p = clf._sigmoid(score)

            # Gradient update: w <- w + lr * (y - p) * x
            for idx, val in feats.items():
                if 0 <= idx < clf.weights.shape[0]:
                    clf.weights[idx] += LEARNING_RATE * (y - p) * val

        # Print training accuracy
        correct = sum(1 for ex in train_exs if clf.predict(ex.words) == ex.label)
        train_acc = correct / len(train_exs)
        print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] train_acc={train_acc:.4f}")

    return clf


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model