# Assignment 4 Fall 2025
# Note: ChatGPT used to understand concepts and as a coding assistant
# - Michael Velez
# factcheck.py

import torch
from typing import List
import numpy as np
import spacy
import gc
import re
import nltk
from nltk.corpus import stopwords
from torch.nn.functional import softmax
from nltk.tokenize import sent_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


class FactExample(object):
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """

    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr(
            "fact="
            + repr(self.fact)
            + "; label="
            + repr(self.label)
            + "; passages="
            + repr(self.passages)
        )


class EntailmentModel(object):
    def __init__(self, model, tokenizer, cuda=False):
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(
                premise, hypothesis, return_tensors="pt", truncation=True, padding=True
            )
            if self.cuda:
                inputs = {key: value.to("cuda") for key, value in inputs.items()}

            # Get the model's prediction
            outputs = self.model(**inputs)
            # 3 logits ["entailment", "neutral", "contradiction"]
            logits = outputs.logits[0]

            # Convert raw logits to probabilities
            probs = softmax(logits, dim=-1)

            # Get the probability of entailment only (higher value means model thinks passage supports the fact)
            entail_prob = float(probs[0])

        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        del inputs, outputs, logits, probs
        gc.collect()

        # Return a single number representing how confident the model is
        return entail_prob


class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(FactChecker):
    def __init__(self, threshold: float = 0.65):
        """
        Simple bag-of-words fact-checker.
        If enough words from the fact appear in a passage, then predict it's supported.

        Args:
            threshold (float): Minimum recall score to label fact as supported.
        """
        nltk.download("stopwords", quiet=True)
        self.stopwords = set(stopwords.words("english"))
        self.token_pattern = re.compile(r"[A-Za-z0-9]+")
        self.threshold = threshold

    def _normalize(self, text: str) -> set:
        """
        Convert text into a cleaned set of words by
        1. Lowercasing
        2. Taking only alphanumeric tokens
        3. Removing stopwords
        4. Removing single character tokens

        Args:
            text (str): Raw input text

        Returns:
            set[str]: Unique processed tokens
        """
        raw_tokens = self.token_pattern.findall(text.lower())
        cleaned_tokens = [
            token
            for token in raw_tokens
            if token not in self.stopwords and len(token) > 1
        ]
        return set(cleaned_tokens)

    def _recall(self, fact_tokens: set, passage_tokens: set) -> float:
        """
        Calculate how much of the fact appears in the passage.

        Args:
            fact_tokens (set): Tokens from the fact
            passage_tokens (set): Tokens from a passage

        Returns:
            float: recall score (between 0.0–1.0)
        """
        if not fact_tokens:
            return 0.0
        overlap_count = len(fact_tokens & passage_tokens)
        return overlap_count / len(fact_tokens)

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Predict whether a fact is supported by comparing max recall score with threshold.

        Args:
            fact (str): Fact to verify
            passages (list[dict]): Wikipedia passages

        Returns:
            str: "S" if supported, else "NS"
        """
        fact_tokens = self._normalize(fact)
        max_recall_score = 0.0

        for passage in passages:
            passage_text = passage.get("text", "")
            passage_tokens = self._normalize(passage_text)
            recall_score = self._recall(fact_tokens, passage_tokens)

            # Keep best match
            if recall_score > max_recall_score:
                max_recall_score = recall_score

        return "S" if max_recall_score >= self.threshold else "NS"


class EntailmentFactChecker(FactChecker):
    def __init__(
        self,
        entailment_model,
        overlap_threshold: float = 0.05,
        entail_threshold: float = 0.45,
    ):
        """
        Combines word overlap and entailment modeling for fact checking.

        Args:
            entailment_model (DeBERTa): Pretrained model used to check if a passage entails the fact
            overlap_threshold (float): Minimum overlap score to consider a sentence relevant
            entail_threshold (float): Minimum entailment probability to predict "S" for supported
        """
        self.entailment_model = entailment_model
        self.overlap_checker = WordRecallThresholdFactChecker(
            threshold=overlap_threshold
        )
        self.entail_threshold = entail_threshold

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Predict whether a fact is supported using entailment.

        Args:
            fact (str): Fact to verify
            passages (list[dict]): Wikipedia passages with text

        Returns:
            str: "S" if supported, else "NS"
        """
        # Skip vague or incomplete facts
        if len(fact.split()) <= 2:
            return "NS"

        highest_entailment_score = 0.0
        fact_tokens = self.overlap_checker._normalize(fact)

        for passage in passages:
            passage_text = passage.get("text", "")

            # Split passage into individual sentences
            for sentence in sent_tokenize(passage_text):
                # Skip very short sentences
                if len(sentence) < 5:
                    continue

                # Compute lexical overlap
                sentence_tokens = self.overlap_checker._normalize(sentence)
                overlap_score = self.overlap_checker._recall(
                    fact_tokens, sentence_tokens
                )

                # Pruning by filtering out sentences with low word overlap
                if overlap_score < self.overlap_checker.threshold:
                    continue

                # Run entailment model to get probability of sentence supporting the fact
                entailment_score = self.entailment_model.check_entailment(
                    sentence, fact
                )
                highest_entailment_score = max(
                    highest_entailment_score, entailment_score
                )

                # Early stop if 100% confident
                if highest_entailment_score > 0.99:
                    return "S"

        # Fact is supported if highest entailment score beats the threshold
        return "S" if highest_entailment_score >= self.entail_threshold else "NS"


# OPTIONAL
class DependencyRecallThresholdFactChecker(FactChecker):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = [
                "punct",
                "ROOT",
                "root",
                "det",
                "case",
                "aux",
                "auxpass",
                "dep",
                "cop",
                "mark",
            ]
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == "VERB" else token.head.text
            dependent = token.lemma_ if token.pos_ == "VERB" else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations
