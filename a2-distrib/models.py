# Assignment 2 Fall 2025 
# Note: ChatGPT used to understand concepts and as a coding assistant 
# - Michael Velez
# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
from typing import List
from sentiment_data import *
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


##############################
# Deep Averaging Network
##############################

class DAN(nn.Module):
    """
    Deep Averaging Network using FFNN style from example.
    """
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_hidden_layers: int = 2, dropout: float = 0.2):
        """
        Constructs the computation graph by instantiating the various layers and initializing weights.
        
        :param embedding_dim: size of input embeddings
        :param hidden_dim: size of hidden layers
        :param output_dim: size of output (number of classes)
        :param num_hidden_layers: number of hidden layers
        :param dropout: dropout probability
        """
        # Initialize hidden layers, activate function and dropout layer
        super(DAN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.g = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Input layer
        self.hidden_layers.append(nn.Linear(embedding_dim, hidden_dim))

        # Append any additional hidden layers
        for _ in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.log_softmax = nn.LogSoftmax(dim=0)

        # Initialize weights for all layers according to a formula due to Xavier Glorot.
        for layer in self.hidden_layers:
            nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        """
        Runs the neural network on the given batch of data and returns log probabilities of the classes.

        :param x: a [batch_size, inp]-shaped tensor of input data
        :return: a [batch_size, out]-shaped tensor of log probabilities
        """
        for layer in self.hidden_layers:
            # Apply linear + ReLU and dropout
            x = self.g(layer(x))
            x = self.dropout(x)
        return self.log_softmax(self.output_layer(x))


def form_input(x) -> torch.Tensor:
    """
    Form the input to the neural network. In general this may be a complex function that synthesizes multiple pieces
    of data, does some computation, handles batching, etc.

    :param x: a [num_samples x inp] numpy array containing input data
    :return: a [num_samples x inp] Tensor
    """ 
    return torch.from_numpy(x).float()


##############################
# NeuralSentimentClassifier
##############################

class NeuralSentimentClassifier(SentimentClassifier):
    """
    Wraps an instance of the network with learned weights and averages word embeddings to be passed to DAN for sentiment prediction.
    predict_all has been overridden to use batching at inference time for faster training
    """
    def __init__(self, word_embeddings: WordEmbeddings, hidden_dim: int = 128, num_hidden_layers: int = 2, device=None):
        self.word_embeddings = word_embeddings
        self.embedding_dim = word_embeddings.get_embedding_length()
        # Positive/negative sentiment
        self.output_dim = 2
        # Width of each hidden layer
        self.hidden_dim = hidden_dim
        # Move between CPU or GPU automatically depending on availability       
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize embeddings
        self.embedding_layer = word_embeddings.get_initialized_embedding_layer(frozen=False, padding_idx=0)
        # Create DAN and train
        self.network = DAN(self.embedding_dim, hidden_dim, self.output_dim, num_hidden_layers=num_hidden_layers)
        self.network.train()

        # Move to device
        self.embedding_layer.to(self.device)
        self.network.to(self.device)

    def _average_embeddings(self, batch_words: List[List[str]]) -> torch.Tensor:
        """
        Converts a batch of tokenized sentences into fixed-size vectors
        :param batch_words: list of tokenized sentences
        :return: tensor of shape [batch_size x embedding_dim]
        """
        # Mapping words to indices (uses UNK for words out of vocab)
        batch_indices = []
        for ex_words in batch_words:
            idxs = [self.word_embeddings.word_indexer.index_of(w) 
                    if self.word_embeddings.word_indexer.index_of(w) != -1
                    else self.word_embeddings.word_indexer.index_of("UNK")
                    for w in ex_words]
            batch_indices.append(torch.tensor(idxs, dtype=torch.long))

        # Pad sentences to equal length
        padded_batch = rnn_utils.pad_sequence(batch_indices, batch_first=True, padding_value=0).to(self.device)
        # Look up embeddings
        batch_embeds = self.embedding_layer(padded_batch)
        # Average embeddings across tokens
        avg_embeds = batch_embeds.mean(dim=1)
        return avg_embeds


    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Predicts sentiment for a single sentence
        :param ex_words: list of tokens in the sentence
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.        
        :return: binary predicted label
        """
        avg_embed = self._average_embeddings([ex_words])
        log_probs = self.network(avg_embed)
        return torch.argmax(log_probs, dim=-1).item()


    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        Predicts sentiment for a batch of sentences in parallel
        :param all_ex_words: list of tokenized sentences
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.        
        :return: list of binary predicted labels
        """
        avg_batch = self._average_embeddings(all_ex_words)
        log_probs = self.network(avg_batch)
        return torch.argmax(log_probs, dim=-1).tolist()


##############################
# DAN Training
##############################

def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    Train a Deep Averaging Network with batching and proper loss.
    :param args: Command-line args
    :param train_exs: training examples
    :param dev_exs: development set
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model.
    """
    # Initialize params 
    hidden_dim = 128
    num_hidden_layers = 2
    batch_size = 64
    num_epochs = 10 
    learning_rate = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralSentimentClassifier(word_embeddings, hidden_dim, num_hidden_layers, device=device)
    optimizer = optim.Adam(list(model.network.parameters()) + list(model.embedding_layer.parameters()), lr=learning_rate)
    criterion = nn.NLLLoss()

    # Precompute word indices for all sentences to speed up training
    all_sentences = [ex.words for ex in train_exs]
    all_labels = torch.tensor([ex.label for ex in train_exs], dtype=torch.long)
    dataset = list(zip(all_sentences, all_labels))

    def collate_fn(batch):
        """
        Prepares a batch for the DataLoader by separating words and stacking labels
        :param batch: list of (words, label) pairs
        :return: tuple (batch_words, batch_labels)
        """
        batch_words, batch_labels = zip(*batch)
        return batch_words, torch.stack(batch_labels)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Training loop
    for epoch in range(num_epochs):
        # Initialize loss
        total_loss = 0.0
        for batch_words, batch_labels in dataloader:
            batch_labels = batch_labels.to(device)
            batch_input = model._average_embeddings(batch_words)

            # Reset gradients
            optimizer.zero_grad()
            # Forward pass
            log_probs = model.network(batch_input)
            # Compute loss
            loss = criterion(log_probs, batch_labels)
            # Backpropagate
            loss.backward()
            # Update weights
            optimizer.step()
            # Accumulate loss
            total_loss += loss.item() * len(batch_words)
        # Print epoch and total loss for debugging visualization
        print(f"Epoch {epoch+1}/{num_epochs}, total_loss={total_loss:.4f}")

    return model
