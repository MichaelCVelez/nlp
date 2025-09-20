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
# Prefix Embeddings
##############################

class PrefixEmbeddings:
    """
    Handles embeddings for N-character prefixes for handling typos.
    """
    def __init__(self, word_embeddings: WordEmbeddings, prefix_length: int = 4):
        """
        :param word_embeddings: Pretrained word embeddings (WordEmbeddings object)
        :param prefix_length: Number of characters to use for prefix keys
        """
        self.prefix_length = prefix_length
        self.embedding_dim = word_embeddings.get_embedding_length()
        self.word_indexer = word_embeddings.word_indexer
        self.word_embeddings = word_embeddings.vectors

        # Build prefix indexer and vectors, including unknown
        self.prefix_indexer = Indexer()
        self.prefix_indexer.add_and_get_index("UNK")
        self.vectors = self._compute_prefix_vectors()

    def _compute_prefix_vectors(self) -> np.ndarray:
        """
        Compute prefix embeddings by averaging word embeddings sharing the same prefix.
        :return: numpy array of prefix embeddings [num_prefixes x embedding_dim]
        """
        prefix_to_vecs = {}

        # Group embeddings by prefix
        for word, idx in self.word_indexer.objs_to_ints.items():
            prefix = word[:self.prefix_length] if len(word) >= self.prefix_length else word
            prefix_to_vecs.setdefault(prefix, []).append(self.word_embeddings[idx])

        # Average vectors per prefix and assign index
        prefix_vectors = []
        for prefix, vecs in prefix_to_vecs.items():
            prefix_idx = self.prefix_indexer.add_and_get_index(prefix)
            avg_vec = np.mean(vecs, axis=0)
            prefix_vectors.append((prefix_idx, avg_vec))

        # Create final prefix matrix
        prefix_matrix = np.zeros((len(self.prefix_indexer), self.embedding_dim))
        for idx, vec in prefix_vectors:
            prefix_matrix[idx] = vec
        return prefix_matrix

    def get_initialized_embedding_layer(self, frozen: bool = False, padding_idx: int = 0) -> nn.Embedding:
        """
        Returns a trainable embedding layer initialized with prefix vectors.
        :param frozen: whether to freeze embeddings
        :param padding_idx: index for padding
        """
        tensor = torch.tensor(self.vectors, dtype=torch.float)
        return nn.Embedding.from_pretrained(tensor, freeze=frozen, padding_idx=padding_idx)

    def get_embedding_length(self) -> int:
        """Return embedding dimension"""
        return self.embedding_dim


##############################
# Deep Averaging Network
##############################

class DANClassifier(nn.Module):
    """
    Deep Averaging Network using FFNN style from example.
    """
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_hidden_layers: int = 2, dropout: float = 0.3):
        """
        Constructs the computation graph by instantiating the various layers and initializing weights.
        
        :param embedding_dim: size of input embeddings
        :param hidden_dim: size of hidden layers
        :param output_dim: size of output (number of classes)
        :param num_hidden_layers: number of hidden layers
        :param dropout: dropout probability
        """
        # Initialize hidden layers, activate function and dropout layer
        super(DANClassifier, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.g = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Input layer
        self.hidden_layers.append(nn.Linear(embedding_dim, hidden_dim))

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        # Softmax across classes to match NLLLoss expectations
        self.log_softmax = nn.LogSoftmax(dim=1)

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
    Wraps an instance of the network with learned weights and averages word embeddings to be passed to DANClassifier for sentiment prediction.
    predict_all has been overridden to use batching at inference time for faster training
    """
    def __init__(self, word_embeddings: WordEmbeddings, hidden_dim: int = 128, num_hidden_layers: int = 2, device=None):
        self.word_embeddings = word_embeddings
        self.prefix_embeddings = PrefixEmbeddings(word_embeddings, prefix_length=4)
        self.embedding_dim = word_embeddings.get_embedding_length()
        # Positive/negative sentiment
        self.output_dim = 2
        # Width of each hidden layer
        self.hidden_dim = hidden_dim
        # Move between CPU or GPU automatically depending on availability
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize both word and prefix embedding layers
        self.embedding_layer = word_embeddings.get_initialized_embedding_layer(frozen=False, padding_idx=0)
        self.prefix_embedding_layer = self.prefix_embeddings.get_initialized_embedding_layer(frozen=False, padding_idx=0)
        # Create DANClassifier and train
        self.network = DANClassifier(self.embedding_dim, hidden_dim, self.output_dim, num_hidden_layers=num_hidden_layers)
        self.network.train()

        # Move to device
        self.embedding_layer.to(self.device)
        self.prefix_embedding_layer.to(self.device)
        self.network.to(self.device)

    def _average_embeddings(self, batch_sentences: List[List[str]], has_typos: bool=False) -> torch.Tensor:
        """
        Converts a batch of tokenized sentences into fixed-size vectors.
        Handles typos using prefix embeddings if has_typos is True.
        :param batch_sentences: list of tokenized sentences
        :param has_typos: True if we are evaluating on data with typos
        :return: tensor of shape [batch_size x embedding_dim]
        """
        # Store all sentence index lists
        all_sentence_indices = []

        # Loop through each sentence in a batch
        for sentence in batch_sentences:
            # Store indices for words in current sentence
            word_indices = []
            # Loop through each word in a sentence
            for word in sentence:
                # Check flag for whether to handle typos
                if has_typos:
                    # First try 4 length prefix otherwise fallback to 3 else UNK
                    idx = -1
                    for length in [self.prefix_embeddings.prefix_length, 3]:
                        prefix = word[:length] if len(word) >= length else word
                        idx = self.prefix_embeddings.prefix_indexer.index_of(prefix)
                        if idx != -1:
                            break
                    # If prefix not found use index of UNK
                    if idx == -1:
                        idx = self.prefix_embeddings.prefix_indexer.index_of("UNK")
                else:
                    # Get index for the word
                    idx = self.word_embeddings.word_indexer.index_of(word)
                    # If word not found use index of UNK
                    if idx == -1:
                        idx = self.word_embeddings.word_indexer.index_of("UNK")
                # Save index
                word_indices.append(idx)
            # Convert word indices to a tensor
            all_sentence_indices.append(torch.tensor(word_indices, dtype=torch.long))

        # Pad all sentences to the same length with 0's
        padded_batch = rnn_utils.pad_sequence(all_sentence_indices, batch_first=True, padding_value=0).to(self.device)
        # Convert padded batch to embedding vectors depending on whether there are typos
        batch_embeds = self.prefix_embedding_layer(padded_batch) if has_typos else self.embedding_layer(padded_batch)
        # Average batch embeddings to get fixed-size vectors
        avg_embeds = batch_embeds.mean(dim=1)
        return avg_embeds

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Predicts sentiment for a single sentence
        :param ex_words: list of tokens in the sentence
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.        
        :return: binary predicted label
        """
        avg_embed = self._average_embeddings([ex_words], has_typos=has_typos)
        log_probs = self.network(avg_embed)
        return torch.argmax(log_probs, dim=-1).item()

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        Predicts sentiment for a batch of sentences in parallel
        :param all_ex_words: list of tokenized sentences
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.        
        :return: list of binary predicted labels
        """
        avg_batch = self._average_embeddings(all_ex_words, has_typos=has_typos)
        log_probs = self.network(avg_batch)
        return torch.argmax(log_probs, dim=-1).tolist()


##############################
# Training Utilities
##############################

def _train_one_epoch(model: NeuralSentimentClassifier, dataloader, optimizer, criterion, device, train_with_typos: bool):
    """
    Single epoch training.
    """
    total_loss = 0.0
    for batch_words, batch_labels in dataloader:
        batch_labels = batch_labels.to(device)
        # Pass typo setting to determine whether to train with prefix embeddings 
        batch_input = model._average_embeddings(batch_words, has_typos=train_with_typos)

        optimizer.zero_grad()
        log_probs = model.network(batch_input)
        loss = criterion(log_probs, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_words)
    return total_loss


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    Train a Deep Averaging Network with batching and proper loss.
    """
    hidden_dim = 128
    num_hidden_layers = 2
    batch_size = 64
    # Changed defaults for higher accuracy
    num_epochs = 20
    learning_rate = 0.0005

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralSentimentClassifier(word_embeddings, hidden_dim, num_hidden_layers, device=device)
    optimizer = optim.Adam(list(model.network.parameters()) + list(model.embedding_layer.parameters()) + list(model.prefix_embedding_layer.parameters()), lr=learning_rate)
    criterion = nn.NLLLoss()

    # Learning rate scheduler added for higher accuracy
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

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
        # Compute total loss over one epoch
        total_loss = _train_one_epoch(model, dataloader, optimizer, criterion, device, train_with_typos=train_model_for_typo_setting)
        # Print epoch and total loss for debugging visualization
        print(f"Epoch {epoch+1}/{num_epochs}, total_loss={total_loss:.4f}")
        scheduler.step()

    return model
