# Assignment 3 Fall 2025 
# Note: ChatGPT used to understand concepts and as a coding assistant 
# - Michael Velez
# transformer.py

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random
from torch import optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import List
from utils import *


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()
        # Converts each character index -> continuous vector
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # Adds positional info to embeddings (uses batching)
        self.pos_encoder = PositionalEncoding(d_model, num_positions=num_positions, batched=True)
        # Stack of Transformer layers
        self.layers = nn.ModuleList([TransformerLayer(d_model, d_internal) for _ in range(num_layers)])
        # Project model output to label logits
        self.classifier = nn.Linear(d_model, num_classes)
        self.model_dim = d_model
        self.max_positions = num_positions

    def forward(self, indices, causal=False):
        """

        :param indices: list of input indices
        :param causal: prevents a token from “looking ahead” before predicting
        :return: A tuple of the softmax log probabilities and a list of the attention maps.
        """
        # Add a batch dimension if necessary
        is_unbatched = False
        if indices.dim() == 1:
            indices = indices.unsqueeze(0)
            is_unbatched = True

        device = next(self.parameters()).device
        indices = indices.to(device)

        # shape (B, T, D)
        token_embeddings = self.token_embedding(indices)

        # Add positional info
        encoded_inputs = self.pos_encoder(token_embeddings)
        # Used for visualization
        attention_maps = []
        # Pass through Transformer layers 
        for transformer_layer in self.layers:
            # return (B, T, D) and attention (B, T, T)
            encoded_inputs, attention_weights = transformer_layer(encoded_inputs, causal=causal)
            attention_maps.append(attention_weights)

        # Classify each token
        logits = self.classifier(encoded_inputs)
        # Convert logits to log-probabilities for NLLLoss
        log_probs = F.log_softmax(logits, dim=-1)

        # Squeeze batch to match unbatched shape
        if is_unbatched:
            log_probs = log_probs.squeeze(0)
            attention_maps = [attn.squeeze(0) for attn in attention_maps]
        return log_probs, attention_maps


# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()
        # attention components without bias
        self.query_projection = nn.Linear(d_model, d_internal, bias=False)
        self.key_projection = nn.Linear(d_model, d_internal, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)
        # project context back into model dimension
        self.output_projection = nn.Linear(d_model, d_model)
        # Feed-forward network
        self.feedforward1 = nn.Linear(d_model, d_internal)
        self.feedforward2 = nn.Linear(d_internal, d_model)
        # Activation between layers
        self.activation = nn.ReLU()

    def forward(self, input_vecs, causal: bool = False):
        """
        Single-layer forward pass implementing 
        - self-attention + residual
        - feed-forward + residual

        :param input_vecs: (B, T, d_model)
        :param causal: if True, prevents a token from attending to future tokens by setting scores to -inf
        :return: (out, attn) where out is (B, T, d_model) and attn is (B, T, T)
        """
        batch_size, seq_length, model_dim = input_vecs.shape

        # Compute projections
        query = self.query_projection(input_vecs)
        key = self.key_projection(input_vecs)
        value = self.value_projection(input_vecs)

        # Get attention scores using scaled dot-product
        key_dim = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(key_dim)

        if causal:
            # Mask future positions with a large negative number
            causal_mask = torch.triu(torch.ones((seq_length, seq_length), dtype=torch.bool, device=input_vecs.device), diagonal=1)
            scores = scores.masked_fill(causal_mask.unsqueeze(0), float('-1e9'))

        # Softmax the keys dimension to get attention weights that sum to 1
        attention_weights = F.softmax(scores, dim=-1)

        # Get context using matmul
        context = torch.matmul(attention_weights, value)

        # Adds input back after attention (residual)
        context = self.output_projection(context)
        residual = input_vecs + context

        # FFN applied per position followed by second residual
        feedforward_output = self.feedforward2(self.activation(self.feedforward1(residual)))
        output = residual + feedforward_output

        return output, attention_weights


# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        # Create position indices avoid device mismatch (GPU/CPU)
        indices_to_embed = torch.arange(0, input_size, device=x.device).long()
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


class LetterDataset(Dataset):
    """
    Dataset wrapper for LetterCountingExample objects.
    """
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        return ex.input_tensor, ex.output_tensor

# This is a skeleton for train_classifier: you can implement this however you want
def train_classifier(args, train, dev):
    """
    Trains a simple Transformer classifier on the letter counting examples.

    :param args: parsed arguments from letter_counting.py
    :param train: list of LetterCountingExample
    :param dev: list of LetterCountingExample (dev examples)
    :return: trained Transformer model moved to CPU for decode()
    """
    # Hyperparameters
    num_positions = 20
    num_classes = 3
    d_model = 64
    d_internal = 32
    num_layers = 2
    learning_rate = 1e-3
    batch_size = 128
    num_epochs = 10
    
    # 26 letters + special token
    vocab_size = 27
    if len(train) > 0:
        # Expands vocab if needed
        max_idx = max([int(np.max(example.input_indexed)) for example in train])
        vocab_size = max(vocab_size, max_idx + 1)

    # Use GPU if available otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build Transformer model
    model = Transformer(vocab_size=vocab_size,
                        num_positions=num_positions,
                        d_model=d_model,
                        d_internal=d_internal,
                        num_classes=num_classes,
                        num_layers=num_layers).to(device)

    # Update model weights
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Scheduler halves the learning rate every 10 steps
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    # Negative Log Likelihood used as loss function for classification
    criterion = nn.NLLLoss()

    # Boolean for if "BEFORE" is passed use a causal mask to only see previous tokens
    count_only_previous = True if args.task == "BEFORE" else False
    
    def evaluate_model(model, examples, use_causal_mask):
        """
        Check how well the model predicts labels without updating weights
        """
        # Switch to evaluation mode        
        model.eval()
        correct = 0
        total = 0
        # Disable gradient tracking for speed
        with torch.no_grad():
            for example in examples:
                # Add batch dimension
                input_tensor = example.input_tensor.unsqueeze(0).to(device)  
                labels = example.output_tensor.unsqueeze(0).to(device)
                # Forward pass through model
                log_probs, _ = model.forward(input_tensor, causal=use_causal_mask)
                # Pick class with the highest log probability
                predictions = log_probs.argmax(dim=-1)
                # Count correct predictions
                correct += (predictions == labels).sum().item()
                total += labels.numel()
        # Return accuracy
        return correct / total if total > 0 else 0.0

    # Wrap training data in a Dataset and DataLoader
    train_dataset = LetterDataset(train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initial test for overfitting using a smaller batch
    if len(train) >= 64:
        model.train()
        # Only first 64 examples
        small_subset = train[:64]
        small_loader = DataLoader(LetterDataset(small_subset), batch_size=32, shuffle=True)
        print("Running overfit test on 64 examples for 10 iterations...")
        for _ in range(10):
            for small_inputs, small_labels in small_loader:
                # Move data to device
                small_inputs = small_inputs.to(device)
                small_labels = small_labels.to(device)
                # Reset gradients
                optimizer.zero_grad()
                # Forward pass
                log_probs_small, _ = model.forward(small_inputs, causal=count_only_previous)
                B_s, T_s, C_s = log_probs_small.shape
                # Compute loss (reshape for NLLLoss)
                loss_small = criterion(log_probs_small.view(B_s * T_s, C_s),
                                       small_labels.view(B_s * T_s))
                # Backpropagation
                loss_small.backward()
                optimizer.step()
        # Check and print accuracy for examples
        overfit_acc = evaluate_model(model, small_subset, count_only_previous)
        print("Overfit test accuracy (64 exs): %.4f" % overfit_acc)

    # ---------------------------
    # Main training loop
    # ---------------------------
    for epoch in range(num_epochs):
        # Switch to training mode
        model.train()
        total_loss = 0.0
        for batch_inputs, batch_labels in train_loader:
            # Move batch to device
            batch_inputs = batch_inputs.to(device)   
            batch_labels = batch_labels.to(device)   
            # Reset gradients
            optimizer.zero_grad()            
            # Forward pass
            log_probs, _ = model.forward(batch_inputs, causal=count_only_previous)
            B, T, C = log_probs.shape
            # Compute loss (reshape to 2D for NLLLoss)
            loss = criterion(log_probs.view(B * T, C), batch_labels.view(B * T))
            # Backpropagation
            loss.backward()
            # Clip gradients to prevent exploding gradients and update weights
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # Accumulate loss scaled by batch size
            total_loss += loss.item() * batch_inputs.size(0)

        # Step the learning rate scheduler
        scheduler.step()
        # Compute and print average loss for the epoch
        avg_loss = total_loss / len(train) if len(train) > 0 else 0.0
        print("Epoch %d/%d - avg_loss=%.6f" % (epoch + 1, num_epochs, avg_loss))

        # Evaluate on dev set and print accuracy
        dev_acc = evaluate_model(model, dev, count_only_previous)
        print("Dev accuracy: %.4f" % dev_acc)
    # Eval mode
    model.eval()
    # Move to CPU for decoding use    
    model.to(torch.device('cpu'))
    return model


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
