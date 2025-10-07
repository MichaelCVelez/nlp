# Assignment 3 Fall 2025 
# Note: ChatGPT used to understand concepts and as a coding assistant 
# - Michael Velez
# transformer_lm.py

import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from typing import List
import matplotlib.pyplot as plt

# Reuse Transformer from part 1
from transformer import Transformer


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class NeuralLanguageModel(LanguageModel):
    def __init__(self, transformer_model: Transformer, vocab_index):
        self.model = transformer_model
        self.vocab_index = vocab_index
        # eval mode to prevent dropout
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.vocab_size = self.model.classifier.out_features if hasattr(self.model, "classifier") else None
        # space character as "start-of-sequence" char
        self.start_char = ' '
        self.start_idx = self.vocab_index.index_of(self.start_char)

    def get_next_char_log_probs(self, context: str) -> np.ndarray:
        # Tokenize context and build indices as start_idx + len(context)
        context_indices = [self.vocab_index.index_of(c) for c in context]
        input_indices = [self.start_idx] + context_indices  
    
        # Max context size
        num_positions = getattr(self.model, "num_positions", None)

        # Fallback to default training seq_len 
        if num_positions is None:
            num_positions = 128

        # Truncate to most recent num_positions tokens
        if len(input_indices) > num_positions:
            input_indices = input_indices[-num_positions:]

        with torch.no_grad():
            # Convert to tensor
            input_tensor = torch.LongTensor(input_indices).unsqueeze(0).to(self.device)
            # Forward pass with casual attention so model doesn't cheat
            log_probs, _ = self.model.forward(input_tensor, causal=True)
            # Take the final log probability of next letter given context
            final_log_probs = log_probs[0, -1, :].cpu().numpy()
        return final_log_probs
    
    def get_log_prob_sequence(self, next_chars: str, context: str) -> float:
        total_log_prob = 0.0
        current_context = context

        for target_ch in next_chars:
            # Get log-probabilities for all next characters given current context
            log_prob_vector = self.get_next_char_log_probs(current_context)
            # Index of the true next character in the vocabulary
            char_index = self.vocab_index.index_of(target_ch)
            # Add the log-probability of true next character to the total
            total_log_prob += float(log_prob_vector[char_index])
            # Extend the context
            current_context = current_context + target_ch
        return total_log_prob
    


class CharChunkDataset(Dataset):
    """
        Helper class for chunking dataset and predicting all next characters in block simultaneously
    """
    def __init__(self, text: str, vocab_index, seq_len: int = 64, step: int = None):
        self.text = text
        self.vocab_index = vocab_index
        # Characters per chunk
        self.seq_len = seq_len
        self.step = seq_len if step is None else step
        self.start_idx = self.vocab_index.index_of(' ')
        self.inputs = []
        self.targets = []

        # iterate over text, forming chunks
        text_cursor = 0
        max_start = max(0, len(text) - seq_len + 1)
        while text_cursor <= max_start:
            chunk = text[text_cursor:text_cursor + seq_len]
            # chunk shorter than seq_len means end of block
            if len(chunk) < seq_len:
                break
            # store input and target indices
            input_indices = [self.start_idx] + [self.vocab_index.index_of(c) for c in chunk[:-1]]
            target_indices = [self.vocab_index.index_of(c) for c in chunk]
            self.inputs.append(np.array(input_indices, dtype=np.int64))
            self.targets.append(np.array(target_indices, dtype=np.int64))
            # Shift cursor forward on to next chunk
            text_cursor += self.step

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # converts to PyTorch tensors
        return torch.from_numpy(self.inputs[idx]).long(), torch.from_numpy(self.targets[idx]).long()
    

def train_lm(args, train_text: str, dev_text: str, vocab_index):
    """
    Trains a Transformer language model on train_text and evaluates on dev_text.
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    
    # Hyperparameters
    seq_len = getattr(args, "seq_len", 128)
    num_positions = seq_len
    d_model = getattr(args, "d_model", 128)
    d_internal = getattr(args, "d_internal", 256)
    num_layers = getattr(args, "num_layers", 2)
    batch_size = getattr(args, "batch_size", 128)
    num_epochs = getattr(args, "epochs", 15)
    learning_rate = getattr(args, "lr", 1e-3)
    step = getattr(args, "step", seq_len // 4)
    random_seed = getattr(args, "seed", 1234)

    # Locks randomness so runs are comparable
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Determine vocab size
    vocab_size = getattr(vocab_index, "__len__", None) and len(vocab_index) if hasattr(vocab_index, "__len__") else None
    if not vocab_size:
        # Scan train_text to find max index
        if len(train_text) > 0:
            max_idx = max([vocab_index.index_of(c) for c in train_text])
            vocab_size = max(27, max_idx + 1)
        else:
        # Fallback to 27 (alphabet + space characters)
            vocab_size = 27

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model reusing Transformer from part 1
    model = Transformer(vocab_size=vocab_size,
                        num_positions=num_positions,
                        d_model=d_model,
                        d_internal=d_internal,
                        num_classes=vocab_size,
                        num_layers=num_layers).to(device)

    model.num_positions = num_positions

    # Optimizer, LR scheduler, and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
    criterion = nn.NLLLoss()

    # Build training dataset by slicing the long text into input/target pairs with overlap for variety
    train_dataset = CharChunkDataset(train_text, vocab_index, seq_len=seq_len, step=step)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initial overfit test 
    if len(train_dataset) >= 64:
        model.train()
        # Up to 32 randomized samples per batch    
        small_loader = DataLoader(train_dataset, batch_size=min(32, len(train_dataset)), shuffle=True)
        print("Running overfit test on a small training subset to sanity-check...")
        # Perform 3x gradient updates to model weights before main training
        for _ in range(3):
            for inputs_tensor, targets_tensor in small_loader:
                # Move to GPU
                inputs_tensor = inputs_tensor.to(device)
                targets_tensor = targets_tensor.to(device)
                # Reset gradients
                optimizer.zero_grad()
                log_probs, _ = model.forward(inputs_tensor, causal=True)
                # Get shape (B, T, V)
                batch_size_dim, seq_len_dim, vocab_size_dim = log_probs.shape
                loss = criterion(
                    log_probs.view(batch_size_dim * seq_len_dim, vocab_size_dim),
                    targets_tensor.view(batch_size_dim * seq_len_dim)
                )
                # Backpropagation
                loss.backward()
                # Prevent gradient explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

    # ---------------------------
    # Main training loop
    # ---------------------------

    # Track metrics for plotting trends
    train_losses = []
    full_stream_ppls = []

    for epoch in range(1, num_epochs + 1):
        # Activate training features
        model.train()
        # Reset loss and timer every epoch
        epoch_loss = 0.0
        epoch_start_time = time.time()
        
        # Training performed per batch
        for batch_inputs, batch_targets in train_loader:
            # Move to GPU
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            # Clear gradients
            optimizer.zero_grad()
            # Get log-probabilities for NLLLoss
            log_probs, _ = model.forward(batch_inputs, causal=True)
            # (Batch size, Time steps, Vocabulary size) or (B, T, V)
            batch_size_dim, seq_len_dim, vocab_size_dim = log_probs.shape

            loss = criterion(
                # Flatten (B, T, V) -> (B*T, V) 
                log_probs.view(batch_size_dim * seq_len_dim, vocab_size_dim),
                # Flatten (B, T) -> (B*T,)
                batch_targets.view(batch_size_dim * seq_len_dim)
            )
            # Backpropagate
            loss.backward()
            # Prevent exploding gradients by clipping gradient norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update model weights
            optimizer.step()
            # Accumulate loss across all batches
            epoch_loss += loss.item() * batch_inputs.size(0)

        # Decay LR every 4 epochs
        scheduler.step()
        # Evaluate and print elapsed time and average training loss per epoch
        epoch_time_sec = time.time() - epoch_start_time
        avg_loss = epoch_loss / len(train_dataset) if len(train_dataset) > 0 else 0.0
        print(f"Epoch {epoch}/{num_epochs} - avg_train_loss={avg_loss:.6f} - time={epoch_time_sec:.1f}s")

        # Evaluate on dev set 
        model.eval()
        lm_wrapper_eval = NeuralLanguageModel(model, vocab_index)

        # Compute and print full-stream dev perplexity in a single pass over entire dev_text
        full_logprob = lm_wrapper_eval.get_log_prob_sequence(dev_text, "")
        full_avg = full_logprob / len(dev_text)
        full_ppl = math.exp(-full_avg)
        print(f"Dev (full-stream): total_logprob={full_logprob:.4f}  avg_logprob/token={full_avg:.6f}  perplexity={full_ppl:.4f}")

        # Record metrics for plotting
        train_losses.append(avg_loss)
        full_stream_ppls.append(full_ppl)

    # Plot training curves for visualization
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss", marker='o')
    plt.plot(epochs, full_stream_ppls, label="Full Stream Dev Perplexity", marker='^')
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Transformer LM Training Progress")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Move to CPU for returning wrapped NeuralLanguageModel
    model.to(torch.device("cpu"))
    lm_wrapper = NeuralLanguageModel(model, vocab_index)
    return lm_wrapper
