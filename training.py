# importing necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

import torchtext
from torchtext.data.utils import get_tokenizer

from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention module for the Transformer architecture.

    Args:
        d_model (int): Dimensionality of the model.
        num_heads (int): Number of attention heads.

    Attributes:
        d_model (int): Dimensionality of the model.
        num_heads (int): Number of attention heads.
        d_k (int): Dimensionality of each attention head.

    Methods:
        scaled_dot_product_attention(Q, K, V, mask=None): Applies scaled dot-product attention mechanism.
        split_heads(x): Splits the input tensor into multiple heads.
        combine_heads(x): Combines the multiple heads back into a single tensor.
        forward(Q, K, V, mask=None): Forward pass of the MultiHeadAttention.

    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model # Dimension of the model --. 512
        self.num_heads = num_heads # no of attention heads
        self.d_k = d_model // num_heads # size of the d_k for multi head attention

        # Linear transformations for Q, K, V, and output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Scaled Dot-Product Attention mechanism.

        Args:
            Q (torch.Tensor): Query tensor.
            K (torch.Tensor): Key tensor.
            V (torch.Tensor): Value tensor.
            mask (torch.Tensor, optional): Mask tensor to mask certain positions during attention computation.

        Returns:
            torch.Tensor: Output tensor after applying scaled dot-product attention.

        """
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        """
        Splits the input tensor into multiple heads.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor after splitting into heads.

        """
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """
        Combines the multiple heads back into a single tensor.

        Args:
            x (torch.Tensor): Tensor with multiple heads.

        Returns:
            torch.Tensor: Combined tensor.

        """
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        """
        Forward pass of the MultiHeadAttention.

        Args:
            Q (torch.Tensor): Query tensor.
            K (torch.Tensor): Key tensor.
            V (torch.Tensor): Value tensor.
            mask (torch.Tensor, optional): Mask tensor to mask certain positions during attention computation.

        Returns:
            torch.Tensor: Output tensor after applying MultiHeadAttention.

        """
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

class PositionWiseFeedForward(nn.Module):
    """
    Position-wise FeedForward module for the Transformer architecture.

    Args:
        d_model (int): Dimensionality of the model.
        d_ff (int): Dimensionality of the feedforward layer.

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        relu (nn.ReLU): ReLU activation function.

    Methods:
        forward(x): Forward pass of the PositionWiseFeedForward.

    """
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of the PositionWiseFeedForward.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying feedforward transformation.

        """
        return self.fc2(self.relu(self.fc1(x)))
    
class PositionalEncoding(nn.Module):
    """
    Positional Encoding module for adding positional information to input sequences.

    Args:
        d_model (int): Dimensionality of the model.
        max_seq_length (int): Maximum length of input sequences.

    Attributes:
        pe (torch.Tensor): Positional encoding tensor.

    Methods:
        forward(x): Forward pass of the PositionalEncoding.

    """
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Forward pass of the PositionalEncoding.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after adding positional encoding.

        """
        return x + self.pe[:, :x.size(1)]
    

class EncoderLayer(nn.Module):
    """
    Encoder Layer module for the Transformer architecture.

    Args:
        d_model (int): Dimensionality of the model.
        num_heads (int): Number of attention heads.
        d_ff (int): Dimensionality of the feedforward layer.
        dropout (float): Dropout rate.

    Attributes:
        self_attn (MultiHeadAttention): MultiHeadAttention module for self-attention mechanism.
        feed_forward (PositionWiseFeedForward): PositionWiseFeedForward module for feedforward transformation.
        norm1 (nn.LayerNorm): Layer normalization for the first sub-layer.
        norm2 (nn.LayerNorm): Layer normalization for the second sub-layer.
        dropout (nn.Dropout): Dropout layer.

    Methods:
        forward(x, mask): Forward pass of the EncoderLayer.

    """
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        Forward pass of the EncoderLayer.

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor): Mask tensor for attention mechanism.

        Returns:
            torch.Tensor: Output tensor after applying the EncoderLayer.

        """
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


import torch.nn as nn

class DecoderLayer(nn.Module):
    """
    Decoder Layer module for the Transformer architecture.

    Args:
        d_model (int): Dimensionality of the model.
        num_heads (int): Number of attention heads.
        d_ff (int): Dimensionality of the feedforward layer.
        dropout (float): Dropout rate.

    Attributes:
        self_attn (MultiHeadAttention): MultiHeadAttention module for self-attention mechanism.
        cross_attn (MultiHeadAttention): MultiHeadAttention module for cross-attention mechanism.
        feed_forward (PositionWiseFeedForward): PositionWiseFeedForward module for feedforward transformation.
        norm1 (nn.LayerNorm): Layer normalization for the first sub-layer.
        norm2 (nn.LayerNorm): Layer normalization for the second sub-layer.
        norm3 (nn.LayerNorm): Layer normalization for the third sub-layer.
        dropout (nn.Dropout): Dropout layer.

    Methods:
        forward(x, enc_output, src_mask, tgt_mask): Forward pass of the DecoderLayer.

    """
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        """
        Forward pass of the DecoderLayer.

        Args:
            x (torch.Tensor): Input tensor.
            enc_output (torch.Tensor): Encoder output tensor.
            src_mask (torch.Tensor): Mask tensor for source sequence.
            tgt_mask (torch.Tensor): Mask tensor for target sequence.

        Returns:
            torch.Tensor: Output tensor after applying the DecoderLayer.

        """
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x



class Transformer(nn.Module):
    """
    Transformer model for sequence-to-sequence tasks.

    Args:
        src_vocab_size (int): Source vocabulary size.
        tgt_vocab_size (int): Target vocabulary size.
        d_model (int): Dimensionality of the model.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of layers in the encoder and decoder.
        d_ff (int): Dimensionality of the feedforward layer.
        max_seq_length (int): Maximum sequence length.
        dropout (float): Dropout rate.

    Attributes:
        encoder_embedding (nn.Embedding): Embedding layer for the source sequence.
        decoder_embedding (nn.Embedding): Embedding layer for the target sequence.
        positional_encoding (PositionalEncoding): PositionalEncoding module for adding positional information.
        encoder_layers (nn.ModuleList): List of EncoderLayer modules.
        decoder_layers (nn.ModuleList): List of DecoderLayer modules.
        fc (nn.Linear): Linear layer for the final output.
        dropout (nn.Dropout): Dropout layer.

    Methods:
        generate_mask(src, tgt): Generate masks for source and target sequences.
        forward(src, tgt): Forward pass of the Transformer model.

    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        # Embedding layers
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        # Encoder and Decoder layers
        # using modulelist for multiple layers
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        # Final output layer
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        """
        Generate masks for source and target sequences.

        Args:
            src (torch.Tensor): Source sequence tensor.
            tgt (torch.Tensor): Target sequence tensor.

        Returns:
            torch.Tensor: Source mask.
            torch.Tensor: Target mask.

        """
        # Create masks for source and target sequences
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        # Create a triangular mask for the target sequence to prevent attending to future tokens
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        """
        Forward pass of the Transformer model.

        Args:
            src (torch.Tensor): Source sequence tensor.
            tgt (torch.Tensor): Target sequence tensor.

        Returns:
            torch.Tensor: Model output.

        """
        # Generate masks
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        # Embedding and positional encoding for the source and target sequences
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        # Encoder layers
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        # Decoder layers
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        # Final linear layer
        output = self.fc(dec_output)
        return output
    

def load_clean_data(save_dir:str) -> list:
    """
    Load cleaned data from a file.

    Args:
        save_dir (str): The directory to load the cleaned data from.

    Returns:
        list: The loaded cleaned data.
    """

    # Open the file in read mode ("r")
    with open(save_dir, "r") as file:

        # Use the readlines() method to read each line of the file into a list
        clean_data_list = file.readlines()

    # Strip newline characters from each item in the list
    clean_data_list = [item.strip() for item in clean_data_list]

    return clean_data_list

def train_test_split(clean_data:list, train_size:int, valid_size:int):
    """
    Split the cleaned data into training, validation, and test sets.

    Args:
        clean_data (list): The cleaned data to be split.
        train_size (int): The size of the training set.
        valid_size (int): The size of the validation set.

    Returns:
        vocab, tokenized_train, tokenized_valid, tokenized_test
    """
    # Calculate the size of the test set
    test_size = len(clean_data) - (train_size + valid_size) - 1

    # Split the data into training, validation, and test sets
    train_data = clean_data[:train_size]
    valid_data = clean_data[train_size:train_size + valid_size]
    test_data = clean_data[train_size + valid_size:train_size + valid_size + test_size]

    # Tokenizer
    tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
    torch.save(tokenizer, 'tokenizer.pth')

    # Tokenize the data
    tokenize_data = lambda example, tokenizer: {'tokens': tokenizer(example)}
    tokenized_train = list(map(lambda example: tokenize_data(example, tokenizer), train_data))
    tokenized_valid = list(map(lambda example: tokenize_data(example, tokenizer), valid_data))
    tokenized_test = list(map(lambda example: tokenize_data(example, tokenizer), test_data))

    # Build vocabulary
    vocab = torchtext.vocab.build_vocab_from_iterator(
        map(lambda example: example['tokens'], tokenized_train),
        min_freq=3
    )

    # Insert special tokens
    vocab.insert_token('<unk>', 0)
    vocab.insert_token('<eos>', 1) # Token for end of the string
    vocab.set_default_index(vocab['<unk>']) # Default token is <unk>

    return [vocab, tokenized_train, tokenized_valid, tokenized_test]


def get_data(dataset:list, vocab:dict, batch_size:int) -> torch.Tensor:
    """
    Get data from a dataset using a vocabulary and specified batch size.

    Args:
        dataset (list): The dataset containing examples.
        vocab (dict): The vocabulary mapping tokens to indices.
        batch_size (int): The desired batch size.

    Returns:
        torch.Tensor: The processed data tensor.
    """
    data = []
    for example in dataset:
        if example['tokens']:
            tokens = example['tokens'].append('<eos>') # Adding <eos> token at the end of each sentence
            tokens = [vocab[token] for token in example['tokens']]
            data.extend(tokens) # add the elements of the tokens list to the data list.
    data = torch.LongTensor(data) # Converting to a long 1D tensor
    num_batches = data.shape[0] // batch_size # Calculating no of batches
    data = data[:num_batches * batch_size]
    data = data.view(batch_size, num_batches) # Resizing to (batch_size,num_batches)
    return data

def get_batch(data, seq_len, num_batches, idx):
    """
    Get a batch from the given data.

    Args:
        data (torch.Tensor): The input data tensor.
        seq_len (int): The sequence length.
        num_batches (int): The total number of batches.
        idx (int): The index of the batch.

    Returns:
        torch.Tensor: The source and target batches.

    Example:
        input: I am a good boy that's why everyone loves me
        src: I am a good boy
        target: am a good boy that's
    """
    src = data[:, idx:idx + seq_len]
    target = data[:, idx + 1:idx + seq_len + 1]  # The target is the src shifted by one batch
    return src, target

clean_data = load_clean_data("/content/clean_data.txt")
train_size = 28000
valid_size = 8000
vocab,tokenized_train,tokenized_valid,tokenized_test = train_test_split(clean_data,train_size,valid_size)

batch_size = 64
train_data = get_data(tokenized_train, vocab, batch_size)
valid_data = get_data(tokenized_valid, vocab, batch_size)
test_data = get_data(tokenized_test, vocab, batch_size)

vocab_size = len(vocab)

src_vocab_size = vocab_size
tgt_vocab_size = vocab_size
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 5
dropout = 0.1
num_batches = 64
idx = 0

# Creating the model
transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)


criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train() # Starting the training
data = train_data
# Drop all batches that are not a multiple of seq_len
num_batches = data.shape[-1]
data = data[:, :num_batches - (num_batches - 1) % max_seq_length]
num_batches = data.shape[-1]
best_valid_loss = float('inf')

n_epoch = 20
for epoch in range(n_epoch):
    epoch_loss = 0
    for idx in tqdm(range(0, num_batches - 1, max_seq_length), desc='Training: ', leave=False):
        optimizer.zero_grad() # initializing with zero gradients
        src_data, tgt_data = get_batch(train_data, max_seq_length, num_batches, idx) # Batch of data
        src_data, tgt_data = src_data.to(device), tgt_data.to(device)
        output = transformer(src_data, tgt_data[:, :-1]) # Prediction
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
        loss.backward() # Backpropagation
        optimizer.step() # Updating optimizer
        epoch_loss += loss.item() * max_seq_length # Updating loss value
    valid_loss = epoch_loss / num_batches
    print(f"Epoch: {epoch+1}, Loss: {valid_loss}")
    if valid_loss<best_valid_loss: # If loss is less than previous loss storing the model
      best_valid_loss = valid_loss
      torch.save(transformer.state_dict(), 'best-val-transformer-model.pt')