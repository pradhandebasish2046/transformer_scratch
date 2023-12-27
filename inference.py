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

def generate_translation(prompt, transformer, src, max_seq_length, device, start_token, end_token, temperature):
    """
    Generate a translation for a given prompt using the trained Transformer model.

    Args:
        prompt (str): Input prompt.
        transformer (Transformer): Trained Transformer model.
        src (torch.Tensor): Source sequence tensor.
        max_seq_length (int): Maximum sequence length.
        device (torch.device): Device to perform computations on.
        start_token (int): Start token for decoding.
        end_token (int): End token for decoding.
        temperature (float): Sampling temperature for generating predictions.

    Returns:
        str: Generated translation.

    """

    # Converting model to evaluation stage
    transformer.eval()

    tokenizer = torch.load('tokenizer.pth') # loading the tokenizer

    # Tokenize and convert to indices
    tokens = tokenizer(prompt) # converting text to tokens
    indices = [vocab[t] for t in tokens] # storing the indices

    with torch.no_grad(): # without gradient
        src_data = torch.tensor(src, dtype=torch.long, device=device)
        src_mask, _ = transformer.generate_mask(src_data, torch.zeros(1, max_seq_length, dtype=torch.long, device=device))

        src_embedded = transformer.dropout(transformer.positional_encoding(transformer.encoder_embedding(src_data)))
        enc_output = src_embedded

        # Encoder layers
        for enc_layer in transformer.encoder_layers: # iterating each encoder layer
            enc_output = enc_layer(enc_output, src_mask)

        # Initialize target data with the start token
        tgt_data = torch.tensor([start_token], dtype=torch.long, device=device).unsqueeze(0)

        for _ in range(max_seq_length):
            tgt_mask = (torch.triu(torch.ones(1, tgt_data.size(1), tgt_data.size(1)), diagonal=1)).bool().to(device)
            tgt_embedded = transformer.dropout(transformer.positional_encoding(transformer.decoder_embedding(tgt_data)))

            # Decoder layers
            for dec_layer in transformer.decoder_layers:
                tgt_embedded = dec_layer(tgt_embedded, enc_output, src_mask, tgt_mask)

            output = transformer.fc(tgt_embedded[:, -1, :]) # Prediction
            pred_token = torch.softmax(output[:, -1] / temperature, dim=-1) 
            prediction = torch.multinomial(pred_token, num_samples=1).item()

            while prediction == vocab['<unk>']:
                prediction = torch.multinomial(pred_token, num_samples=1).item()

            if prediction == vocab['<eos>']: # if prediction is end of string token then stop the prediction
                break
            indices.append(prediction) # appending each prediction

    # Convert indices back to tokens
    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return " ".join(tokens)

tokenizer = torch.load('tokenizer.pth')
start_token = src_data[0][0]  
end_token = vocab.lookup_indices(['<eos>'])[0]  
src_sequence = src_data  

# checking prediction for each temperature value
temperatures = [0.1, 0.3, 0.5, 0.7, 0.75, 0.8, 1.0]

# Predicting for input text
prompt = "Do not tell me "
for temp in temperatures:
    translation = generate_translation(prompt, transformer, src_sequence, max_seq_length, device, start_token, end_token, temp)
    print(f"{temp} {translation}")
