import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

from latentexp.utils import LOGGER


# Hyperparameters
n_embd = 128 # embedding dimension
n_head = 4 # number of heads
n_layer = 1 # number of layers
dropout = 0.1 # dropout rate
block_size = 128 # maximum length of sequence
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size = 32000


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=block_size):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class MatAttEncoder(nn.Module):
    def __init__(self,n_features, n_tokens=10):
        super(MatAttEncoder, self).__init__()
        
        # self.embedding = nn.Embedding(vocab_size, n_embd)
        self.embedding = nn.Linear(n_features, n_embd)
        self.elu = nn.ELU()
        self.pos_encoder = PositionalEncoding(n_embd, dropout)

        # Definition of the the transformer encoder
        encoder_layers = TransformerEncoderLayer(n_embd, n_head, dim_feedforward=256, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layer)

        self.pooling_layer = nn.Linear(n_tokens, 1)

        self.output_layer = nn.Linear(n_embd, 1)
        
    def forward(self, x):

        x = self.elu(self.embedding(x))  # B T -> B T C
        # x = self.pos_encoder(x)   # B T C -> B T C

        LOGGER.info(f"Tensor shape after embedding and positional encoding: {x.shape}")

        x = self.transformer_encoder(x) # B T C -> B T C

        LOGGER.info(f"Tensor shape after transformer encoder: {x.shape}")
        
        # Transpose tensor for pooling across token dimension
        x = torch.transpose(x, -1, -2)  # B T C -> B C T

        pooled_x = self.pooling_layer(x)[:,:,0] # B C T -> B C 1 -> B C
        LOGGER.info(f"Tensor shape after transposing: {x.shape}")

        out = self.output_layer(pooled_x) # B C -> B 1

        LOGGER.info(f"Tensor shape after transformer output layer: {x.shape}")
        return out



if __name__ == "__main__":
    # Import the necessary libraries
    # Create an instance of the MatAttEncoder class
    model = MatAttEncoder(n_tokens=10)

    # Generate a random input tensor
    input_tensor = torch.rand(64, 10, 128) # B T C

    # Pass the input tensor through the model
    output_tensor = model(input_tensor)



