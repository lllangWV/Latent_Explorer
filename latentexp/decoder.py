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
vocab_size = 128


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=block_size):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)


# class MatAttDecoder(nn.Module):
#     def __init__(self, n_tokens=10):
#         super(MatAttDecoder, self).__init__()
        
#         # self.embedding = nn.Embedding(block_size, n_embd)
#         self.embedding = nn.Linear(block_size, n_embd)
#         self.pos_encoder = PositionalEncoding(n_embd, dropout)

#         # Definition of the the transformer encoder
#         decoder_layers = TransformerDecoderLayer(n_embd, n_head, dim_feedforward=256, dropout=dropout)
#         self.transformer_decoder = TransformerDecoder(decoder_layers, n_layer)

#         self.pooling_layer = nn.Linear(n_tokens, 1)

#         self.output_layer = nn.Linear(n_embd, 1)
        
#     def forward(self, x):

#         x = self.embedding(x)  # B T -> B T C
#         x = self.pos_encoder(x)   # B T C -> B T C

#         LOGGER.info(f"Tensor shape after embedding and positional encoding: {x.shape}")

#         x = self.transformer_decoder(x) # B T C -> B T C

#         LOGGER.info(f"Tensor shape after transformer decoder: {x.shape}")
        
#         # Transpose tensor for pooling across token dimension
#         x = torch.transpose(x, -1, -2)  # B T C -> B C T

#         pooled_x = self.pooling_layer(x)[:,:,0] # B C T -> B C 1 -> B C
#         LOGGER.info(f"Tensor shape after transposing: {x.shape}")

#         out = self.output_layer(pooled_x) # B C -> B 1

#         LOGGER.info(f"Tensor shape after transformer output layer: {x.shape}")
#         return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, mask=False):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

        self.mask=mask
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        if self.mask is True:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, mask=False):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size,mask=mask) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        LOGGER.info(f"Shape after head concatenation: {out.shape}")
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, mask=False):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()

        if n_embd % n_head != 0:
            raise ValueError(f"embedding dimension {n_embd} should be divisible by the number of heads {n_head}")
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, mask=mask)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        LOGGER.info(f"Shape before self-attention: {x.shape}")
        x = x + self.sa(self.ln1(x))
        LOGGER.info(f"Shape after self-attention: {x.shape}")
        x = x + self.ffwd(self.ln2(x))
        LOGGER.info(f"Shape after feed-forward: {x.shape}")
        return x


class TransformerDecoder(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.token_embedding_table = nn.Linear(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head,mask=True) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # B T -> (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        LOGGER.info(f"Shape before blocks: {x.shape}")
        x = self.blocks(x) # (B,T,C)
        LOGGER.info(f"Shape after blocks: {x.shape}")
        x = self.ln_f(x) # (B,T,C)
        LOGGER.info(f"Shape after ln_f: {x.shape}")
        logits = self.lm_head(x) # (B,T,vocab_size)
        LOGGER.info(f"Shape after lm_head: {logits.shape}")
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            LOGGER.info(f"Shape of logits after view: {logits.shape}")
            targets = targets.view(B*T)
            LOGGER.info(f"Shape of targets after view: {targets.shape}")
            loss = F.cross_entropy(logits, targets)
            LOGGER.info(f"Shape of loss: {loss.shape}")

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# if __name__ == "__main__":
#     # Import the necessary libraries
#     # Create an instance of the MatAttEncoder class
#     model = MatAttDecoder(n_tokens=10)

#     # Generate a random input tensor
#     input_tensor = torch.rand(64, 10, 128) # B T C

#     # Pass the input tensor through the model
#     output_tensor = model(input_tensor)



if __name__ == "__main__":
    # Import the necessary libraries
    # Create an instance of the MatAttEncoder class
    model = TransformerDecoder()

    # Generate a random input tensor
    input_tensor = torch.rand(64, 10, 128) # B T C

    # Pass the input tensor through the model
    output_tensor = model(input_tensor)

