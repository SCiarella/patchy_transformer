import time
import math
import torch
from numpy import log
from torch import nn
import sys


class PositionalEncoding(nn.Module):
    """
    Classic Attention-is-all-you-need positional encoding.
    From PyTorch docs.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
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


def generate_square_subsequent_mask(size: int):
    """Generate a triangular (size, size) mask. From PyTorch docs."""
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class Transformer(nn.Module):
    """
    Classic Transformer that both encodes and decodes.
    
    Prediction-time inference is done greedily.

    NOTE: start token is hard-coded to be 0, end token to be 1. If changing, update predict() accordingly.
    """

    def __init__(self, num_classes: int, max_output_length: int, dim: int = 2**5):
        super().__init__()

        # Parameters
        self.dim = dim
        self.max_output_length = max_output_length
        nhead = 2
        num_layers = 2
        dim_feedforward = dim

        # Encoder part
        self.embedding = nn.Embedding(num_classes, dim)
        self.pos_encoder = PositionalEncoding(d_model=self.dim)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=self.dim, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_layers
        )

        # Decoder part
        self.y_mask = generate_square_subsequent_mask(self.max_output_length)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(d_model=self.dim, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_layers
        )
        self.fc = nn.Linear(self.dim, num_classes)

        # It is empirically important to initialize weights properly
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)
      
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Input
            x: (B, Sx) with elements in (0, C) where C is num_classes
            y: (B, Sy) with elements in (0, C) where C is num_classes
        Output
            (B, C, Sy) logits
        """
        encoded_x = self.encode(x)  # (Sx, B, E)
        output = self.decode(y, encoded_x)  # (Sy, B, C)
        return output.permute(1, 2, 0)  # (B, C, Sy)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input
            x: (B, Sx) with elements in (0, C) where C is num_classes
        Output
            (Sx, B, E) embedding
        """
        x = x.permute(1, 0)  # (Sx, B, E)
        x = self.embedding(x) * math.sqrt(self.dim)  # (Sx, B, E)
        x = self.pos_encoder(x)  # (Sx, B, E)
        x = self.transformer_encoder(x)  # (Sx, B, E)
        return x

    def decode(self, y: torch.Tensor, encoded_x: torch.Tensor) -> torch.Tensor:
        """
        Input
            encoded_x: (Sx, B, E)
            y: (B, Sy) with elements in (0, C) where C is num_classes
        Output
            (Sy, B, C) logits
        """
        y = y.permute(1, 0)  # (Sy, B)
        y = self.embedding(y) * math.sqrt(self.dim)  # (Sy, B, E)
        y = self.pos_encoder(y)  # (Sy, B, E)
        Sy = y.shape[0]
        y_mask = self.y_mask[:Sy, :Sy].type_as(encoded_x)  # (Sy, Sy)
        output = self.transformer_decoder(y, encoded_x, y_mask)  # (Sy, B, E)
        output = self.fc(output)  # (Sy, B, C)
        return output

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method to use at inference time. Predict y from x one token at a time. This method is greedy
        decoding. Beam search can be used instead for a potential accuracy boost.

        Input
            x: (B, Sx) with elements in (0, C) where C is num_classes
        Output
            (B, C, Sy) logits
        """
        encoded_x = self.encode(x)
        
        output_tokens = (torch.ones((x.shape[0], self.max_output_length))).type_as(x).long() # (B, max_length)
        output_tokens[:, 0] = 0  # Set start token
        for Sy in range(1, self.max_output_length):
            y = output_tokens[:, :Sy]  # (B, Sy)
            output = self.decode(y, encoded_x)  # (Sy, B, C)
            #output = torch.argmax(output, dim=-1)  # (Sy, B) #argmax

            a = (torch.tensor([1, 2, 3, 4, 5])-1).to(output.device)
            m = nn.Softmax(dim=1)
            p = m(output[Sy-1,:,:])

            index = p.multinomial(num_samples=1, replacement=True).to(output.device)
            b = a[index].to(output.device)

            output_tokens[:, Sy] = b[:,0] #output[-1:]  # Set the last output token
        return output_tokens
