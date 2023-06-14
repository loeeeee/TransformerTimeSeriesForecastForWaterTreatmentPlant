import math

import torch
from torch import nn

from typing import Tuple

import pandas as pd

class PositionalEncoding(nn.Module):
    """
    Copied from pytorch tutorial
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class TimeSeriesTransformer(nn.Module):
    """
    Loe learned a lot doing this
    """
    def __init__(self, 
                 input_size: int,
                 forecast_feature_size: int,
                 *args, 
                 embedding_dimension: int               = 512,
                 multi_head_attention_head_size: int    = 8,
                 num_of_encoder_layers: int             = 4,
                 num_of_decoder_layers: int             = 4,
                 **kwargs
                 ) -> None:
        """
        input_size: number of features
        forecast_feature_size: number of features to forecast
        embedding_dimension: the internal dimension of the model
        multi_head_attention_head_size: number of head of the attention layer, 
            applied to all the attention layers
        num_of_encoder_layers: literally
        num_of_decoder_layers: literally
        """
        super().__init__()

        # Input embedding
        self.encoder_input_layer = nn.Linear(
            in_features     = input_size, 
            out_features    = embedding_dimension
        )
        
        # Output embedding
        self.decoder_input_layer = nn.Linear(
            in_features     = forecast_feature_size,
            out_features    = embedding_dimension
        )
        
        # Final forecast output of decoder
        self.final_output = nn.Linear(
            in_features     = embedding_dimension,
            out_features    = forecast_feature_size
        )

        # Positional encoding layer after the encoder input
        self.encoder_positional_encoder_layer = PositionalEncoding(
            d_model         = embedding_dimension
        )

        # Positional encoding layer after the decoder input
        self.decoder_positional_encoder_layer = PositionalEncoding(
            d_model         = embedding_dimension
        )

        # Example encoder layer to pass into nn.TransformerEncoder
        example_encoder = nn.TransformerEncoderLayer(
            d_model         = embedding_dimension,
            nhead           = multi_head_attention_head_size,
        )
        encoder_norm_layer = nn.BatchNorm1d(
            num_features    = embedding_dimension, # TODO
        )

        # Build encoder with the encoder layers
        self.encoder = nn.TransformerEncoder(
            encoder_layer   = example_encoder,
            num_layers      = num_of_encoder_layers,
            norm            = None
        )

        # Example decoder layer to pass into nn.TransformerDecoder
        example_decoder = nn.TransformerDecoderLayer(
            d_model         = embedding_dimension,
            nhead           = multi_head_attention_head_size
        )
        decoder_norm_layer = nn.BatchNorm1d(
            num_features    = embedding_dimension,
        )

        # Build decoder with the decoder layers
        self.decoder = nn.TransformerDecoder(
            decoder_layer   = example_decoder,
            num_layers      = num_of_decoder_layers,
            norm            = None
        )
    
    def forward(self, 
                src: torch.Tensor, 
                tgt: torch.Tensor, 
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor
                ) -> torch.Tensor:
        # Before encoder
        src = self.encoder_input_layer(src)

        # Encoder positional encoding layer
        src = self.encoder_positional_encoder_layer(src)

        # Encoder
        src = self.encoder(src)

        # Before decoder
        tgt = self.decoder_input_layer(tgt)

        # Decoder positional encoding layer
        tgt = self.decoder_positional_encoder_layer(tgt)

        # Decoder
        combined = self.decoder(
            tgt         = tgt,
            memory      = src,
            tgt_mask    = tgt_mask,
            memory_mask = src_mask
        )

        # Final linear layer
        forecast = self.final_output(combined)

        return forecast
    

class TransformerDataset(torch.utils.data.Dataset):
    """
    Hand-crafted dataset for transformer
    knowledge_length: the amount of time series data the transformer model will know to make forecast
    forecast_length: the amount of the time series data the transformer model will forecast given the information
    """
    def __init__(self,
                 src: pd.DataFrame,
                 tgt: pd.DataFrame,
                 knowledge_length: int,
                 forecast_length: int
                 ) -> None:
        super().__init__()
        self.src = src
        self.tgt = tgt
        self.knowledge_length = knowledge_length
        self.forecast_length = forecast_length
        return
    
    def __len__(self) -> int:
        return (self.src.shape[0] - self.forecast_length - self.knowledge_length)
    
    def __getitem__(self, knowledge_start: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Here is what the src would look like:

        src = [xt-4, xt-3, xt-2, xt-1, xt]

        where x denotes the series we're dealing with, e.g. electricity prices.

        The objective is to predict tgt_y which would be:

        tgt_y = [xt+1, xt+2, xt+3]

        So our tgt , which the model needs as input in order to make its forecast for tgt_y , should be:

        tgt = [xt, xt+1, xt+2]
        """
        result_src = self.src[knowledge_start:
                               knowledge_start + self.knowledge_length]
        result_tgt = self.tgt[knowledge_start + self.knowledge_length - 1: 
                              knowledge_start + self.knowledge_length - 1 + self.forecast_length]
        result_tgt_y = self.tgt[knowledge_start + self.knowledge_length: 
                                knowledge_start + self.knowledge_length + self.forecast_length]      
        return result_src, result_tgt, result_tgt_y
    
def transformer_collate_fn(data):
    """
    src, tgt, tgt_y
    """
    result = [[], [], []]
    for i in range(3):
        result[i] = [torch.Tensor(temp[i]) for temp in data]
        result[i] = torch.stack(result[i])
        result[i] = result[i].permute(1, 0, 2)
    return result[0], result[1], result[2]
