import math
from typing import Iterable
from pytorch_transformer_from_scratch.log_utils import shape_logger
import torch
import torch.nn as nn

# TODO: Add type hints
# TODO: Add docstrings
# TODO: Add logs indicating shape of tensors


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        rval = self.embedding(x) * math.sqrt(self.d_model)

        shape_logger.debug(f"InputEmbedding size(input): {x.size()}")
        shape_logger.debug(f"InputEmbedding size(output): {rval.size()}")


        return rval


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # 2i
        pe[:, 1::2] = torch.cos(position * div_term)  # 2i+1
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        rval = self.dropout(x)

        shape_logger.debug(f"PositionalEncoding size(input): {x.size()}")
        shape_logger.debug(f"PositionalEncoding size(output): {rval.size()}")

        return rval


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # Multiplied
        self.bias = nn.Parameter(torch.zeros(1))  # Added

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        rval = self.alpha * (x - mean) / (std + self.eps) + self.bias

        shape_logger.debug(f"LayerNormalization size(input): {x.size()}")
        shape_logger.debug(f"LayerNormalization size(mean): {mean.size()}")
        shape_logger.debug(f"LayerNormalization size(std): {std.size()}")
        shape_logger.debug(f"LayerNormalization size(output): {rval.size()}")
        return rval


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout

        self.linear1 = nn.Linear(d_model, d_ff)  # W1 * x + b1
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model)  # W2 * x + b2

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        rval = self.linear2(self.dropout(torch.relu(self.linear1(x))))

        shape_logger.debug(f"FeedForwardBlock size(input): {x.size()}")
        shape_logger.debug(f"FeedForwardBlock size(output): {rval.size()}")

        return rval


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = dropout

        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (batch_size, h, seq_len, d_k) @ (batch_size, h, d_k, seq_len) -> (batch_size, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # (batch_size, h, seq_len, seq_len)
        attention_scores = torch.softmax(attention_scores, dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (batch_size, h, seq_len, seq_len) @ (batch_size, h, seq_len, d_k) -> (batch_size, h, seq_len, d_k) , (batch_size, h, seq_len, seq_len)
        return attention_scores @ value, attention_scores

    def forward(self, q, k, v, mask):
        # Multiply with weight matrices
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Split into h heads
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, h, d_k) -> (batch_size, h, seq_len, d_k)
        query = query.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)

        # Apply attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )

        # Concatenate heads
        # (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Apply final linear layer
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        rval = self.w_o(x)

        shape_logger.debug(f"MultiHeadAttentionBlock size(input.q): {q.size()}")
        shape_logger.debug(f"MultiHeadAttentionBlock size(input.k): {k.size()}")
        shape_logger.debug(f"MultiHeadAttentionBlock size(input.v): {v.size()}")
        shape_logger.debug(f"MultiHeadAttentionBlock size(input.mask): {mask.size()}")
        shape_logger.debug(f"MultiHeadAttentionBlock size(x): {x.size()}")
        shape_logger.debug(f"MultiHeadAttentionBlock size(attention_scores): {self.attention_scores.size()}")
        shape_logger.debug(f"MultiHeadAttentionBlock size(rval): {rval.size()}")

        return rval

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        # In the research paper "Attention is all you need", they apply norm after adding x and sublayer. That is the followig code:
        # rval = self.norm(x + self.dropout(sublayer(x)))

        # But in many popular implementations the norm is applied before
        rval = x + self.dropout(sublayer(self.norm(x)))

        shape_logger.debug(f"ResidualConnection size(input.x): {x.size()}")
        shape_logger.debug(f"ResidualConnection size(sublayer(x)): {sublayer(x).size()}")
        shape_logger.debug(f"ResidualConnection size(output): {rval.size()}")

        return rval


class EncoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections: Iterable[ResidualConnection] = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        x = self.residual_connections[1](x, self.feed_forward_block)
        rval = x

        shape_logger.debug(f"EncoderBlock size(input.x): {x.size()}")
        shape_logger.debug(f"EncoderBlock size(input.src_mask): {src_mask.size()}")
        shape_logger.debug(f"EncoderBlock size(output): {rval.size()}")

        return rval

class Encoder(nn.Module):
    def __init__(
        self,
        encoder_blocks: Iterable[EncoderBlock],
    ) -> None:
        super().__init__()
        self.encoder_blocks = encoder_blocks
        self.norm = LayerNormalization()

    def forward(self, x, src_mask):
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, src_mask)
        rval = self.norm(x)

        shape_logger.debug(f"Encoder size(input.x): {x.size()}")
        shape_logger.debug(f"Encoder size(input.src_mask): {src_mask.size()}")
        shape_logger.debug(f"Encoder size(output): {rval.size()}")

        return rval



class DecoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections: Iterable[ResidualConnection] = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, src_mask
            ),
        )
        x = self.residual_connections[2](x, self.feed_forward_block)
        rval = x

        shape_logger.debug(f"DecoderBlock size(input.x): {x.size()}")
        shape_logger.debug(f"DecoderBlock size(input.encoder_output): {encoder_output.size()}")
        shape_logger.debug(f"DecoderBlock size(input.src_mask): {src_mask.size()}")
        shape_logger.debug(f"DecoderBlock size(input.tgt_mask): {tgt_mask.size()}")
        shape_logger.debug(f"DecoderBlock size(output): {rval.size()}")

        return rval


class Decoder(nn.Module):
    def __init__(
        self, decoder_blocks: Iterable[DecoderBlock]  # TODO check type hint
    ) -> None:
        super().__init__()
        self.decoder_blocks = decoder_blocks
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, encoder_output, src_mask, tgt_mask)
        rval = self.norm(x)

        shape_logger.debug(f"Decoder size(input.x): {x.size()}")
        shape_logger.debug(f"Decoder size(input.encoder_output): {encoder_output.size()}")
        shape_logger.debug(f"Decoder size(input.src_mask): {src_mask.size()}")
        shape_logger.debug(f"Decoder size(input.tgt_mask): {tgt_mask.size()}")
        shape_logger.debug(f"Decoder size(output): {rval.size()}")

        return rval


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)
        rval = torch.log_softmax(self.linear(x), dim=-1)

        shape_logger.debug(f"ProjectionLayer size(input): {x.size()}")
        shape_logger.debug(f"ProjectionLayer size(output): {rval.size()}")

        return rval


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbedding,
        tgt_embed: InputEmbedding,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection: ProjectionLayer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection = projection

    def encode(self, src, src_mask):
        rval = self.encoder(self.src_pos(self.src_embed(src)), src_mask)
        
        shape_logger.debug(f"Transformer.encode size(ipnut.src): {src.size()}")
        shape_logger.debug(f"Transformer.encode size(ipnut.src_mask): {src_mask.size()}")
        shape_logger.debug(f"Transformer.encode size(output): {rval.size()}")

        return rval

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        rval = self.decoder(
            self.tgt_pos(self.tgt_embed(tgt)), encoder_output, src_mask, tgt_mask
        )

        shape_logger.debug(f"Transformer.decode size(ipnut.encoder_output): {encoder_output.size()}")
        shape_logger.debug(f"Transformer.decode size(ipnut.src_mask): {src_mask.size()}")
        shape_logger.debug(f"Transformer.decode size(ipnut.tgt): {tgt.size()}")
        shape_logger.debug(f"Transformer.decode size(ipnut.tgt_mask): {tgt_mask.size()}")
        shape_logger.debug(f"Transformer.decode size(output): {rval.size()}")

        return rval

    def project(self, x):
        rval = self.projection(x)

        shape_logger.debug(f"Transformer.project size(ipnut): {x.size()}")
        shape_logger.debug(f"Transformer.project size(output): {rval.size()}")
        
        return rval


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
) -> Transformer:
    """Builds, initializes and returns the transformer model from the hyperparameters provided as args. The transformer is built following the paper "Attention is all you need". Weights are initialized using Xavier uniform.

    Args:
        src_vocab_size (int): Size of the source vocabulary.
        tgt_vocab_size (int): Size of the target vocabulary.
        src_seq_len (int): Maximum length of the source sequence.
        tgt_seq_len (int): Maximum length of the target sequence.
        d_model (int, optional): Dimensionality of the model. Defaults to 512.
        N (int, optional): Number of encoder and decoder blocks. Defaults to 6.
        h (int, optional): Number of attention heads. Defaults to 8.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        d_ff (int, optional): Dimensionality of the feed-forward layer. Defaults to 2048.

    Returns:
        Transformer: The initialized transformer model.
    """
    # Create embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    # Create positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create encoder blocks
    encoder_blocks: list[EncoderBlock] = []
    for _ in range(N):
        self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create decoder blocks
    decoder_blocks: list[DecoderBlock] = []
    for _ in range(N):
        self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            self_attention_block, cross_attention_block, feed_forward_block, dropout
        )
        decoder_blocks.append(decoder_block)

    # Create encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create projection layer
    projection = ProjectionLayer(d_model, tgt_vocab_size)

    # Create transformer
    transformer = Transformer(
        encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection
    )

    # Initialize the weights using Xavier uniform
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
