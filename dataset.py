import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tokenizers import Tokenizer


class BilingualDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer_src: Tokenizer,
        tokenizer_tgt: Tokenizer,
        src_lang,
        tgt_lang,
        seq_len,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor(
            [tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64  # TODO check
        )
        self.pad_token = torch.tensor(
            [tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        src_tgt_pair = self.dataset[index]
        src_text = src_tgt_pair["translation"][self.src_lang]
        tgt_text = src_tgt_pair["translation"][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = (
            self.seq_len - len(enc_input_tokens) - 2
        )  # -2 for SOS and EOS
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # -1 for SOS

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sequence length is too short")

        # Add SOS and EOS to the source text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * enc_num_padding_tokens, dtype=torch.int64
                ),
            ]
        )

        # Add SOS to the decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ]
        )

        # Add EOS to the label (what we expect from the decoder)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        rval = {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),  # (1,1,seq_len)
            "decoder_mask": (decoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int()
            & causal_mask(decoder_input.size(0)),  # (1,1,seq_len)
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

        shape_logger.debug(f"size(encoder_input): {rval['encoder_input'].size()}")
        shape_logger.debug(f"size(decoder_input): {rval['decoder_input'].size()}")
        shape_logger.debug(f"size(encoder_mask): {rval['encoder_mask'].size()}")
        shape_logger.debug(f"size(decoder_mask): {rval['decoder_mask'].size()}")
        shape_logger.debug(f"size(label): {rval['label'].size()}")

        return rval


def causal_mask(size: int):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    rval = (mask == 0)
    shape_logger.debug(f"size(causal_mask): {rval.size()}")
    return rval
