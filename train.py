from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pytorch_transformer_from_scratch.config import get_config, get_weights_file_path
from pytorch_transformer_from_scratch.dataset import BilingualDataset, causal_mask
from pytorch_transformer_from_scratch.transformer_model import build_transformer


def get_all_sentences(dataset, lang):
    for item in dataset:
        yield item["translation"][lang]


def get_or_build_tokenizer(config, dataset, lang) -> Tokenizer:
    # TODO move tokenizers into a folder
    tokenizer_path = Path(config["tokenizer_path"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            min_frequency=2,
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            show_progress=True,
        )
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer: Tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_dataset(config):
    # We use the OPUS Books dataset from the HuggingFace. Only train split is available.
    dataset_raw = load_dataset(
        "opus_books", f"{config['src_lang']}-{config['tgt_lang']}", split="train"
    )

    # Build tokenizer
    tokenizer_src = get_or_build_tokenizer(config, dataset_raw, config["src_lang"])
    tokenizer_tgt = get_or_build_tokenizer(config, dataset_raw, config["tgt_lang"])

    # Keep 90% of the dataset for training and 10% for validation.
    train_ds_size = int(len(dataset_raw) * 0.9)
    val_ds_size = len(dataset_raw) - train_ds_size
    train_dataset_raw, val_dataset_raw = random_split(
        dataset_raw, [train_ds_size, val_ds_size]
    )

    train_ds = BilingualDataset(
        train_dataset_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["src_lang"],
        config["tgt_lang"],
        config["seq_len"],
    )
    val_ds = BilingualDataset(
        val_dataset_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["src_lang"],
        config["tgt_lang"],
        config["seq_len"],
    )

    max_seq_len_src = max(
        len(tokenizer_src.encode(sentence).ids)
        for sentence in get_all_sentences(dataset_raw, config["src_lang"])
    )
    max_seq_len_tgt = max(
        len(tokenizer_tgt.encode(sentence).ids)
        for sentence in get_all_sentences(dataset_raw, config["tgt_lang"])
    )

    print(f"Max sequence length for source language: {max_seq_len_src}")
    print(f"Max sequence length for target language: {max_seq_len_tgt}")

    train_dataloader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(
        val_ds, batch_size=1, shuffle=False
    )  # TODO is batch_size=1 or from configfor validation

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, src_vocab_size, tgt_vocab_size):
    model = build_transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_seq_len=config["seq_len"],
        tgt_seq_len=config["seq_len"],
        d_model=config["d_model"],
    )

    return model


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    model = get_model(
        config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()
    ).to(device)

    writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    if config["preload"]:
        model_filename = get_weights_file_path(config=config, epoch=config["preload"])
        print(f"Loading model weights from {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        optimizer.load_state_dict(state["optimizer_state_dict"])

    loss_function = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)  # (batch_size, seq_len)
            decoder_input = batch["decoder_input"].to(device)  # (batch_size, seq_len)
            encoder_mask = batch["encoder_mask"].to(
                device
            )  # (batch_size, 1, 1, seq_len)
            decoder_mask = batch["decoder_mask"].to(
                device
            )  # (batch_size, 1, seq_len, seq_len)

            # Run the forward pass
            encoder_output = model.encode(
                encoder_input, encoder_mask
            )  # (batch_size, seq_len, d_model)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )  # (batch_size, seq_len, d_model)
            projection_output = model.project(
                decoder_output
            )  # (batch_size, seq_len, tgt_vocab_size)

            label = batch["label"].to(device)  # (batch_size, seq_len)

            loss = loss_function(
                projection_output.view(-1, projection_output.size(-1)), label.view(-1)
            )
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar("Training loss", loss.item(), global_step)
            writer.flush()

            # Backprop the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        # Save the model weights
        model_filename = get_weights_file_path(config=config, epoch=f"{epoch:02d}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )


def main():
    config = get_config()
    train_model(config)


if __name__ == "__main__":
    main()
