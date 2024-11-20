import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from sae import SaeConfig, SaeTrainer, TrainConfig
from sae.data import chunk_and_tokenize_streaming

if __name__ == "__main__":
    model_name = "EleutherAI/pythia-70m-deduped"

    dataset = load_dataset(
        "allenai/c4",
        "en",
        split="train",
        trust_remote_code=True,
        streaming=True,
    )

    batch_size = 4
    max_seq_len = 1024

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = chunk_and_tokenize_streaming(dataset, tokenizer, max_seq_len=max_seq_len)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
    )

    model = AutoModel.from_pretrained(
        model_name,
        device_map={"": "cuda"},
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    
    cfg = TrainConfig(
        SaeConfig(
            expansion_factor = 16,
            normalize_decoder = True,
            k = 64,
            multi_topk = False,
            init = "orthogonal"
        ),
        hookpoints=["layers.2", "layers.3", "layers.4"],
        max_seq_len=max_seq_len,
        num_training_tokens=500_000_000,
        batch_size=batch_size, # token per batch: 4 * 1024 = 4096
        save_every=25_000,
        lr=3e-4,
        lr_warmup_steps=0.0005,
    )
    trainer = SaeTrainer(cfg, data_loader, model)
    trainer.fit()
