import logging
import timeit
from dataclasses import asdict, dataclass, field

import torch
from tqdm import tqdm
from transformers import HfArgumentParser

import cs336_basics
from cs336_basics.dataset import Dataset
from cs336_basics.model import Transformer, cross_entropy
from cs336_basics.optimizer import AdamW, clip_grad_norm, get_cosine_lr
from cs336_basics.tokenizer import Tokenizer
from cs336_systems.model import annotated_scaled_dot_product_attention

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention


# parsing the training configuration
@dataclass
class TrainingConfig:
    # dataset parameters
    dataset_name: str = field(default="tinystories")
    context_length: int = field(default=256)
    batch_size: int = field(default=16)
    device: str | None = field(default="cuda" if torch.cuda.is_available() else "cpu")

    # model parameters (default values from GPT2 config)
    vocab_size: int | None = field(default=1000)
    num_layers: int | None = field(default=12)
    d_model: int | None = field(default=768)
    num_heads: int | None = field(default=12)
    d_ff: int | None = field(default=3072)
    dropout: float | None = field(default=0.1)
    init_from: str = field(default="scratch")

    # training parameters (additional adamW parameter use as default)
    total_iters: int | None = field(default=100)
    warmup_iters: int | None = field(default=5)
    lr_max: float | None = field(default=5e-4)
    lr_min: float | None = field(default=0)
    weight_decay: float | None = field(default=0.001)

    # logging parameters
    wandb_logging: bool | None = field(default=False)
    wandb_project: str | None = field(default="cs336")
    wandb_run_name: str | None = field(default="gpt")
    log_interval: int | None = field(default=None)
    eval_interval: int | None = field(default=None)
    eval_iters: int | None = field(default=100)
    gen_iters: int | None = field(default=500)

    def __post_init__(self):
        if self.warmup_iters is None:
            self.warmup_iters = int(self.total_iters * 0.01)
        if self.log_interval is None:
            self.log_interval = int(self.total_iters * 0.001)
        if self.eval_interval is None:
            self.eval_interval = int(self.total_iters * 0.01)
        if self.wandb_logging:
            assert self.wandb_project is not None, "wandb_project must be provided if wandb_logging is True"
            assert self.wandb_run_name is not None, "wandb_run_name must be provided if wandb_logging is True"
        # self.ablation = self.no_rmsnorm or self.parallel_layers or self.post_norm


# parsing config
parser = HfArgumentParser(TrainingConfig)
config = parser.parse_args_into_dataclasses()[0]
logging.info(f"Training with config: {asdict(config)}")

tokenizer = Tokenizer.from_files("outputs/tinystories_vocab.json", "outputs/tinystories_merges.txt")
config.vocab_size = tokenizer.vocab_size
dataset = Dataset(config.dataset_name, config.context_length, config.batch_size, device=config.device)
model = Transformer(
    num_layers=config.num_layers,
    vocab_size=config.vocab_size,
    context_length=config.context_length,
    d_model=config.d_model,
    num_heads=config.num_heads,
    d_ff=config.d_ff,
    dropout=config.dropout,
    device=config.device,
)
model.to(config.device)
# model = torch.compile(model)
optimizer = AdamW(model.parameters(), lr=config.lr_max, weight_decay=config.weight_decay)

warm_up_iters = 5

for _ in range(warm_up_iters):
    x, y = dataset.get_batch("train")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

for iter_num in tqdm(range(config.total_iters)):
    optimizer.zero_grad()
    lr = get_cosine_lr(iter_num, config.lr_max, config.lr_min, config.warmup_iters, config.total_iters)
    optimizer.set_lr(lr)
    x, y = dataset.get_batch("train")

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(x)
        loss = cross_entropy(logits, y)

        loss.backward()
        clip_grad_norm(model.parameters(), 1.0)
        optimizer.step()


# import matplotlib.pyplot as plt

# plt.figure(figsize=(14, 5))
# plt.subplot(1, 2, 1)
# plt.plot(forward_times, label="Forward Pass Time")
# plt.xlabel("Iteration")
# plt.ylabel("Time (seconds)")
# plt.title(
#     f"Forward | mean: {sum(forward_times) / len(forward_times):.4f} sec | std: {torch.std(torch.tensor(forward_times)):.4f} sec"
# )
# plt.legend()
# plt.subplot(1, 2, 2)
# plt.plot(backward_times, label="Backward Pass Time")
# plt.xlabel("Iteration")
# plt.ylabel("Time (seconds)")
# plt.title(
#     f"Backward | mean: {sum(backward_times) / len(backward_times):.4f} sec | std: {torch.std(torch.tensor(backward_times)):.4f} sec"
# )
# plt.legend()
# plt.tight_layout()
# plt.savefig("benchmark_times.png")
