import logging
import time
from dataclasses import asdict, dataclass, field

import torch
from dataset import Dataset
from optimizer import AdamW, clip_grad_norm, get_cosine_lr
from tokenizer import Tokenizer
from transformers import HfArgumentParser
from utils import generate, load_checkpoint, save_checkpoint

import wandb
from cs336_basics.model import Transformer, cross_entropy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# parsing the training configuration
@dataclass
class TrainingConfig:
    # dataset parameters
    dataset_name: str
    context_length: int
    batch_size: int
    device: str | None = field(default="cuda" if torch.cuda.is_available() else "cpu")

    # model parameters (default values from GPT2 config)
    vocab_size: int | None = field(default=50257)
    context_size: int | None = field(default=1024)
    num_layers: int | None = field(default=12)
    d_model: int | None = field(default=768)
    num_heads: int | None = field(default=12)
    d_ff: int | None = field(default=3072)
    dropout: float | None = field(default=0.1)
    # attn_pdrop: float | None = field(default=0.1)
    # resid_pdrop: float | None = field(default=0.1)
    init_from: str = field(default="scratch")

    # training parameters (additional adamW parameter use as default)
    total_iters: int | None = field(default=10 * (10**3))
    warmup_iters: int | None = field(default=None)
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

    # ablation studies
    # no_rmsnorm: bool | None = field(default=False)
    # parallel_layers: bool | None = field(default=False)
    # post_norm: bool | None = field(default=False)

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
if config.wandb_logging:
    import wandb

    wandb.init(project=config.wandb_project, name=config.wandb_run_name)
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
    device=config.device,
)
model.to(config.device)
optimizer = AdamW(model.parameters(), lr=config.lr_max, weight_decay=config.weight_decay)

if config.init_from != "scratch":
    ckpt_dir = f"outputs/checkpoints/{config.init_from}"
    iter_num = load_checkpoint(model, optimizer, ckpt_dir)


def eval():
    model.eval()
    total_loss = 0
    for _ in range(config.eval_iters):
        x, y = dataset.get_batch("val")
        x, y = x.to(config.device), y.to(config.device)
        with torch.no_grad():
            logits = model(x)
            loss = cross_entropy(logits, y)
            total_loss += loss.item()
    total_loss /= config.eval_iters
    logging.info(f"Iter: {iter_num}, Val loss: {loss.item():.4f}, LR: {lr:.6f}")
    if config.wandb_logging:
        wandb.log({"val/loss": total_loss, "lr": lr, "step": iter_num})
        save_checkpoint(model, optimizer, iter_num, f"outputs/checkpoints/{config.wandb_run_name}.pt")
    model.train()


if config.wandb_logging:
    wandb.init(project=config.wandb_project, name=config.wandb_run_name)
    wandb.config.update(asdict(config))

iter_num = 0
curr_time = time.time()
while iter_num < config.total_iters:
    optimizer.zero_grad()

    # core backward pass
    x, y = dataset.get_batch("train")
    logits = model(x)
    loss = cross_entropy(logits, y)
    loss.backward()
    clip_grad_norm(model.parameters(), 1.0)
    lr = get_cosine_lr(iter_num, config.lr_max, config.lr_min, config.warmup_iters, config.total_iters)
    optimizer.set_lr(lr)
    optimizer.step()
    finish_time = time.time()

    if iter_num % config.log_interval == 0:
        logging.info(
            f"Iter: {iter_num}, Train loss: {loss.item():.4f}, LR: {lr:.6f}, Time: {1000 * (finish_time - curr_time):.2f}ms"
        )
        if config.wandb_logging:
            wandb.log({"train/loss": loss.item(), "lr": lr, "step": iter_num})
    if iter_num % config.eval_interval == 0:
        eval()
    if iter_num % config.gen_iters == 0:
        prompt = "Once upon a time"
        gen_seq = generate(prompt, model, tokenizer, top_p=0.9, t=1.0, max_steps=100, device=config.device)
        gen_text = tokenizer.decode(gen_seq.tolist()[0])
        logging.info(f"Iter: {iter_num}, Prompt: {prompt}, Generated: {gen_text}")

    curr_time = finish_time
    iter_num += 1
