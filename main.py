import os
import warnings
import torch
import logging
from typing import Optional
from dataclasses import dataclass
from torch import GradScaler
from src.model import Transformer
from src.train_utils import set_all_seeds, get_data_loader, get_sample_text, WarmUpCosineLR, train


warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('high')
logging.basicConfig(format='[%(asctime)s - %(levelname)s] %(message)s', 
                    filename='training.log', 
                    datefmt="%Y-%m-%d %H:%M:%S",
                    encoding='utf-8', 
                    level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelArgs:
    # Arch params
    dim: int = 576
    intermediate_dim: int = 1536
    n_layers: int = 30
    n_heads: int = 9
    n_kv_heads: Optional[int] = 3
    vocab_size: int = 49152  # defined later by tokenizer
    norm_eps: float = 1.0e-05
    init_scale: float = 0.041666666666666664
    rope_theta: int = 10000
    dropout: float = 0.1

    # Training params
    seed: int = 42
    max_batch_size: int = 2
    max_seq_len: int = 2048
    steps: int = 5050
    breakpoint_step: int = 5000
    warmup_steps_frac: float = 0.5
    save_interval:int = 1000
    eval_interval:int = 500
    log_interval: int = 1
    grad_accum_steps: int = 8
    checkpoint_path = os.path.join(os.getcwd(), "checkpoints")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimizer
    initial_lr: float = 5e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1.0e-08
    weight_decay: float = 0.01
    use_fused: bool = True


def main() -> None:

    # Model config
    config = ModelArgs()
    logger.info(config)

    set_all_seeds(config.seed)

    # Get the best available device
    device = config.device
        
    # Create checkpoints directory if it doesn't exist
    os.makedirs(config.checkpoint_path, exist_ok=True)
       
    # Initialize mixed precision training
    use_amp = device != "cpu"
    dtype = torch.float32
    if use_amp:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False):
            dtype = torch.bfloat16
            scaler = None
        else:
            dtype = torch.float16
            scaler = GradScaler(device=device)
    else:
        scaler = None
    logger.info(f"Using dtype = {dtype}")
    
    # Move model to device and set dtype
    model = Transformer(config)
    model = model.to(device)

    # > Torch2.x only
    model = torch.compile(model)
    
    train_dataset, tokenizer = get_data_loader(batch_size=config.max_batch_size, seq_len=config.max_seq_len)
    train_iter = iter(train_dataset)
    sample_text = get_sample_text(train_iter, tokenizer)

    # Criterion
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.initial_lr, 
                                  betas=(config.adam_beta1, config.adam_beta2),
                                  eps=config.adam_eps,
                                  weight_decay=config.weight_decay,
                                  fused=config.use_fused
                                  )
    
    # Scheduler
    scheduler = WarmUpCosineLR(optimizer=optimizer, 
                               warmup_steps=config.warmup_steps_frac * config.steps, 
                               t_total=config.steps)

    # call train
    train(config, 
          model,
          train_iter,
          sample_text,
          tokenizer,
          optimizer,
          criterion,
          scheduler,
          use_amp,
          device,
          dtype,
          scaler)

    # setattr(config, "completed_steps", orig_steps + config.steps)
    logger.info("Training completed successfully!")
    

if __name__ == "__main__":
    main()
