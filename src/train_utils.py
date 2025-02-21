import os
import gc
import re
import time
import glob
import warnings
import math
import torch
import random
import logging
import numpy as np
from typing import Tuple, Any
from datetime import timedelta
from torch import autocast
from torchinfo import summary
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer, PreTrainedTokenizer
from src.model import Transformer


warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('high')
logger = logging.getLogger(__name__)


class WarmUpCosineLR(LambdaLR):
    """
        Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """

    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        """_summary_

        Args:
            optimizer : optimizer used
            warmup_steps : steps to warm up
            t_total : total steps
            cycles : Learning rate tranversal. Defaults to 0.5.
            last_epoch : last epoch. Defaults to -1.
        """
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step) -> float:
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


def set_all_seeds(seed) -> None:
    """
    Ensures reproducible behaviour by resetting all seeds with the seed given by `seed`.
    Moreover, additional parameters are set to ensure deterministic behaviour.

    Reference:
    [1] https://pytorch.org/docs/stable/notes/randomness.html, Accessed: 2021-07-19

    Args:
        seed: The desired seed to be set
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_model_summary(config) -> None:
    # Initialize model
    model = Transformer(config)
    # Generate Summary
    input_tensor = torch.randint(0, config.vocab_size, (1, config.max_seq_len))
    # mask = torch.tril(torch.ones(2048, 2048)).unsqueeze(0).type_as(input_tensor)
    logger.info(f"="*95)
    logger.info(f"Model Summary".center(95, " "))
    summary(model, input_data=input_tensor, device="cpu", depth=2)
    del model, input_tensor
    gc.collect()


def load_checkpoint(checkpoint_path) -> Any | None:
    """Safely load checkpoint with error handling"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        return checkpoint
    except (RuntimeError, EOFError, Exception) as e:
        logger.info(f"[ERROR]Failed to load checkpoint at {checkpoint_path}")
        logger.info(f"Error: {str(e)}")
        logger.info("Starting training from scratch")
        return None


def save_checkpoint(checkpoint_path, step, loss, model, optimizer, scheduler=None, scaler=None) -> None:
    checkpoint = {
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if scaler:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    
    torch.save(checkpoint, checkpoint_path)


def get_data_loader(batch_size, seq_len) -> Tuple [ DataLoader, PreTrainedTokenizer ]:
    # dataset
    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", streaming=True, split="train")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    tokenizer.pad_token = tokenizer.eos_token

    # encoder
    def encode(item):# -> dict[str, Any]:
        tokens = tokenizer(
            item["text"],
            padding="max_length",
            truncation=True,
            max_length=seq_len,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"]
        input_ids = torch.clamp(input_ids, max=len(tokenizer)-1)   # max == vocab_size
        attn_mask = tokens["attention_mask"]

        return {
            "input_ids": input_ids.squeeze(0).clone().detach(),
            "attn_mask": attn_mask.squeeze(0).clone().detach()
        }
    dataset = dataset.map(encode, batched=True, remove_columns=dataset.column_names, batch_size=batch_size)
    return dataset, tokenizer


def get_sample_text(train_iter, tokenizer) -> list:
    sample_text = []
    sample_text_count = 1
    for i in range(sample_text_count):
        if i >= sample_text_count:
            break

        batch = next(train_iter)
        decoded_text = tokenizer.decode(batch["input_ids"], skip_special_tokens=True)

        prompt = " ".join(decoded_text.split()[:10])
        sample_text.append({
            "full_text": decoded_text[:200] + "...",
            "prompt": prompt,
            "input_ids": batch["input_ids"],
            "attn_mask": batch["attn_mask"]
        })

    return sample_text


def train(config, 
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
          scaler=None) -> None:
       
    step = 1  # Initialize step to 1
    
    # Try to find latest checkpoint
    checkpoint_files = glob.glob(os.path.join(config.checkpoint_path, 'step_*.pth'))
    latest_checkpoint = None
    
    if checkpoint_files:

        logger.info(f"checkpoints {checkpoint_files}")
        
        # Sort checkpoints by step number
        checkpoint_files.sort(key=lambda x: int(re.search(r'step_(\d+)', x).group(1)))
        rest_ckpt, latest_checkpoint = checkpoint_files[:-1], checkpoint_files[-1]
        
        logger.info(f"Found checkpoint: {latest_checkpoint}")
        checkpoint = load_checkpoint(latest_checkpoint)
       
        if checkpoint is not None:
            try:
                # handling cached_keys & cached_values
                model.load_state_dict({k: v for k, v in checkpoint["model_state_dict"].items() if 'cached_keys' not in k and 'cached_values' not in k})
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if scheduler and checkpoint["scheduler_state_dict"]:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                if scaler and "scaler_state_dict" in checkpoint:
                    scaler.load_state_dict(checkpoint["scaler_state_dict"])
                step = checkpoint['step'] + 1 # Use loaded step directly
                logger.info(f"Resumed from step {step}")
            except Exception as e:
                logger.info(f"[ERROR]Failed to restore checkpoint state: {str(e)}")
                logger.info("Starting training from step 1")
                step = 1  # Reset to 1 if checkpoint loading fails
                
            
            logger.info("Removing old weights")
            _ = [os.system(f"rm -rf {c}") for c in rest_ckpt]

    # Skip batches if resuming from checkpoint
    if step > 1:
        logger.info(f"Skipping {(step -1) * config.grad_accum_steps} batches to resume position...")
        for _ in range((step-1) * config.grad_accum_steps):
            _ = next(train_iter)
        logger.info(f"Resuming Training from step: {step}")
        if step > config.breakpoint_step:
            breakpoint_step = None
        else:
            breakpoint_step = config.breakpoint_step
    else:    
        logger.info(f"Initiate Training from step: {step}")
        breakpoint_step = config.breakpoint_step
        
    while step <= config.steps:

        if breakpoint_step:
            if step > breakpoint_step:
                logger.info(f"First Milestone of {breakpoint_step} steps completed, stopping the training!")
                break
        
        step_start_time = time.time()
        accumulated_loss = 0
        
        # Training step
        for _ in range(config.grad_accum_steps):
            batch = next(train_iter)
            input_ids = batch["input_ids"].to(device)
            
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            
            if use_amp:
                with autocast(device_type=device, dtype=dtype):
                    outputs = model(input_ids)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))
                    loss = loss / config.grad_accum_steps
                
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                outputs = model(input_ids)
                loss = torch.nn.functional.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    input_ids.view(-1)
                )
                loss = loss / config.grad_accum_steps
                loss.backward()
            
            accumulated_loss += loss.item() * config.grad_accum_steps
        
        # Optimizer step
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        scheduler.step()
        optimizer.zero_grad()
               
        # Calculate step time and tokens/sec
        torch.cuda.synchronize()
        total_step_time = time.time() - step_start_time
        batch_tokens = input_ids.size(0) * input_ids.size(1)
        total_tokens = batch_tokens * config.grad_accum_steps
        tokens_per_sec = total_tokens / total_step_time

        # logging
        if step % config.log_interval == 0:
            avg_loss = accumulated_loss / config.log_interval
            logger.info(f"Step {step:4d}/{config.steps} | Avg. Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | "
                  f"Step Time: {str(timedelta(seconds=total_step_time))} | "
                  f"Tokens/sec: {tokens_per_sec:.2f} (accumulated over {config.grad_accum_steps} batches)")
        
        # Text generation
        if step % config.eval_interval == 0:
            logger.info(f"{'- '*40}")
            prompt = sample_text[0]['prompt']
            logger.info(f"Prompt: {prompt}")
            input_ids = tokenizer(prompt, 
                                  padding=True,
                                  truncation=True,
                                  max_length=config.max_seq_len,
                                  return_tensors="pt")["input_ids"].to(device)
            generated = model.generate(
                input_ids,
                max_length=64,
                min_length=28,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.8,
                top_k=50,
                top_p=0.95
            )
            # outputs = model(input_ids)
            # generated = torch.argmax(outputs, 1)
            generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            logger.info(f"Generated: {generated_text}")
            logger.info(f"{'- '*40}")
            model.train()

        # Increment step counter at the end
        step += 1
        
        # Checkpointing at every 1000 steps
        if step % config.save_interval == 0:
            checkpoint_path = f"{config.checkpoint_path}/step_{step}.pth"
            try:
                save_checkpoint(checkpoint_path, step, accumulated_loss, model, optimizer, scheduler, scaler)
                logger.info(f"Checkpoint saved to: {checkpoint_path} at step {step}")
            except Exception as e:
                logger.info(f"[ERROR]Failed to save checkpoint: {str(e)} at step {step}")
    
    logger.info("Training completed!")
    
    # Save final checkpoint and model
    final_checkpoint_path = f"step_final_{step}.pth"
    save_checkpoint(final_checkpoint_path, step, accumulated_loss, model, optimizer, scheduler, scaler)
    logger.info(f"Saved final checkpoint: {final_checkpoint_path}")
    
    # Save final model_state_dict in .pth format for HF Space
    model_save_path = f"{config.checkpoint_path}/smollm2_HF.pth"
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"Saved HF model to: {model_save_path}")
