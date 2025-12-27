import torch
import random
import time
import math
import tiktoken
import json
from pathlib import Path
from datetime import datetime

from torch.nn import functional as F
from typing import Tuple, Dict, Any
# torch 2.0.x exposes autocast in torch.amp, but GradScaler is in torch.cuda.amp.
try:
    from torch.amp import autocast  # type: ignore
except Exception:  # pragma: no cover
    from torch.cuda.amp import autocast  # type: ignore
from torch.cuda.amp import GradScaler
from argparse import ArgumentParser
from train_utils import save_checkpoint, get_model_info, get_model_and_config
from data import DataLoader


class TrainingLogger:
    def __init__(self, log_dir: Path, embedding_type: str, config: Dict[str, Any]):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"training_{embedding_type}_{timestamp}.log"
        self.metrics_file = self.log_dir / f"metrics_{embedding_type}_{timestamp}.json"
        
        self.metrics = []
        
        # Save config
        config_file = self.log_dir / f"config_{embedding_type}_{timestamp}.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.log_file_handle = open(self.log_file, 'w')
        self.log(f"Training started at {datetime.now()}")
        self.log(f"Configuration: {json.dumps(config, indent=2)}")

    def log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        print(formatted_message)
        self.log_file_handle.write(formatted_message + "\n")
        self.log_file_handle.flush()

    def log_metrics(self, step: int, metrics: Dict[str, float]):
        metrics_entry = {"step": step, "timestamp": datetime.now().isoformat(), **metrics}
        self.metrics.append(metrics_entry)
        
        # Save metrics incrementally
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def close(self):
        self.log_file_handle.close()


def train(cfg, model, tokenizer, batch_size: int, embedding_type: str, 
          max_steps: int = 5000, log_interval: int = 50, eval_interval: int = 500,
          save_interval: int = 1000, output_dir: str = "outputs"):
    
    # Setup directories
    output_path = Path(output_dir)
    log_dir = output_path / "logs"
    checkpoint_dir = output_path / "checkpoints"
    
    # Setup logger
    config_dict = {
        'embedding_type': embedding_type,
        'batch_size': batch_size,
        'max_steps': max_steps,
        'model_config': cfg.dict() if hasattr(cfg, 'dict') else cfg.__dict__,
        'model_info': get_model_info(model)
    }
    
    logger = TrainingLogger(log_dir, embedding_type, config_dict)
    logger.log(f"Model info: {get_model_info(model)}")
    
    # Move model to device
    model.to(cfg.device)
    logger.log(f"Model moved to device: {cfg.device}")
    
    # Setup data loaders
    data_dir = Path("data/tinystories")
    train_dl = DataLoader(data_dir / "train.txt", tokenizer, batch_size, cfg.ctx_size)
    val_dl = DataLoader(data_dir / "validation.txt", tokenizer, batch_size, cfg.ctx_size)
    
    logger.log(f"Training batches: {len(train_dl)}, Validation batches: {len(val_dl)}")
    
    # Setup optimizer and scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    use_amp = cfg.device in ["cuda", "mps"]
    scaler = GradScaler() if use_amp else None
    
    pad_token_id = tokenizer.encode("<|endoftext|>", allowed_special="all")[0]

    @torch.inference_mode()
    def evaluate() -> Tuple[float, float]:
        model.eval()
        total_loss, total_tokens = 0.0, 0
        
        for x, y in val_dl:
            x, y = x.to(cfg.device), y.to(cfg.device)
            
            with autocast(device_type=cfg.device.replace("mps", "cpu"), enabled=use_amp):
                logits = model(x)
                logits = logits.view(-1, logits.size(-1))
                loss = F.cross_entropy(
                    logits, y.view(-1), 
                    ignore_index=pad_token_id, 
                    reduction="none"
                ).reshape_as(y)

            mask = (y != pad_token_id)
            total_loss += (loss * mask).sum().item()
            total_tokens += mask.sum().item()
        
        model.train()
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        ppl = math.exp(min(avg_loss, 10))  # Cap to prevent overflow
        return avg_loss, ppl

    def generate_sample() -> str:
        model.eval()
        with torch.no_grad():
            prompt = "I am going to"
            x = torch.tensor([tokenizer.encode(prompt)]).to(cfg.device)
            out = model.generate(x, max_new_tokens=30)
            decoded_str = tokenizer.decode(out.detach().tolist()[0])
        model.train()
        return decoded_str

    # Training loop
    model.train()
    start_time = time.time()
    best_val_loss = float('inf')
    
    logger.log("Starting training...")
    
    step = 0
    for epoch in range(1000):  # Large number, will break based on max_steps
        for x, y in train_dl:
            if step >= max_steps:
                break
                
            x, y = x.to(cfg.device), y.to(cfg.device)
            
            # Forward pass
            with autocast(device_type=cfg.device.replace("mps", "cpu"), enabled=use_amp):
                logits = model(x)
                logits = logits.view(-1, logits.size(-1))
                loss = F.cross_entropy(logits, y.view(-1), ignore_index=pad_token_id)

            # Backward pass
            optimizer.zero_grad()
            if use_amp and scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # Logging
            if (step + 1) % log_interval == 0:
                logger.log(f"Step {step + 1:,} | Train Loss: {loss.item():.4f}")
                logger.log_metrics(step + 1, {"train_loss": loss.item()})

            # Evaluation
            if (step + 1) % eval_interval == 0 or step == max_steps - 1:
                val_loss, ppl = evaluate()
                sample_text = generate_sample()
                
                logger.log(f"Step {step + 1:,} | Val Loss: {val_loss:.4f} | PPL: {ppl:.2f}")
                logger.log(f"Sample generation: {sample_text}")
                
                logger.log_metrics(step + 1, {
                    "train_loss": loss.item(),
                    "val_loss": val_loss,
                    "perplexity": ppl
                })
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = save_checkpoint(
                        model, optimizer, step + 1, val_loss, 
                        checkpoint_dir, f"{embedding_type}_best"
                    )
                    logger.log(f"New best model saved: {checkpoint_path}")

            # Periodic checkpoint saving
            if (step + 1) % save_interval == 0:
                checkpoint_path = save_checkpoint(
                    model, optimizer, step + 1, loss.item(), 
                    checkpoint_dir, embedding_type
                )
                logger.log(f"Checkpoint saved: {checkpoint_path}")

            step += 1
        
        if step >= max_steps:
            break

    # Final checkpoint
    final_checkpoint = save_checkpoint(
        model, optimizer, step, loss.item(), 
        checkpoint_dir, f"{embedding_type}_final"
    )
    
    training_time = time.time() - start_time
    logger.log(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    logger.log(f"Final checkpoint saved: {final_checkpoint}")
    logger.log(f"Best validation loss: {best_val_loss:.4f}")
    
    logger.close()


def main():
    parser = ArgumentParser(description="Train LLaMA-style models with different positional encodings")
    parser.add_argument("--embedding", type=str, choices=["alibi", "rope", "pos"], required=True,
                       help="Type of positional embedding to use")
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device selector. Use an integer GPU index (default: 0), or 'cpu' / 'mps'.",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum number of training steps")
    parser.add_argument("--log_interval", type=int, default=50, help="Steps between logging")
    parser.add_argument("--eval_interval", type=int, default=500, help="Steps between evaluations")
    parser.add_argument("--save_interval", type=int, default=1000, help="Steps between checkpoints")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Setup device
    device_arg = args.device.strip().lower()
    if device_arg in {"cpu", "mps"}:
        device = device_arg
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Requested device 'mps' but torch.backends.mps is not available.")
    else:
        # Treat as CUDA device index
        try:
            cuda_index = int(device_arg)
        except ValueError as e:
            raise ValueError(
                f"Invalid --device '{args.device}'. Use an integer GPU index (e.g. 0) or 'cpu'/'mps'."
            ) from e

        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested CUDA device index {cuda_index}, but CUDA is not available."
            )
        if cuda_index < 0 or cuda_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested CUDA device index {cuda_index}, but only {torch.cuda.device_count()} CUDA device(s) are visible."
            )
        torch.cuda.set_device(cuda_index)
        device = "cuda"
    
    print(f"Using device: {device}" + (f":{torch.cuda.current_device()}" if device == "cuda" else ""))
    
    # Setup tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Get model and config
    model, cfg = get_model_and_config(args.embedding, tokenizer, device)
    
    # Start training
    train(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        embedding_type=args.embedding,
        max_steps=args.max_steps,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()