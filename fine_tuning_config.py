# 2.3 - Fine-Tuning Pipeline (Minimal Config)
# LoRA configuration + Training config (no actual training in 45min)

from dataclasses import dataclass, asdict
from typing import List, Dict, Any

@dataclass
class LoRAConfig:
    """LoRA configuration"""
    r: int = 8  # LoRA rank
    lora_alpha: int = 16  # LoRA scaling
    target_modules: List[str] = None
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]

@dataclass
class TrainingConfig:
    """SFT Training configuration"""
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    num_epochs: int = 1
    warmup_steps: int = 10
    output_dir: str = "./models/checkpoint"
    save_strategy: str = "epoch"
    logging_steps: int = 10

class MinimalFineTuningPipeline:
    """Minimal fine-tuning pipeline (config-only for 45min)"""

    def __init__(self, use_qlora: bool = False):
        self.use_qlora = use_qlora
        self.lora_config = LoRAConfig()
        self.training_config = TrainingConfig()
        self.run_history: List[Dict[str, Any]] = []

    def get_lora_config(self) -> Dict[str, Any]:
        """Get LoRA configuration as dict"""
        return asdict(self.lora_config)

    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration as dict"""
        return asdict(self.training_config)

    def log_metrics(self, step: int, loss: float, eval_loss: float = None):
        """Log training metrics"""
        metrics = {
            "step": step,
            "loss": loss,
            "eval_loss": eval_loss
        }
        self.run_history.append(metrics)
        print(f"Step {step}: loss={loss:.4f}", end="")
        if eval_loss:
            print(f", eval_loss={eval_loss:.4f}", end="")
        print()

    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        if not self.run_history:
            return {"status": "no_training"}

        return {
            "total_steps": len(self.run_history),
            "final_loss": self.run_history[-1]["loss"],
            "lora_params_reduction": "99%",
            "config": {
                "lora": self.get_lora_config(),
                "training": self.get_training_config()
            }
        }


# Example usage
if __name__ == "__main__":
    pipeline = MinimalFineTuningPipeline(use_qlora=False)

    print("=== LoRA Configuration ===")
    lora_cfg = pipeline.get_lora_config()
    print(f"LoRA Rank: {lora_cfg['r']}")
    print(f"LoRA Alpha: {lora_cfg['lora_alpha']}")
    print(f"Target Modules: {lora_cfg['target_modules']}")
    print(f"Dropout: {lora_cfg['lora_dropout']}\n")

    print("=== Training Configuration ===")
    train_cfg = pipeline.get_training_config()
    print(f"Batch Size: {train_cfg['batch_size']}")
    print(f"Learning Rate: {train_cfg['learning_rate']}")
    print(f"Epochs: {train_cfg['num_epochs']}")
    print(f"Output Dir: {train_cfg['output_dir']}\n")

    print("=== Simulated Training Log ===")
    for step in [1, 5, 10]:
        loss = 2.0 - (step * 0.1)
        pipeline.log_metrics(step, loss)

    print("\n=== Training Summary ===")
    summary = pipeline.get_training_summary()
    print(f"Total Steps: {summary['total_steps']}")
    print(f"Final Loss: {summary['final_loss']:.4f}")
    print(f"Parameter Reduction: {summary['lora_params_reduction']}")
