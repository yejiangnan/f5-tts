import argparse
import os
import shutil
from datetime import datetime
from importlib.resources import files

# Set NCCL timeout before importing torch (force set to override any existing value)
os.environ["NCCL_TIMEOUT"] = "1800"  # 30 minutes (in seconds)
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
print(f"[NCCL Config] NCCL_TIMEOUT={os.environ.get('NCCL_TIMEOUT')}")

from cached_path import cached_path
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from datetime import timedelta

from f5_tts.model import CFM, DiT
from f5_tts.model.trainer_grpo import GRPOTrainer
from f5_tts.model.grpo_dataset import GRPODataset
from f5_tts.model.utils import get_tokenizer
import torch
from torch.utils.data import Dataset


# -------------------------- Dataset Settings --------------------------- #
target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"  # 'vocos' or 'bigvgan'


# -------------------------- Argument Parsing --------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Train CFM Model")

    parser.add_argument(
        "--exp_name",
        type=str,
        default="F5TTS_v1_Base",
        choices=["F5TTS_v1_Base", "F5TTS_Base", "E2TTS_Base"],
        help="Experiment name",
    )
    parser.add_argument("--dataset_name", type=str, default="Emilia_ZH_EN", help="Name of the dataset to use")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--batch_size_per_gpu", type=int, default=3200, help="Batch size per GPU")
    parser.add_argument(
        "--batch_size_type", type=str, default="frame", choices=["frame", "sample"], help="Batch size type"
    )
    parser.add_argument("--max_samples", type=int, default=64, help="Max sequences per batch")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--num_warmup_updates", type=int, default=20000, help="Warmup updates")
    parser.add_argument("--save_per_updates", type=int, default=50000, help="Save checkpoint every N updates")
    parser.add_argument(
        "--keep_last_n_checkpoints",
        type=int,
        default=-1,
        help="-1 to keep all, 0 to not save intermediate, > 0 to keep last N checkpoints",
    )
    parser.add_argument("--last_per_updates", type=int, default=5000, help="Save last checkpoint every N updates")
    parser.add_argument("--finetune", action="store_true", help="Use Finetune")
    parser.add_argument("--pretrain", type=str, default=None, help="Path to the pretrained checkpoint file (.pt or .safetensors)")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to save checkpoints (default: ckpts/{dataset_name}_{save_name})")
    parser.add_argument(
        "--tokenizer", type=str, default="pinyin", choices=["pinyin", "char", "custom"], help="Tokenizer type"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to custom tokenizer vocab file (only used if tokenizer = 'custom')",
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        help="Log inferenced samples per ckpt save updates",
    )
    parser.add_argument("--logger", type=str, default=None, choices=[None, "wandb", "tensorboard"], help="logger")
    parser.add_argument(
        "--bnb_optimizer",
        action="store_true",
        help="Use 8-bit Adam optimizer from bitsandbytes",
    )

    parser.add_argument(
        "--save_name", 
        type=str,
        help="The name of the saved checkpoint"
    )
    # GRPO 音质与 reward 平衡：beta>0 用 KL 约束防止策略偏离参考导致音质下降；clip_range 过小会导致更新无效
    parser.add_argument("--beta", type=float, default=0.05, help="KL 散度权重，0 表示无 KL；建议 0.01~0.1 以在 reward 与音质间取得平衡")
    parser.add_argument("--clip_range", type=float, default=0.15, help="PPO clip 范围，建议 0.1~0.2；过小(如1e-4)会导致 policy 更新几乎无效")

    return parser.parse_args()


# -------------------------- Training Settings -------------------------- #


def main():
    args = parse_args()
    
    # 提前初始化 Accelerator，便于多机训练时在 copy pretrain 后做 barrier
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    log_with = args.logger if args.logger == "wandb" else None
    accelerator = Accelerator(
        log_with=log_with,
        kwargs_handlers=[ddp_kwargs, init_kwargs],
        gradient_accumulation_steps=32,  # 与 GRPOTrainer 中一致
    )
    
    # 设置 checkpoint 保存目录
    if args.checkpoint_dir:
        checkpoint_path = args.checkpoint_dir
    else:
        checkpoint_path = str(files("f5_tts").joinpath(f"../../ckpts/{args.dataset_name}_{args.save_name}"))
    
    os.makedirs(checkpoint_path, exist_ok=True)
    if accelerator.is_main_process:
        shutil.copy2(args.pretrain, f"{checkpoint_path}/pretrained_ckpt.pt")
    accelerator.wait_for_everyone()

    # Model parameters based on experiment name
    wandb_resume_id = None
    model_cls = DiT
    model_cfg = dict(
        dim=1024,
        depth=22,
        heads=16,
        ff_mult=2,
        text_dim=512,
        conv_layers=4,
        emphasis_enhanced="transfomer",
    )
  
    # Use the tokenizer and tokenizer_path provided in the command line arguments
    tokenizer = "pinyin"
    tokenizer_path = "data/sft_data_pinyin/vocab.txt"
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        vocab_char_map = {}
        for i, char in enumerate(f):
            vocab_char_map[char[:-1]] = i
    vocab_size = len(vocab_char_map)
    assert vocab_char_map[" "] == 0, "make sure space is of idx 0 in vocab.txt, cuz 0 is used for unknown char"



    mel_spec_kwargs = dict(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    )

    
    train_dataset = GRPODataset(
        data_path="grpo_data/grpo_data.json",
    )

    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )


    trainer = GRPOTrainer(
        model,
        accelerator=accelerator,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        num_warmup_updates=args.num_warmup_updates,
        save_per_updates=args.save_per_updates,
        keep_last_n_checkpoints=args.keep_last_n_checkpoints,
        checkpoint_path=checkpoint_path,
        batch_size_per_gpu=args.batch_size_per_gpu,
        batch_size_type=args.batch_size_type,
        max_samples=args.max_samples,
        grad_accumulation_steps=32,  # grpo 特有的 gradient_accumulation_steps
        max_grad_norm=args.max_grad_norm,
        logger=args.logger,
        wandb_project=args.dataset_name,
        wandb_run_name=args.exp_name + "_" + args.save_name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        wandb_resume_id=wandb_resume_id,
        log_samples=args.log_samples,
        last_per_updates=args.last_per_updates,
        bnb_optimizer=args.bnb_optimizer,

        # grpo：beta 与 clip_range 影响 reward/音质权衡
        same_latent=False,
        beta=args.beta,
        clip_range=args.clip_range,
    )

    trainer.train(
        train_dataset,
        resumable_with_seed=666,  # seed for shuffling dataset
    )



if __name__ == "__main__":
    main()
