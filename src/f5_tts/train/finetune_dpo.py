import argparse
import os
import shutil
from datetime import datetime
from importlib.resources import files

from cached_path import cached_path

from f5_tts.model import CFM, DiT, Trainer, UNetT, DPOTrainer
from f5_tts.model.dataset import load_dataset, DPODataset
from f5_tts.model.utils import get_tokenizer


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
    parser.add_argument("--pretrain", type=str, default=None, help="the path to the checkpoint")
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
    parser.add_argument(
        "--dpo_beta",
        type=float,
        default=0.1,
        help="DPO beta parameter (temperature). Higher values (0.2-0.5) keep model closer to reference, lower values (0.01-0.1) allow more deviation. Default: 0.1",
    )
    parser.add_argument(
        "--dpo_loss_weight",
        type=float,
        default=1.0,
        help="Weight for DPO loss. Default: 1.0",
    )
    parser.add_argument(
        "--sft_loss_weight",
        type=float,
        default=1.0,
        help="Weight for SFT loss. Default: 1.0",
    )

    return parser.parse_args()


# -------------------------- Training Settings -------------------------- #


def main():
    args = parse_args()
    

    checkpoint_path = str(files("f5_tts").joinpath(f"../../ckpts/{args.dataset_name}_{args.save_name}"))

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

    ckpt_path = args.pretrain
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)

    file_checkpoint = os.path.basename(ckpt_path)
    if not file_checkpoint.startswith("pretrained_"):  # Change: Add 'pretrained_' prefix to copied model
        file_checkpoint = "pretrained_" + file_checkpoint
    file_checkpoint = os.path.join(checkpoint_path, file_checkpoint)
    if not os.path.isfile(file_checkpoint):
        shutil.copy2(ckpt_path, file_checkpoint)
        print("copy checkpoint for finetune")
  
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

    train_dataset = DPODataset(
        data_path=["/data/F5-TTS/outputs/batch_transformer_21500_dpo_data/dpo_metadata.json",
                   "/data/F5-TTS/outputs/batch_stepf15_dpo_data/dpo_metadata.json",
                   "/data/F5-TTS/outputs/batch_stepf06_dpo_data/dpo_metadata.json",
                   "/data/F5-TTS/outputs/batch_stepm284_dpo_data/dpo_metadata.json"],
        target_sample_rate=target_sample_rate,
        hop_length=hop_length,
        n_mel_channels=n_mel_channels,
        n_fft=n_fft,
        win_length=win_length,
        mel_spec_type=mel_spec_type,
    )

    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )

    # 验证 emphasis_enhanced 参数设置
    print("\n" + "=" * 60)
    print("模型参数验证 (Model Parameter Verification)")
    print("=" * 60)
    
    if hasattr(model.transformer, 'text_embed'):
        text_embed = model.transformer.text_embed
        if hasattr(text_embed, 'emphasis_enhanced'):
            if text_embed.emphasis_enhanced:
                print("✅ Emphasis Enhanced Mode: ENABLED")
                if hasattr(text_embed, 'emphasis_mlp'):
                    print(f"   - Emphasis MLP: {type(text_embed.emphasis_mlp).__name__}")
                    # 计算 MLP 参数量
                    mlp_params = sum(p.numel() for p in text_embed.emphasis_mlp.parameters())
                    print(f"   - Emphasis MLP Parameters: {mlp_params:,}")
                if hasattr(text_embed, 'emphasis_gate'):
                    print(f"   - Emphasis Gate: {type(text_embed.emphasis_gate).__name__}")
                    gate_params = sum(p.numel() for p in text_embed.emphasis_gate.parameters())
                    print(f"   - Emphasis Gate Parameters: {gate_params:,}")
                if hasattr(text_embed, 'emphasis_scale'):
                    print(f"   - Emphasis Scale: {text_embed.emphasis_scale.item():.4f} (learnable)")
                if hasattr(text_embed, 'emphasis_dim'):
                    print(f"   - Emphasis Dimension: {text_embed.emphasis_dim if text_embed.emphasis_dim else 'default (text_dim)'}")
            else:
                print("⚠️  Emphasis Enhanced Mode: DISABLED (using simple version)")
                if hasattr(text_embed, 'emphasis_embed'):
                    print(f"   - Using simple emphasis_embed parameter")
        else:
            print("⚠️  Emphasis Enhanced attribute not found in TextEmbedding")
    else:
        print("⚠️  TextEmbedding not found in transformer")
    
    # 打印模型配置信息
    print(f"\n模型配置 (Model Config):")
    print(f"   - Model Class: {model_cls.__name__}")
    print(f"   - Experiment: {args.exp_name}")
    for key, value in model_cfg.items():
        print(f"   - {key}: {value}")
    
    
    print("=" * 60 + "\n")


    trainer = DPOTrainer(
        model,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        num_warmup_updates=args.num_warmup_updates,
        save_per_updates=args.save_per_updates,
        keep_last_n_checkpoints=args.keep_last_n_checkpoints,
        checkpoint_path=checkpoint_path,
        batch_size_per_gpu=args.batch_size_per_gpu,
        batch_size_type=args.batch_size_type,
        max_samples=args.max_samples,
        grad_accumulation_steps=args.grad_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        logger=args.logger,
        wandb_project=args.dataset_name,
        wandb_run_name=args.exp_name + "_" + args.save_name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        wandb_resume_id=wandb_resume_id,
        log_samples=args.log_samples,
        last_per_updates=args.last_per_updates,
        bnb_optimizer=args.bnb_optimizer,
        dpo_beta=args.dpo_beta,
        dpo_loss_weight=args.dpo_loss_weight,
        sft_loss_weight=args.sft_loss_weight,
    )

    trainer.train(
        train_dataset,
        resumable_with_seed=666,  # seed for shuffling dataset
    )

    # trainer.test(
    #     train_dataset,
    #     resumable_with_seed=666,  # seed for shuffling dataset
    # )


if __name__ == "__main__":
    main()
