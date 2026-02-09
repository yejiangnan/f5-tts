from __future__ import annotations

import copy
import gc
import math
import os

import torch
import torchaudio
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm

from f5_tts.model import CFM
from f5_tts.model.dataset import DynamicBatchSampler, DPODynamicBatchSampler, collate_fn, dpo_collate_fn
from f5_tts.model.utils import default, exists


# trainer


class Trainer:
    def __init__(
        self,
        model: CFM,
        epochs,
        learning_rate,
        num_warmup_updates=20000,
        save_per_updates=1000,
        keep_last_n_checkpoints: int = -1,  # -1 to keep all, 0 to not save intermediate, > 0 to keep last N checkpoints
        checkpoint_path=None,
        batch_size_per_gpu=32,
        batch_size_type: str = "sample",
        max_samples=32,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        noise_scheduler: str | None = None,
        duration_predictor: torch.nn.Module | None = None,
        logger: str | None = "wandb",  # "wandb" | "tensorboard" | None
        wandb_project="test_f5-tts",
        wandb_run_name="test_run",
        wandb_resume_id: str = None,
        log_samples: bool = False,
        last_per_updates=None,
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        bnb_optimizer: bool = False,
        mel_spec_type: str = "vocos",  # "vocos" | "bigvgan"
        is_local_vocoder: bool = False,  # use local path vocoder
        local_vocoder_path: str = "",  # local vocoder path
        model_cfg_dict: dict = dict(),  # training config
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        if logger == "wandb" and not wandb.api.api_key:
            logger = None
        self.log_samples = log_samples

        self.accelerator = Accelerator(
            log_with=logger if logger == "wandb" else None,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
            **accelerate_kwargs,
        )

        self.logger = logger
        if self.logger == "wandb":
            if exists(wandb_resume_id):
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name, "id": wandb_resume_id}}
            else:
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name}}

            if not model_cfg_dict:
                model_cfg_dict = {
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "num_warmup_updates": num_warmup_updates,
                    "batch_size_per_gpu": batch_size_per_gpu,
                    "batch_size_type": batch_size_type,
                    "max_samples": max_samples,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "noise_scheduler": noise_scheduler,
                }
            model_cfg_dict["gpus"] = self.accelerator.num_processes
            self.accelerator.init_trackers(
                project_name=wandb_project,
                init_kwargs=init_kwargs,
                config=model_cfg_dict,
            )

        elif self.logger == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=f"runs/{wandb_run_name}")

        self.model = model

        if self.is_main:
            self.ema_model = EMA(model, include_online_model=False, **ema_kwargs)
            self.ema_model.to(self.accelerator.device)

            print(f"Using logger: {logger}")
            if grad_accumulation_steps > 1:
                print(
                    "Gradient accumulation checkpointing with per_updates now, old logic per_steps used with before f992c4e"
                )

        self.epochs = epochs
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.last_per_updates = default(last_per_updates, save_per_updates)
        self.checkpoint_path = default(checkpoint_path, "ckpts/test_f5-tts")

        self.batch_size_per_gpu = batch_size_per_gpu
        self.batch_size_type = batch_size_type
        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # mel vocoder config
        self.vocoder_name = mel_spec_type
        self.is_local_vocoder = is_local_vocoder
        self.local_vocoder_path = local_vocoder_path

        self.noise_scheduler = noise_scheduler

        self.duration_predictor = duration_predictor

        if bnb_optimizer:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save_checkpoint(self, update, last=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                ema_model_state_dict=self.ema_model.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                update=update,
            )
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if last:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_last.pt")
                print(f"Saved last checkpoint at update {update}")
            else:
                if self.keep_last_n_checkpoints == 0:
                    return
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_{update}.pt")
                if self.keep_last_n_checkpoints > 0:
                    # Updated logic to exclude pretrained model from rotation
                    checkpoints = [
                        f
                        for f in os.listdir(self.checkpoint_path)
                        if f.startswith("model_")
                        and not f.startswith("pretrained_")  # Exclude pretrained models
                        and f.endswith(".pt")
                        and f != "model_last.pt"
                    ]
                    checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
                    while len(checkpoints) > self.keep_last_n_checkpoints:
                        oldest_checkpoint = checkpoints.pop(0)
                        os.remove(os.path.join(self.checkpoint_path, oldest_checkpoint))
                        print(f"Removed old checkpoint: {oldest_checkpoint}")

    def load_checkpoint(self):
        if (
            not exists(self.checkpoint_path)
            or not os.path.exists(self.checkpoint_path)
            or not any(filename.endswith((".pt", ".safetensors")) for filename in os.listdir(self.checkpoint_path))
        ):
            return 0

        self.accelerator.wait_for_everyone()
        if "model_last.pt" in os.listdir(self.checkpoint_path):
            latest_checkpoint = "model_last.pt"
        else:
            # Updated to consider pretrained models for loading but prioritize training checkpoints
            all_checkpoints = [
                f
                for f in os.listdir(self.checkpoint_path)
                if (f.startswith("model_") or f.startswith("pretrained_")) and f.endswith((".pt", ".safetensors"))
            ]

            # First try to find regular training checkpoints
            training_checkpoints = [f for f in all_checkpoints if f.startswith("model_") and f != "model_last.pt"]
            if training_checkpoints:
                latest_checkpoint = sorted(
                    training_checkpoints,
                    key=lambda x: int("".join(filter(str.isdigit, x))),
                )[-1]
            else:
                # If no training checkpoints, use pretrained model
                latest_checkpoint = next(f for f in all_checkpoints if f.startswith("pretrained_"))

        if latest_checkpoint.endswith(".safetensors"):  # always a pretrained checkpoint
            from safetensors.torch import load_file

            checkpoint = load_file(f"{self.checkpoint_path}/{latest_checkpoint}", device="cpu")
            checkpoint = {"ema_model_state_dict": checkpoint}
        elif latest_checkpoint.endswith(".pt"):
            # checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", map_location=self.accelerator.device)  # rather use accelerator.load_state ಥ_ಥ
            checkpoint = torch.load(
                f"{self.checkpoint_path}/{latest_checkpoint}", weights_only=True, map_location="cpu"
            )

        # patch for backward compatibility, 305e3ea
        for key in ["ema_model.mel_spec.mel_stft.mel_scale.fb", "ema_model.mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["ema_model_state_dict"]:
                del checkpoint["ema_model_state_dict"][key]

        if self.is_main:
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"], strict=False)

        if "update" in checkpoint or "step" in checkpoint:
            # patch for backward compatibility, with before f992c4e
            if "step" in checkpoint:
                checkpoint["update"] = checkpoint["step"] // self.grad_accumulation_steps
                if self.grad_accumulation_steps > 1 and self.is_main:
                    print(
                        "F5-TTS WARNING: Loading checkpoint saved with per_steps logic (before f992c4e), will convert to per_updates according to grad_accumulation_steps setting, may have unexpected behaviour."
                    )
            # patch for backward compatibility, 305e3ea
            for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
                if key in checkpoint["model_state_dict"]:
                    del checkpoint["model_state_dict"][key]

            # Handle size mismatch for text_embed.weight
            model_state_dict = checkpoint["model_state_dict"].copy()
            current_model = self.accelerator.unwrap_model(self.model)
            current_state_dict = current_model.state_dict()
            
            for key in list(model_state_dict.keys()):
                if key in current_state_dict:
                    if model_state_dict[key].shape != current_state_dict[key].shape:
                        if "text_embed.weight" in key and len(model_state_dict[key].shape) == 2:
                            # Handle vocab size mismatch: only load matching part
                            if model_state_dict[key].shape[0] > current_state_dict[key].shape[0]:
                                model_state_dict[key] = model_state_dict[key][:current_state_dict[key].shape[0]]
                            else:
                                # Skip if checkpoint size < model size
                                del model_state_dict[key]
                        else:
                            # Skip other size mismatches
                            del model_state_dict[key]
            
            current_model.load_state_dict(model_state_dict, strict=False)
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            update = checkpoint["update"]
        else:
            checkpoint["model_state_dict"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "update", "step"]
            }
            # Handle size mismatch for text_embed.weight
            model_state_dict = checkpoint["model_state_dict"].copy()
            current_model = self.accelerator.unwrap_model(self.model)
            current_state_dict = current_model.state_dict()
            
            for key in list(model_state_dict.keys()):
                if key in current_state_dict:
                    if model_state_dict[key].shape != current_state_dict[key].shape:
                        if "text_embed.weight" in key and len(model_state_dict[key].shape) == 2:
                            # Handle vocab size mismatch: only load matching part
                            if model_state_dict[key].shape[0] > current_state_dict[key].shape[0]:
                                model_state_dict[key] = model_state_dict[key][:current_state_dict[key].shape[0]]
                            else:
                                # Skip if checkpoint size < model size
                                del model_state_dict[key]
                        else:
                            # Skip other size mismatches
                            del model_state_dict[key]
            
            current_model.load_state_dict(model_state_dict, strict=False)
            update = 0

        del checkpoint
        gc.collect()
        return update
    


    def train(self, train_dataset: Dataset, num_workers=16, resumable_with_seed: int = None):
        if self.log_samples:
            from f5_tts.infer.utils_infer import cfg_strength, load_vocoder, nfe_step, sway_sampling_coef

            vocoder = load_vocoder(
                vocoder_name=self.vocoder_name, is_local=self.is_local_vocoder, local_path=self.local_vocoder_path
            )
            target_sample_rate = self.accelerator.unwrap_model(self.model).mel_spec.target_sample_rate
            log_samples_path = f"{self.checkpoint_path}/samples"
            os.makedirs(log_samples_path, exist_ok=True)

        if exists(resumable_with_seed):
            generator = torch.Generator()
            generator.manual_seed(resumable_with_seed)
        else:
            generator = None

        if self.batch_size_type == "sample":
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_size=self.batch_size_per_gpu,
                shuffle=True,
                generator=generator,
            )
        elif self.batch_size_type == "frame":
            self.accelerator.even_batches = False
            sampler = SequentialSampler(train_dataset)
            batch_sampler = DynamicBatchSampler(
                sampler,
                self.batch_size_per_gpu,
                max_samples=self.max_samples,
                random_seed=resumable_with_seed,  # This enables reproducible shuffling
                drop_residual=False,
            )
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_sampler=batch_sampler,
            )
        else:
            raise ValueError(f"batch_size_type must be either 'sample' or 'frame', but received {self.batch_size_type}")

        #  accelerator.prepare() dispatches batches to devices;
        #  which means the length of dataloader calculated before, should consider the number of devices
        warmup_updates = (
            self.num_warmup_updates * self.accelerator.num_processes
        )  # consider a fixed warmup steps while using accelerate multi-gpu ddp
        # otherwise by default with split_batches=False, warmup steps change with num_processes
        total_updates = math.ceil(len(train_dataloader) / self.grad_accumulation_steps) * self.epochs
        decay_updates = total_updates - warmup_updates
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates]
        )
        train_dataloader, self.scheduler = self.accelerator.prepare(
            train_dataloader, self.scheduler
        )  # actual multi_gpu updates = single_gpu updates / gpu nums
        start_update = self.load_checkpoint()
        global_update = start_update

        if exists(resumable_with_seed):
            orig_epoch_step = len(train_dataloader)
            start_step = start_update * self.grad_accumulation_steps
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0

        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            if exists(resumable_with_seed) and epoch == skipped_epoch:
                progress_bar_initial = math.ceil(skipped_batch / self.grad_accumulation_steps)
                current_dataloader = skipped_dataloader
            else:
                progress_bar_initial = 0
                current_dataloader = train_dataloader

            # Set epoch for the batch sampler if it exists
            if hasattr(train_dataloader, "batch_sampler") and hasattr(train_dataloader.batch_sampler, "set_epoch"):
                train_dataloader.batch_sampler.set_epoch(epoch)

            progress_bar = tqdm(
                range(math.ceil(len(train_dataloader) / self.grad_accumulation_steps)),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                unit="update",
                disable=not self.accelerator.is_local_main_process,
                initial=progress_bar_initial,
            )

            for batch in current_dataloader:
                with self.accelerator.accumulate(self.model):
                    text_inputs = batch["text"]
                    mel_spec = batch["mel"].permute(0, 2, 1)
                    mel_lengths = batch["mel_lengths"]
                    emphasis_ids = batch["emphasis_ids"]

                    # TODO. add duration predictor training
                    if self.duration_predictor is not None and self.accelerator.is_local_main_process:
                        dur_loss = self.duration_predictor(mel_spec, lens=batch.get("durations"))
                        self.accelerator.log({"duration loss": dur_loss.item()}, step=global_update)
                    
                    
                    loss, cond, pred = self.model(
                        mel_spec, text=text_inputs, lens=mel_lengths, noise_scheduler=self.noise_scheduler, emphasis_ids=emphasis_ids
                    )
                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    if self.is_main:
                        self.ema_model.update()

                    global_update += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(update=str(global_update), loss=loss.item())

                if self.accelerator.is_local_main_process:
                    self.accelerator.log(
                        {"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}, step=global_update
                    )
                    if self.logger == "tensorboard":
                        self.writer.add_scalar("loss", loss.item(), global_update)
                        self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_update)

                if global_update % self.last_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update, last=True)

                if global_update % self.save_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update)

                    if self.log_samples and self.accelerator.is_local_main_process:
                        ref_audio_len = mel_lengths[0]
                        infer_text = [
                            text_inputs[0] + ([" "] if isinstance(text_inputs[0], list) else " ") + text_inputs[0]
                        ]
                        infer_emphasis_ids = [emphasis_ids[0] + [0] + emphasis_ids[0]]
                        # print("infer_text and infer_emphasis_ids in training:")
                        print(f"infer_text: {infer_text}")
                        print(f"infer_emphasis_ids: {infer_emphasis_ids}")
                        # # 在多卡训练中，使用异常来停止训练循环
                        # # 这样可以让所有进程同步退出，而不是强制终止单个进程
                        # raise KeyboardInterrupt("Debug output completed, stopping training")

                        with torch.inference_mode():
                            generated, _ = self.accelerator.unwrap_model(self.model).sample(
                                cond=mel_spec[0][:ref_audio_len].unsqueeze(0),
                                text=infer_text,
                                duration=ref_audio_len * 2,
                                steps=nfe_step,
                                cfg_strength=cfg_strength,
                                sway_sampling_coef=sway_sampling_coef,
                                emphasis_ids=infer_emphasis_ids,
                            )
                            generated = generated.to(torch.float32)
                            gen_mel_spec = generated[:, ref_audio_len:, :].permute(0, 2, 1).to(self.accelerator.device)
                            ref_mel_spec = batch["mel"][0].unsqueeze(0)
                            if self.vocoder_name == "vocos":
                                gen_audio = vocoder.decode(gen_mel_spec).cpu()
                                ref_audio = vocoder.decode(ref_mel_spec).cpu()
                            elif self.vocoder_name == "bigvgan":
                                gen_audio = vocoder(gen_mel_spec).squeeze(0).cpu()
                                ref_audio = vocoder(ref_mel_spec).squeeze(0).cpu()

                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_gen.wav", gen_audio, target_sample_rate
                        )
                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_ref.wav", ref_audio, target_sample_rate
                        )
                        self.model.train()

        self.save_checkpoint(global_update, last=True)

        self.accelerator.end_training()



class DPOTrainer(Trainer):
    def __init__(
        self,
        model: CFM,
        dpo_beta: float = 0.1,  # DPO温度参数
        dpo_loss_weight: float = 0.5,  # DPO损失权重
        sft_loss_weight: float = 1.0,  # SFT损失权重（可选，混合训练）
        **kwargs,  # 其他参数直接传给父类Trainer
    ):
        super().__init__(model, **kwargs)  # 所有其他参数传给父类
        
        # DPO特有参数
        self.dpo_beta = dpo_beta
        self.dpo_loss_weight = dpo_loss_weight
        self.sft_loss_weight = sft_loss_weight
        
 
        self.ref_model = copy.deepcopy(self.accelerator.unwrap_model(self.model))
        self.ref_model.to(self.accelerator.device)
        
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()
    

  
    
    def train(self, train_dataset: Dataset, num_workers=16, resumable_with_seed: int = None):
        if self.log_samples:
            from f5_tts.infer.utils_infer import cfg_strength, load_vocoder, nfe_step, sway_sampling_coef

            vocoder = load_vocoder(
                vocoder_name=self.vocoder_name, is_local=self.is_local_vocoder, local_path=self.local_vocoder_path
            )
            target_sample_rate = self.accelerator.unwrap_model(self.model).mel_spec.target_sample_rate
            log_samples_path = f"{self.checkpoint_path}/samples"
            os.makedirs(log_samples_path, exist_ok=True)

        if exists(resumable_with_seed):
            generator = torch.Generator()
            generator.manual_seed(resumable_with_seed)
        else:
            generator = None

        if self.batch_size_type == "sample":
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=dpo_collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_size=self.batch_size_per_gpu,
                shuffle=True,
                generator=generator,
            )
        elif self.batch_size_type == "frame":
            self.accelerator.even_batches = False
            sampler = SequentialSampler(train_dataset)
            # Use DPODynamicBatchSampler for DPO training
            batch_sampler = DPODynamicBatchSampler(
                sampler,
                self.batch_size_per_gpu,
                max_samples=self.max_samples,
                random_seed=resumable_with_seed,  # This enables reproducible shuffling
                drop_residual=False,
            )
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=dpo_collate_fn,  # Use DPO collate function
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_sampler=batch_sampler,
            )
       
        #  accelerator.prepare() dispatches batches to devices;
        #  which means the length of dataloader calculated before, should consider the number of devices
        warmup_updates = (
            self.num_warmup_updates * self.accelerator.num_processes
        )  # consider a fixed warmup steps while using accelerate multi-gpu ddp
        # otherwise by default with split_batches=False, warmup steps change with num_processes
        total_updates = math.ceil(len(train_dataloader) / self.grad_accumulation_steps) * self.epochs
        decay_updates = total_updates - warmup_updates
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates]
        )
        train_dataloader, self.scheduler = self.accelerator.prepare(
            train_dataloader, self.scheduler
        )  # actual multi_gpu updates = single_gpu updates / gpu nums
        start_update = self.load_checkpoint()
        global_update = start_update
    
        ### load ref_model checkpoint ###
        ckpt_path = "/data/F5-TTS/ckpts/sft_data_emphasis_ids_enhanced_transformer/model_21500.pt"
        print(f"Loading checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, weights_only=True, map_location="cpu")
        if "ema_model_state_dict" in checkpoint:
            model_state_dict = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "update", "step"]
            }
        else:
            model_state_dict = checkpoint["model_state_dict"]
        ref_model = self.accelerator.unwrap_model(self.ref_model)
        ref_model.load_state_dict(model_state_dict, strict=True)

        if exists(resumable_with_seed):
            orig_epoch_step = len(train_dataloader)
            start_step = start_update * self.grad_accumulation_steps
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0
        
        ### train loop ### 
        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            self.ref_model.eval()
            if exists(resumable_with_seed) and epoch == skipped_epoch:
                progress_bar_initial = math.ceil(skipped_batch / self.grad_accumulation_steps)
                current_dataloader = skipped_dataloader
            else:
                progress_bar_initial = 0
                current_dataloader = train_dataloader

            # Set epoch for the batch sampler if it exists
            if hasattr(train_dataloader, "batch_sampler") and hasattr(train_dataloader.batch_sampler, "set_epoch"):
                train_dataloader.batch_sampler.set_epoch(epoch)

            progress_bar = tqdm(
                range(math.ceil(len(train_dataloader) / self.grad_accumulation_steps)),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                unit="update",
                disable=not self.accelerator.is_local_main_process,
                initial=progress_bar_initial,
            )


            for batch in current_dataloader:
                with self.accelerator.accumulate(self.model):
                    text_inputs = batch["text"]
                    chosen_mel_spec = batch["chosen_mel_spec"].permute(0, 2, 1)
                    reject_mel_spec = batch["reject_mel_spec"].permute(0, 2, 1)
                    chosen_mel_lengths = batch["chosen_mel_lengths"]
                    reject_mel_lengths = batch["reject_mel_lengths"]
                    emphasis_ids = batch["emphasis_ids"]


                    # TODO. add duration predictor training
                    if self.duration_predictor is not None and self.accelerator.is_local_main_process:
                        chosen_dur_loss = self.duration_predictor(chosen_mel_spec, lens=chosen_mel_lengths)
                        reject_dur_loss = self.duration_predictor(reject_mel_spec, lens=reject_mel_lengths)
                        dur_loss = (chosen_dur_loss + reject_dur_loss) / 2
                        self.accelerator.log({"duration loss": dur_loss.item()}, step=global_update)
                    
                    batch_size = chosen_mel_spec.shape[0]
                    b_chosen, t_chosen, _ = chosen_mel_spec.shape
                    b_reject, t_reject, _ = reject_mel_spec.shape

                    time = torch.rand((batch_size,), dtype=chosen_mel_spec.dtype, device=chosen_mel_spec.device)
                    noise_chosen = torch.randn_like(chosen_mel_spec)
                    noise_reject = torch.randn_like(reject_mel_spec)

                    mask_chosen = torch.ones((b_chosen, t_chosen), dtype=torch.bool, device=chosen_mel_spec.device)
                    mask_reject = torch.ones((b_reject, t_reject), dtype=torch.bool, device=reject_mel_spec.device)


                    chosen_loss, chosen_cond, chosen_pred = self.model(
                        chosen_mel_spec, 
                        text=text_inputs, 
                        time=time, 
                        lens=chosen_mel_lengths, 
                        noise_scheduler=self.noise_scheduler, 
                        emphasis_ids=emphasis_ids,
                        is_dpo=True,
                        noise=noise_chosen,
                        span_mask=mask_chosen,
                    )
                    reject_loss, reject_cond, reject_pred = self.model(
                        reject_mel_spec, 
                        text=text_inputs, 
                        time=time, 
                        lens=reject_mel_lengths, 
                        noise_scheduler=self.noise_scheduler, 
                        emphasis_ids=emphasis_ids,
                        is_dpo=True,
                        noise=noise_reject,
                        span_mask=mask_reject,
                    )
                    
                    # 检查损失是否异常
                    if torch.isnan(chosen_loss).any() or torch.isinf(chosen_loss).any() or \
                        torch.isnan(reject_loss).any() or torch.isinf(reject_loss).any():

                        print(f"Warning: Loss contains NaN or Inf at step {global_update}. chosen_loss: {chosen_loss}, reject_loss: {reject_loss}")
                        self.optimizer.zero_grad()
                        continue
                    
                    with torch.no_grad():
                        ref_chosen_loss, _, _ = ref_model(
                            chosen_mel_spec, 
                            text=text_inputs, 
                            time=time, 
                            lens=chosen_mel_lengths, 
                            noise_scheduler=self.noise_scheduler, 
                            emphasis_ids=emphasis_ids,
                            is_dpo=True,
                            noise=noise_chosen,
                            span_mask=mask_chosen,
                        )
                        ref_reject_loss, _, _ = ref_model(
                            reject_mel_spec, 
                            text=text_inputs, 
                            time=time, 
                            lens=reject_mel_lengths, 
                            noise_scheduler=self.noise_scheduler, 
                            emphasis_ids=emphasis_ids,
                            is_dpo=True,
                            noise=noise_reject,
                            span_mask=mask_reject,
                        )

                    # 计算DPO损失
                    # 将损失转换为奖励：reward = -loss（更小的损失 = 更大的奖励）
                    # 对于 Flow Matching，reward = -loss 可以看作是 log-probability 的近似
                    reward_chosen = -chosen_loss
                    reward_rejected = -reject_loss
                    ref_reward_chosen = -ref_chosen_loss
                    ref_reward_rejected = -ref_reject_loss
                    
                    # DPO损失的标准公式：
                    # loss = -log(σ(β * ((log π_θ(y_w|x) - log π_θ(y_l|x)) - (log π_ref(y_w|x) - log π_ref(y_l|x)))))
                    # 其中：log π_θ(y|x) ≈ -loss = reward
                    # 所以：loss = -log(σ(β * ((reward_w - reward_l) - (ref_reward_w - ref_reward_l))))
                    
                    # 计算 log-ratio（当前模型的 chosen 和 rejected 的奖励差）
                    policy_log_ratio = reward_chosen - reward_rejected
                    # 计算参考模型的 log-ratio
                    ref_log_ratio = ref_reward_chosen - ref_reward_rejected
                    # 计算差异（这就是 DPO 的核心：让当前模型比参考模型更偏好 chosen）
                    logits_diff = policy_log_ratio - ref_log_ratio
                    
                    # 使用数值稳定的实现，应用温度参数 β
                    # 注意：logsigmoid 对每个样本计算，然后需要求平均
                    dpo_loss = -torch.nn.functional.logsigmoid(self.dpo_beta * logits_diff)
                    
                    # 检查 DPO 损失是否异常
                    # 注意：dpo_loss 是 tensor（可能有多个元素），需要使用 .any() 检查
                    if torch.isnan(dpo_loss).any() or torch.isinf(dpo_loss).any():
                        print(f"Warning: DPO loss contains NaN or Inf at step {global_update}. logits_diff: {logits_diff}, dpo_loss: {dpo_loss}")
                        self.optimizer.zero_grad()
                        continue
                    
                    loss = self.dpo_loss_weight * dpo_loss.mean() + self.sft_loss_weight * chosen_loss.mean()
                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    if self.is_main:
                        self.ema_model.update()

                    global_update += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(update=str(global_update), loss=loss.item())

                if self.accelerator.is_local_main_process:
                    # 记录更多调试信息用于 DPO 训练
                    # 注意：chosen_loss, reject_loss 等是每个样本的损失（形状 [batch_size]）
                    # 需要使用 .mean() 求平均后再用 .item() 转换为标量
                    log_dict = {
                        "loss": loss.item(),
                        "lr": self.scheduler.get_last_lr()[0],
                        "chosen_loss": chosen_loss.mean().item(),
                        "reject_loss": reject_loss.mean().item(),
                        "ref_chosen_loss": ref_chosen_loss.mean().item(),
                        "ref_reject_loss": ref_reject_loss.mean().item(),
                        "policy_log_ratio": policy_log_ratio.mean().item(),
                        "ref_log_ratio": ref_log_ratio.mean().item(),
                        "logits_diff": logits_diff.mean().item(),
                    }
                    self.accelerator.log(log_dict, step=global_update)
                    if self.logger == "tensorboard":
                        self.writer.add_scalar("loss", loss.item(), global_update)
                        self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_update)
                        self.writer.add_scalar("chosen_loss", chosen_loss.mean().item(), global_update)
                        self.writer.add_scalar("reject_loss", reject_loss.mean().item(), global_update)
                        self.writer.add_scalar("logits_diff", logits_diff.mean().item(), global_update)

                if global_update % self.last_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update, last=True)

                if global_update % self.save_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update)

                    if self.log_samples and self.accelerator.is_local_main_process:
                        # DPO trainer 中使用 chosen_mel_lengths 和 chosen_mel_spec
                        ref_audio_len = chosen_mel_lengths[0]
                        infer_text = [
                            text_inputs[0] + ([" "] if isinstance(text_inputs[0], list) else " ") + text_inputs[0]
                        ]
                        infer_emphasis_ids = [emphasis_ids[0] + [0] + emphasis_ids[0]]

                        with torch.inference_mode():
                            # chosen_mel_spec 的格式是 (b, n, d)，需要取第一个样本的前 ref_audio_len 个时间步
                            # 注意：chosen_mel_spec 已经在训练循环中被 permute 成 (b, n, d) 了
                            cond_mel = chosen_mel_spec[0][:ref_audio_len].unsqueeze(0)  # (1, ref_audio_len, d)
                            generated, _ = self.accelerator.unwrap_model(self.model).sample(
                                cond=cond_mel,
                                text=infer_text,
                                duration=ref_audio_len * 2,
                                steps=nfe_step,
                                cfg_strength=cfg_strength,
                                sway_sampling_coef=sway_sampling_coef,
                                emphasis_ids=infer_emphasis_ids,
                            )
                            generated = generated.to(torch.float32)


                            ref_generated, _ = ref_model.sample(
                                cond=cond_mel,
                                text=infer_text,
                                duration=ref_audio_len * 2,
                                steps=nfe_step,
                                cfg_strength=cfg_strength,
                                sway_sampling_coef=sway_sampling_coef,
                                emphasis_ids=infer_emphasis_ids,
                            )
                            
                            # 检查生成的 mel 是否有效
                            
                            # 取生成的部分（去掉条件部分）
                            gen_mel_spec = generated[:, ref_audio_len:, :].permute(0, 2, 1).to(self.accelerator.device)  # (1, d, gen_len)
                            ref_gen_mel_spec = ref_generated[:, ref_audio_len:, :].permute(0, 2, 1).to(self.accelerator.device)  # (1, d, gen_len)
                            
                            # 使用原始的 batch["chosen_mel_spec"]，格式是 (b, d, n)，直接取第一个样本
                            # 注意：不要使用 permute 后的 chosen_mel_spec，要用原始的 batch 数据
                            ref_mel_spec = batch["chosen_mel_spec"][0:1, :, :ref_audio_len]  # (1, d, ref_audio_len)
                            if self.vocoder_name == "vocos":
                                gen_audio = vocoder.decode(gen_mel_spec).cpu()
                                ref_audio = vocoder.decode(ref_mel_spec).cpu()
                                ref_gen_audio = vocoder.decode(ref_gen_mel_spec).cpu()
                            elif self.vocoder_name == "bigvgan":
                                gen_audio = vocoder(gen_mel_spec).squeeze(0).cpu()
                                ref_audio = vocoder(ref_mel_spec).squeeze(0).cpu()
                                ref_gen_audio = vocoder(ref_gen_mel_spec).squeeze(0).cpu()
                     

                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_gen.wav", gen_audio, target_sample_rate
                        )
                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_ref.wav", ref_audio, target_sample_rate
                        )
                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_ref_gen.wav", ref_gen_audio, target_sample_rate
                        )
                        self.model.train()

        self.save_checkpoint(global_update, last=True)

        self.accelerator.end_training()