import json
import re

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn
from torch.utils.data import Dataset, Sampler

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import convert_char_to_pinyin, default

class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size  # Batch size per replica
        self.k = k                    # Number of repetitions per sample
        self.num_replicas = num_replicas  # Total number of replicas
        self.rank = rank              # Current replica rank
        self.seed = seed              # Random seed for synchronization
        
        # Compute the number of unique samples needed per iteration
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # Number of unique samples
        self.epoch = 0

    def __iter__(self):
        while True:
            # Generate a deterministic random sequence to ensure all replicas are synchronized
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            # Randomly select m unique samples
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            
            # Repeat each sample k times to generate n*b total samples
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # Shuffle to ensure uniform distribution
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            
            # Split samples to each replica
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            
            # Return current replica's sample indices
            yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch):
        self.epoch = epoch  # Used to synchronize random state across epochs


class GRPODataset(Dataset):
    def __init__(
        self,
        data_path: str,
        target_sample_rate=24_000,
        hop_length=256,
        n_mel_channels=100,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        mel_spec_module: nn.Module | None = None,
    ):
        super().__init__()
        self.data = []
        if isinstance(data_path, list):
            for path in data_path:
                with open(path, "r") as f:
                    self.data.extend(json.load(f))
        else:
            with open(data_path, "r") as f:
                self.data = json.load(f)

        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type

        self.mel_spectrogram = default(
            mel_spec_module,
            MelSpec(
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mel_channels=n_mel_channels,
                target_sample_rate=target_sample_rate,
                mel_spec_type=mel_spec_type,
            ),
        )

        self.prompt_mel = []
        self.prompt_duration = []
        self.prompt_text = []
        for prompt, text in self.data["prompts"]:
            try:
                prompt_audio, prompt_sr = torchaudio.load(prompt)
                if prompt_audio.shape[0] > 1:
                    prompt_audio = torch.mean(prompt_audio, dim=0, keepdim=True)
                if prompt_sr != self.target_sample_rate:
                    resampler = torchaudio.transforms.Resample(prompt_sr, self.target_sample_rate)
                    prompt_audio = resampler(prompt_audio)
                prompt_duration = prompt_audio.shape[-1] / self.target_sample_rate
                prompt_mel = self.mel_spectrogram(prompt_audio)
                prompt_mel = prompt_mel.squeeze(0)  # (1, d, n) -> (d, n)
            except Exception as e:
                continue
            self.prompt_mel.append(prompt_mel)
            self.prompt_duration.append(prompt_duration)
            self.prompt_text.append(text)

        self.prompt_len = len(self.prompt_mel)
    
    def __len__(self):
        return len(self.data["texts"])
    
    def __getitem__(self, index):
        raw_text = self.data["texts"][index]
        text_length = len(raw_text.replace("<strong>", "").replace("</strong>", "").encode("utf-8"))
        text_list, emphasis_ids_list = convert_char_to_pinyin([raw_text], return_emphasis_ids=True)
        text = text_list[0]
        emphasis_ids = emphasis_ids_list[0]
        prompt_mel = self.prompt_mel[index % self.prompt_len]
        prompt_text = self.prompt_text[index % self.prompt_len]
        ref_text_length = len(prompt_text.replace("<strong>", "").replace("</strong>", "").encode("utf-8"))
        prompt_text_list, ref_emphasis_ids = convert_char_to_pinyin([prompt_text], return_emphasis_ids=True)
        ref_emphasis_ids = ref_emphasis_ids[0]
        prompt_text = prompt_text_list[0]

        emphasis_word = " ".join(re.findall(r"<strong>(.*?)</strong>", raw_text))

        result = {
            "ref_mel": prompt_mel,
            "ref_text": prompt_text,
            "ref_text_length": ref_text_length,
            "ref_emphasis_ids": ref_emphasis_ids,
            "raw_text": raw_text,
            "text": text,
            "emphasis_word": emphasis_word,
            "text_length": text_length,
            "emphasis_ids": emphasis_ids
        }
        return result


def grpo_collate_fn(batch):
    """Collate function for GRPO dataset.

    Handles batching of ref mel spectrograms, ref_text, text, and emphasis_ids.
    GRPODataset returns: {"ref_mel", "ref_duration", "ref_text", "text", "emphasis_ids"}
    """
    # Extract ref mel specs (dataset 返回 ref_mel)
    ref_mel_specs = []
    for item in batch:
        spec = item["ref_mel"]
        if spec.dim() == 3:
            spec = spec.squeeze(0)  # (1, d, n) -> (d, n)
        ref_mel_specs.append(spec)

    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in ref_mel_specs])
    max_mel_length = mel_lengths.amax()

    padded_mel_specs = []
    for spec in ref_mel_specs:
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_mel_specs.append(padded_spec)
    mel_specs = torch.stack(padded_mel_specs)  # (b, d, n)

    text = [item["text"] for item in batch]
    raw_text = [item["raw_text"] for item in batch]
    emphasis_word = [item["emphasis_word"] for item in batch]
    text_lengths = torch.LongTensor([item["text_length"] for item in batch])
    ref_text = [item["ref_text"] for item in batch]
    ref_text_lengths = torch.LongTensor([item["ref_text_length"] for item in batch])

    result = dict(
        ref_mel=mel_specs,
        ref_mel_lengths=mel_lengths,
        ref_text=ref_text,
        ref_text_lengths=ref_text_lengths,
        raw_text=raw_text,
        emphasis_word=emphasis_word,
        text=text,
        text_lengths=text_lengths,
    )

    if "emphasis_ids" in batch[0]:
        result["emphasis_ids"] = [item["emphasis_ids"] for item in batch]
    if "ref_emphasis_ids" in batch[0]:
        result["ref_emphasis_ids"] = [item["ref_emphasis_ids"] for item in batch]

    return result

