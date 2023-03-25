import math
import wave
import struct
import os
import urllib.request
import tarfile
from audiolm_pytorch import SoundStream, SoundStreamTrainer, HubertWithKmeans, SemanticTransformer, SemanticTransformerTrainer, HubertWithKmeans, CoarseTransformer, CoarseTransformerWrapper, CoarseTransformerTrainer, FineTransformer, FineTransformerWrapper, FineTransformerTrainer, AudioLM
from torch import nn
import torch
import torchaudio

# define all dataset paths, checkpoints, etc
dataset_folder = "placeholder_dataset"
soundstream_ckpt = "results/soundstream.8.pt" # this can change depending on number of steps
hubert_ckpt = 'hubert/hubert_base_ls960.pt'
hubert_quantizer = f'hubert/hubert_base_ls960_L9_km500.bin' # listed in row "HuBERT Base (~95M params)", column Quantizer
# Placeholder data generation

soundstream = SoundStream(
    codebook_size = 1024,
    rq_num_quantizers = 8,
)

soundstream.load(f"./{soundstream_ckpt}")

fine_transformer = FineTransformer(
    num_coarse_quantizers = 3,
    num_fine_quantizers = 5,
    codebook_size = 1024,
    dim = 512,
    depth = 6
)

trainer = FineTransformerTrainer(
    transformer = fine_transformer,
    codec = soundstream,
    folder = dataset_folder,
    batch_size = 1,
    data_max_length = 320 * 32,
    num_train_steps = 9
)
# NOTE: I changed num_train_steps to 9 (aka 8 + 1) from 10000 to make things go faster for demo purposes
# adjusting save_*_every variables for the same reason

trainer.train()