# load packages
import time
import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
from nltk.tokenize import word_tokenize

from models import *
from utils import *
from text_utils import TextCleaner
import phonemizer
from Utils.PLBERT.util import load_plbert
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

textclenaer = TextCleaner()

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300
)
MEAN, STD = -4, 4


def load_modules(config, model_path, device):
    # global_phonemizer = phonemizer.backend.EspeakBackend(
    #     language="en-us", preserve_punctuation=True, with_stress=True
    # ) # espeak is not yet installed on my machine

    config = yaml.safe_load(open(config, "r"))
    # load pretrained ASR model
    ASR_config = config.get("ASR_config", False)
    ASR_path = config.get("ASR_path", False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    # load pretrained F0 model
    F0_path = config.get("F0_path", False)
    pitch_extractor = load_F0_models(F0_path)

    # load BERT model
    BERT_path = config.get("PLBERT_dir", False)
    plbert = load_plbert(BERT_path)

    model_params = recursive_munch(config["model_params"])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].eval() for key in model]
    _ = [model[key].to() for key in model]

    params_whole = torch.load(model_path, map_location="cpu")
    params = params_whole["net"]

    for key in model:
        if key in params:
            print("%s loaded" % key)
            try:
                model[key].load_state_dict(params[key])
            except:
                from collections import OrderedDict

                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                # load params
                model[key].load_state_dict(new_state_dict, strict=False)
    _ = [model[key].eval() for key in model]

    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(
            sigma_min=0.0001, sigma_max=3.0, rho=9.0
        ),  # empirical parameters
        clamp=False,
    )

    return (
        model,
        sampler,
        model_params,  # for matching decoder type
    )


def load_pipeline():
    pass


def length_to_mask(lengths):
    mask = (
        torch.arange(lengths.max())
        .unsqueeze(0)
        .expand(lengths.shape[0], -1)
        .type_as(lengths)
    )
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - MEAN) / STD
    return mel_tensor


def compute_style(path, device, model):
    wave, sr = librosa.load(path, sr=24000)
    audio, index = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)
    mel_tensor = preprocess(audio).to(device)

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

    return torch.cat([ref_s, ref_p], dim=1), ref_s, ref_p
