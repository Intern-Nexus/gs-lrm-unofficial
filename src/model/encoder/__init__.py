from typing import Optional

from .encoder import Encoder
from .encoder_lrm import EncoderLRM, EncoderLRMCfg


ENCODERS = {
    "lrm": (EncoderLRM),
}

EncoderCfg = EncoderLRMCfg


def get_encoder(cfg: EncoderCfg) -> Encoder:

    encoder = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    return encoder