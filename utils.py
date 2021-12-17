import logging
import random
import numpy as np
import torch
from model.modeling_phobert import CrossEncoderPhoBERT
from transformers import (
    AutoTokenizer,
    RobertaConfig,
)

MODEL_CLASSES = {
    "phobert": (RobertaConfig, CrossEncoderPhoBERT, AutoTokenizer),
}

MODEL_PATH_MAP = {
    "phobert": "vinai/phobert-base",
}


def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path,
                                                             do_lower_case=args.do_lower_case)


def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
