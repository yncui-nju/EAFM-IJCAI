"""Evaluate a pretrained EAFM checkpoint on entity-alignment datasets."""

import argparse
import logging
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch

from experiment import Experiment


def prepare(args):
    """Normalize CLI values, create output paths, and configure logging."""

    def add_note(path, dic):
        for key, value in dic:
            path = path + "_" + key + "_" + str(value)
        return path

    # Keep the original naming convention for published run artifacts.
    args.log_path = args.log_path + datetime.now().strftime("%Y%m%d/")  # noqa: DTZ005
    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)
    log_note_dict = [
        ("", args.train_dataset_list),
        ("", args.test_dataset_list),
        ("dim", args.hidden_dim),
        ("layer", args.n_layer),
        ("rel_layer", args.n_relation_encoder_layer),
        ("", args.note),
    ]
    args.log_path = add_note(args.log_path, log_note_dict)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    if torch.cuda.is_available():
        args.device = torch.device("cuda", args.gpu)
    else:
        args.device = torch.device("cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    logger = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s: %(message)s")
    console_formatter = logging.Formatter("%(asctime)-8s: %(message)s")
    logging_file_name = args.log_path + ".txt"
    file_handler = logging.FileHandler(logging_file_name)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.formatter = console_formatter
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    args.logger = logger


parser = argparse.ArgumentParser(description="Parser for KGFM")

parser.add_argument("--checkpoint_path", type=str, default="../checkpoint/pretrain/")
parser.add_argument("--train_dataset_list", type=str, default=None)
parser.add_argument("--valid_dataset_list", type=str, default=None)
parser.add_argument(
    "--test_dataset_list", type=str, default="OpenEA#D_W_15K_V1 OpenEA#D_Y_15K_V1"
)
parser.add_argument("--fold_id", type=int, default=1, help="Fold ID (1-5)")

# model structure
parser.add_argument("--use_text_embeddings", type=str, default="False")
parser.add_argument("--text_encoder_path", type=str, default="BAAI/bge-large-en-v1.5")
parser.add_argument("--K", type=str, default="3")

parser.add_argument("--use_merge_relation_graph", type=str, default="True")
parser.add_argument("--use_anchor_conditioned_initialize", type=str, default="True")
parser.add_argument("--use_two_tower", type=str, default="True")
parser.add_argument("--use_interact", type=str, default="True")

# hyper_parameters
parser.add_argument("--hidden_dim", type=int, default=32)
parser.add_argument("--attn_dim", type=int, default=5)
parser.add_argument("--n_relation_encoder_layer", type=int, default=6)
parser.add_argument("--n_layer", type=int, default=6)
parser.add_argument("--test_batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--lamb", type=float, default=0.0001)
parser.add_argument("--decay_rate", type=float, default=0.995)
parser.add_argument("--act", type=str, default="idd")
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--shell", type=str, default="False")

# others
parser.add_argument("--data_base_path", type=str, default="../datasets")
parser.add_argument("--data_path", type=str, default="processed_data/")
parser.add_argument("--log_path", type=str, default="../log/pretrain/")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--note", type=str, default="")
parser.add_argument("--ea_rate", type=str, default="1.0")

args = parser.parse_args()

if not os.path.exists(args.log_path):
    os.makedirs(args.log_path)

if args.shell == "True":
    args.data_base_path = os.path.join("..", args.data_base_path)
    args.log_path = os.path.join("..", args.log_path)

# Normalize each optional dataset-list argument to ``list[str]``.
if args.train_dataset_list is None:
    pass
elif len(args.train_dataset_list.split()) > 1:
    args.train_dataset_list = args.train_dataset_list.split()
else:
    args.train_dataset_list = [args.train_dataset_list]
if args.valid_dataset_list is None:
    pass
elif len(args.valid_dataset_list.split()) > 1:
    args.valid_dataset_list = args.valid_dataset_list.split()
else:
    args.valid_dataset_list = [args.valid_dataset_list]
if args.test_dataset_list is None:
    pass
elif len(args.test_dataset_list.split()) > 1:
    args.test_dataset_list = args.test_dataset_list.split()
else:
    args.test_dataset_list = [args.test_dataset_list]

args.train_dirs = (
    [
        os.path.join(
            args.data_base_path,
            args.data_path,
            dataset.split("#")[0],
            dataset.split("#")[1],
        )
        for dataset in args.train_dataset_list
    ]
    if args.train_dataset_list is not None
    else None
)
args.valid_dirs = (
    [
        os.path.join(
            args.data_base_path,
            args.data_path,
            dataset.split("#")[0],
            dataset.split("#")[1],
        )
        for dataset in args.valid_dataset_list
    ]
    if args.valid_dataset_list is not None
    else None
)
args.test_dirs = (
    [
        os.path.join(
            args.data_base_path,
            args.data_path,
            dataset.split("#")[0],
            dataset.split("#")[1],
        )
        for dataset in args.test_dataset_list
    ]
    if args.test_dataset_list is not None
    else None
)


prepare(args)


def test(args, experiment):
    """Evaluate all configured test datasets."""
    experiment.load_test()
    _, out_str = experiment.evaluate(mode="test")
    args.logger.info(out_str)


if __name__ == "__main__":
    args.logger.info(args)
    experiment = Experiment(args)
    experiment.load_checkpoint(args.checkpoint_path, gpu=args.gpu)
    test(args, experiment)
