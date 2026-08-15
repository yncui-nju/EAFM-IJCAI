"""Training, checkpointing, and evaluation orchestration for EAFM."""

import os
import random
import sys
import time

import numpy as np
import torch
from torch.nn import init
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from data_loader import EADataLoader
from encoder.EntityEncoder import EntityEncoder
from utils import cal_performance, cal_ranks


class Experiment:
    """Own the model, data loaders, optimizer, and experiment lifecycle."""

    def __init__(self, args):
        self.args = args

        if args.train_dirs is not None:
            self.train_loaders = [
                EADataLoader(train_dir, fold_id=getattr(args, "fold_id", 1), args=args)
                for train_dir in args.train_dirs
            ]
        else:
            self.train_loaders = None
        if args.valid_dirs is not None:
            self.valid_loaders = [
                EADataLoader(valid_dir, fold_id=getattr(args, "fold_id", 1), args=args)
                for valid_dir in args.valid_dirs
            ]
        else:
            self.valid_loaders = None

        self.model = EntityEncoder(self.args)

        for name, param in self.model.named_parameters():
            if "weight" in name and param.dim() > 1:
                init.xavier_uniform_(param.data)
            elif "bias" in name:
                init.constant_(param.data, 0)
        self.num_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.args.logger.info(f"Number of parameters: {self.num_params}")

        self.model.to(self.args.device)

        self.optimizer = Adam(
            self.model.parameters(), lr=args.lr, weight_decay=args.lamb
        )
        self.scheduler = ExponentialLR(self.optimizer, args.decay_rate)
        self.t_time = 0

    def load_test(self):
        """Create loaders for the configured test datasets."""
        self.test_loaders = [
            EADataLoader(
                test_dir, fold_id=getattr(self.args, "fold_id", 1), args=self.args
            )
            for test_dir in self.args.test_dirs
        ]

    def save_model(self, is_best=False, epoch=0, save_path=""):
        """Save model and optimizer state when the current epoch is best."""
        checkpoint_dict = {}
        checkpoint_dict["state_dict"] = self.model.state_dict()
        checkpoint_dict["optimizer_state_dict"] = self.optimizer.state_dict()
        checkpoint_dict["epoch_id"] = epoch

        if is_best:
            print(f"Saving Best Model to {save_path}/model_best.tar")
            out_tar = os.path.join(save_path, "model_best.tar")
            torch.save(checkpoint_dict, out_tar)

    def load_checkpoint(self, input_file, gpu):
        """Load ``model_best.tar`` from a checkpoint directory."""
        input_file = os.path.join(input_file, "model_best.tar")
        if os.path.isfile(os.path.join(os.getcwd(), input_file)):
            print(f"=> loading checkpoint '{os.path.join(os.getcwd(), input_file)}'")
            if torch.cuda.is_available():
                checkpoint = torch.load(
                    os.path.join(os.getcwd(), input_file), map_location=f"cuda:{gpu}"
                )
            else:
                checkpoint = torch.load(
                    os.path.join(os.getcwd(), input_file), map_location="cpu"
                )
            for key in list(checkpoint["state_dict"].keys()):
                if key not in list(self.model.state_dict().keys()):
                    del checkpoint["state_dict"][key]
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            print(f"=> no checkpoint found at '{input_file}'")

    def train_batch(self, finetune=False, step_num_list=None, max_step=1000):
        """Run one training epoch and return validation MRR, log text, and loss."""
        # This legacy argument was ignored by the original implementation.
        _ = step_num_list
        epoch_loss = 0
        success = False
        while not success:
            try:  # if the memory is not enough, reduce the batch size
                for train_loader in self.train_loaders:
                    train_loader.train_links = train_loader.train_links[
                        np.random.permutation(len(train_loader.train_links))
                    ]

                batch_size = (
                    self.args.train_batch_size
                    if hasattr(self.args, "train_batch_size")
                    else 256
                )
                n_batchs = [
                    len(train_loader.train_links) // batch_size
                    + int(len(train_loader.train_links) % batch_size > 0)
                    for train_loader in self.train_loaders
                ]
                train_batchs = [
                    [loader_id, batch_id]
                    for loader_id, batches in enumerate(n_batchs)
                    for batch_id in range(batches)
                ]
                random.shuffle(train_batchs)

                t_time = time.time()
                self.model.train()
                count_step = 0
                for batch in tqdm(train_batchs, ncols=100, position=0, leave=True):
                    count_step += 1
                    if max_step is not None and count_step > max_step:
                        break
                    self.model.zero_grad()
                    j, i = batch
                    now_loader = self.train_loaders[j]
                    batch_size = (
                        self.args.train_batch_size
                        if hasattr(self.args, "train_batch_size")
                        else 256
                    )
                    triple = self.train_loaders[j].train_links[
                        i * batch_size : min(
                            len(now_loader.train_links), (i + 1) * batch_size
                        )
                    ]

                    if len(triple) == 0:
                        continue

                    triple = triple.astype(np.int64)
                    subs = triple[:, 0]
                    objs = triple[:, 1]

                    if random.random() > 0.5:
                        source = subs
                        target = objs
                        order = True
                    else:
                        source = objs
                        target = subs
                        order = False
                    scores_all, source_kg_entities, target_kg_entities, _ = (
                        self.model.encode_entities_with_alignment(
                            source, now_loader, objs=target, mode="train"
                        )
                    )
                    pos_scores = scores_all[
                        [
                            torch.arange(len(scores_all)).to(self.args.device),
                            torch.LongTensor(target).to(self.args.device),
                        ]
                    ]
                    scores = scores_all
                    valid_samples = torch.nonzero(pos_scores != 0).squeeze()
                    scores = scores[valid_samples]
                    pos_scores = pos_scores[valid_samples]
                    if len(scores.shape) == 1:
                        scores = scores.unsqueeze(0)
                    scores = scores[
                        :, target_kg_entities if order else source_kg_entities
                    ]
                    max_n = torch.max(scores, 1, keepdim=True)[0]
                    loss = torch.mean(
                        -pos_scores
                        + max_n
                        + torch.log(torch.sum(torch.exp(scores - max_n), 1))
                    )
                    loss.backward()
                    self.optimizer.step()

                    # avoid NaN
                    for p in self.model.parameters():
                        X = p.data.clone()
                        flag = torch.isnan(X)
                        X[flag] = np.random.random()
                        p.data.copy_(X)
                    epoch_loss += loss.item()
                self.scheduler.step()
                self.t_time += time.time() - t_time
                if finetune:  # if finetune, we do not need to validate the model
                    valid_mrr, out_str = 0, 0
                else:
                    valid_mrr, out_str = self.evaluate(mode="valid")
                success = True
            except Exception:
                if "CUDA out of memory" in str(sys.exc_info()[1]):
                    print("CUDA out of memory, try to reduce batch size")
                    # Reduce the actual batch size, not entity_num
                    if not hasattr(self.args, "train_batch_size"):
                        self.args.train_batch_size = 256
                    self.args.train_batch_size = self.args.train_batch_size // 2
                    print("batch size reduced to", self.args.train_batch_size)
                else:
                    print("Unexpected error:", sys.exc_info()[0])
                    raise
                torch.cuda.empty_cache()
        return valid_mrr, out_str, epoch_loss

    def evaluate(self, mode="test"):
        """Evaluate a configured split and aggregate its per-dataset MRR."""
        if mode == "valid":
            loaders = self.valid_loaders
        elif mode == "test":
            loaders = self.test_loaders
        elif mode == "train":
            loaders = self.train_loaders
        else:
            raise ValueError("evaluation mode error")

        v_mrr_list = []
        out_str_list = []
        with torch.no_grad():
            for loader in loaders:
                success = False
                while not success:
                    try:
                        if mode == "train":
                            eval_links = loader.train_links
                        elif mode == "valid":
                            eval_links = loader.valid_links
                        elif mode == "test":
                            eval_links = loader.test_links
                            if len(eval_links) > 10000:
                                # 打乱顺序随机选10000个
                                eval_links = eval_links[
                                    np.random.permutation(len(eval_links))[:10000]
                                ]
                        else:
                            eval_links = loader.train_links

                        batch_size = (
                            self.args.test_batch_size
                            if hasattr(self.args, "test_batch_size")
                            else 128
                        )
                        n_data = len(eval_links)
                        n_batch = n_data // batch_size + int(n_data % batch_size > 0)

                        self.model.eval()
                        i_time = time.time()

                        ranking = []
                        for i in tqdm(
                            range(n_batch), ncols=100, position=0, leave=True
                        ):
                            start = i * batch_size
                            end = min(n_data, (i + 1) * batch_size)
                            batch_links = eval_links[start:end]

                            subs = batch_links[:, 0]
                            objs = batch_links[:, 1]
                            scores_all, _, target_kg_entities, _ = (
                                self.model.encode_entities_with_alignment(
                                    subs, loader, objs=objs, mode="eval"
                                )
                            )
                            scores = scores_all
                            # Rank only target-KG entities and exclude visible train anchors.
                            scores[:, target_kg_entities] += 1000000
                            links = torch.from_numpy(loader.train_links).to(
                                self.args.device
                            )
                            filt = torch.cat([links[:, 0], links[:, 1]])
                            scores[:, filt] -= 2000000
                            objs = torch.from_numpy(objs).to(self.args.device)
                            ranks = cal_ranks(scores, objs)
                            ranks = ranks.cpu().numpy()
                            ranking += list(ranks)

                        ranking = np.array(ranking)
                        v_mrr, v_h1, v_h3, v_h5, v_h10 = cal_performance(ranking)

                        i_time = time.time() - i_time
                        out_str = (
                            f"{loader.name} MRR:{v_mrr:.4f} H@1:{v_h1:.4f} "
                            f"H@3:{v_h3:.4f} H@5:{v_h5:.4f} H@10:{v_h10:.4f}"
                            f"[TIME] train:{self.t_time:.4f} inference:{i_time:.4f}\n"
                        )
                        self.args.logger.info(out_str)
                        print(out_str)
                        v_mrr_list.append(v_mrr.item())
                        out_str_list.append(out_str)
                        success = True
                    except Exception:
                        if "CUDA out of memory" in str(sys.exc_info()[1]):
                            print("CUDA out of memory, try to reduce batch size")
                            loader.args.test_batch_size = (
                                loader.args.test_batch_size // 2
                            )
                            print("batch size reduced to", loader.args.test_batch_size)
                        else:
                            print("Unexpected error:", sys.exc_info()[0])
                            raise
        v_mrr = np.mean(v_mrr_list)
        out_str = "".join(out_str_list)
        return v_mrr, out_str
