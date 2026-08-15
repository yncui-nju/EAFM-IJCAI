# EAFM-IJCAI

This repository contains the reference implementation and benchmark data for
the EAFM entity-alignment experiments reported in the IJCAI submission.
EAFM learns entity representations from the relational structure of two
knowledge graphs and evaluates cross-graph entity rankings.

## Repository contents

```text
EAFM-IJCAI/
+-- src/
|   +-- pretrain.py          # multi-dataset pretraining entry point
|   +-- finetune.py          # checkpoint fine-tuning entry point
|   +-- evaluation.py        # checkpoint evaluation entry point
|   +-- experiment.py        # training, validation, and ranking logic
|   +-- data_loader.py       # processed dataset and graph construction
|   +-- encoder/             # entity and relation graph encoders
|   +-- utils.py             # ranking metrics and shared utilities
|   +-- requirements.txt
|   +-- README.md            # command-line reference
+-- datasets/
    +-- EA/
        +-- OpenEA/          # OpenEA benchmark variants
        +-- SRPRS/           # SRPRS benchmark variants
        +-- Others/          # additional benchmarks
        +-- processed_data/  # generated arrays and mappings
```

Dataset identifiers use the form `<collection>#<dataset>`, for example
`OpenEA#D_W_15K_V1`.

## Requirements

The reference environment is Python 3.9 with PyTorch 2.2.0 and CUDA 11.8.
Install a PyTorch build and a matching `torch-scatter` wheel for the target
CUDA runtime, then install the remaining packages:

```bash
cd src
python -m pip install -r requirements.txt
```

For large datasets, use a CUDA-capable GPU and choose a batch size that fits
available memory. Checkpoints and logs are created outside `src/` by the
command-line programs.

## Pretraining

Run from `src/` so the default paths resolve to the repository root:

```bash
cd src
python pretrain.py \
  --train_dataset_list "OpenEA#D_W_15K_V1 OpenEA#EN_DE_15K_V1" \
  --valid_dataset_list "OpenEA#D_W_15K_V1 OpenEA#EN_DE_15K_V1" \
  --test_dataset_list "OpenEA#D_W_15K_V1 OpenEA#EN_DE_15K_V1" \
  --gpu 0 \
  --seed 1234
```

The best validation checkpoint is saved as `model_best.tar` in the generated
checkpoint directory. The configured test datasets are evaluated after
training and the results are written under `../log/`.

Useful options include `--hidden_dim`, `--n_layer`,
`--n_relation_encoder_layer`, `--train_batch_size`, `--test_batch_size`,
`--K`, and `--fold_id`. Structural model components can be selected with
`--use_merge_relation_graph`, `--use_anchor_conditioned_initialize`,
`--use_two_tower`, and `--use_interact`.

## Checkpoint evaluation

`--checkpoint_path` points to a directory containing `model_best.tar`:

```bash
cd src
python evaluation.py \
  --checkpoint_path ../checkpoint/pretrain/<run-directory> \
  --test_dataset_list "OpenEA#D_W_15K_V1 OpenEA#D_Y_15K_V1" \
  --gpu 0 \
  --seed 1234 \
  --K 2 \
  --test_batch_size 32
```

Keep the encoder configuration consistent with the configuration used to
produce the checkpoint.

## Fine-tuning

```bash
cd src
python finetune.py \
  --checkpoint_path ../checkpoint/pretrain/<run-directory> \
  --train_dataset_list "OpenEA#D_W_15K_V1" \
  --valid_dataset_list "OpenEA#D_W_15K_V1" \
  --test_dataset_list "OpenEA#D_W_15K_V1" \
  --fold_id 1 \
  --gpu 0 \
  --seed 1234
```

Fine-tuning writes its checkpoint and log output under the configured paths.

## Dataset layout

Before training or evaluation, generate the processed datasets in the following
form:

```text
datasets/EA/processed_data/<collection>/<dataset>/
|-- id_mappings.pkl
|-- triples_id_1.npy
|-- triples_id_2.npy
|-- alignment_pairs.npy
|-- statistics.json
`-- 5fold/<fold_id>/
    |-- train_links.npy
    |-- valid_links.npy
    `-- test_links.npy
```

The repository retains the raw benchmark folders for provenance and
preprocessing. The preprocessing scripts are located in the corresponding
collection directory under `datasets/EA/`. The generated `processed_data/`
directory is intentionally not versioned; create it locally before running
the model.

## Git LFS

Some benchmark arrays are tracked with Git LFS. Install Git LFS before cloning
or pulling the complete dataset:

```bash
git lfs install
git clone https://github.com/yncui-nju/EAFM-IJCAI.git
cd EAFM-IJCAI
git lfs pull
```

## Citation

If you use this implementation or its processed benchmark files, please cite
the IJCAI 2026 paper:

```bibtex
@inproceedings{cui2026breaking,
  title     = {Breaking the Reasoning Horizon in Entity Alignment Foundation Models},
  author    = {Cui, Yuanning and Sun, Zequn and Hu, Wei and Xin, Kexuan and Fu, Zhangjie},
  booktitle = {Proceedings of the Thirty-Fifth International Joint Conference on Artificial Intelligence},
  year      = {2026}
}
```
