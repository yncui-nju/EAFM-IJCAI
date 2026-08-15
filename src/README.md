# EAFM IJCAI implementation

This directory contains the training, fine-tuning, and evaluation code for the IJCAI release, including the paper's model computation, split handling, checkpoint format, and ranking protocol.

## Environment

The reference environment uses Python 3.9, PyTorch 2.2.0, and CUDA 11.8. Install PyTorch and `torch-scatter` with wheels matching the local CUDA runtime, then install the remaining packages from `requirements.txt`.

Before running the model, generate processed datasets with the preprocessing
scripts under `datasets/EA/`. The generated files must follow this layout:

```text
datasets/processed_data/<collection>/<dataset>/
  id_mappings.pkl
  triples_id_1.npy
  triples_id_2.npy
  alignment_pairs.npy
  5fold/<fold_id>/{train,valid,test}_links.npy
```

Dataset arguments use the form `<collection>#<dataset>`, for example `OpenEA#D_W_15K_V1`.

## Pretraining

Run commands from this directory so that the default relative paths resolve to the repository-level `datasets`, `checkpoint`, and `log` directories.

```bash
cd src_EAFM_IJCAI
python pretrain.py \
  --train_dataset_list "OpenEA#D_W_15K_V1 OpenEA#EN_DE_15K_V1" \
  --valid_dataset_list "OpenEA#D_W_15K_V1 OpenEA#EN_DE_15K_V1" \
  --test_dataset_list "OpenEA#D_W_15K_V1 OpenEA#EN_DE_15K_V1" \
  --gpu 0 --seed 1234
```

The validation-best checkpoint is written as `model_best.tar` under the configured `--save_path`.

## Evaluation

```bash
python evaluation.py \
  --checkpoint_path /path/to/checkpoint_directory \
  --test_dataset_list "OpenEA#D_W_15K_V1 OpenEA#D_Y_15K_V1" \
  --gpu 0 --seed 1234 --K 2 --test_batch_size 32
```

`--checkpoint_path` must point to the directory containing `model_best.tar`, not to the tar file itself.
When evaluating a checkpoint produced by `pretrain.py`, keep `--K 2` and
`--test_batch_size 32` aligned with the pretraining configuration.

## Full 29-dataset evaluation

The complete OpenEA, SRPRS, and Others benchmark list is built into
`run_full_29_eval.py`. It writes one log per dataset, `details.csv`, and a
`COMPLETE` marker after every dataset succeeds. Use one physical GPU per
worker; `--test_batch_size 16` is a memory-safe evaluation setting for the
100K/200K-entity datasets.

```bash
python run_full_29_eval.py \
  --checkpoint_suffix ijcai_release_seed1234_20260811_1746 \
  --output_dir ../log/eafm_ijcai_full29 \
  --gpus GPU-UUID-1 GPU-UUID-2 GPU-UUID-3
```

## Fine-tuning

```bash
python finetune.py \
  --checkpoint_path /path/to/checkpoint_directory \
  --train_dataset_list "OpenEA#D_W_15K_V1" \
  --valid_dataset_list "OpenEA#D_W_15K_V1" \
  --test_dataset_list "OpenEA#D_W_15K_V1" \
  --gpu 0 --seed 1234
```


## Main files

- `encoder/EntityEncoder.py`: relation-graph and entity-graph message passing.
- `data_loader.py`: dataset splits, alignment mappings, and relation-graph construction.
- `experiment.py`: training, checkpoint loading, validation, and ranking evaluation.
- `pretrain.py`, `finetune.py`, `evaluation.py`: command-line entry points.
- `utils.py`: ranking metrics.
