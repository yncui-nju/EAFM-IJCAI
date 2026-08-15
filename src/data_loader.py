"""Dataset and graph construction utilities for entity alignment."""

import json
import os
import pickle
import time

import numpy as np
import torch
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer


def _mxnet_device_id(device_str: str) -> int:
    """Map the configured torch device to the legacy encoder device value."""
    if device_str is None:
        return -1
    d = str(device_str).lower()
    if d == "cpu":
        return -1
    if d.startswith("cuda"):
        if ":" in d:
            return int(d.split(":")[1])
        return 0
    return -1


class Reader:
    """Read one processed entity-alignment dataset from disk."""

    def __init__(self, data_dir, args=None):
        """
        Initialize the Entity Alignment Dataset Loader

        Args:
            data_dir: Path to the processed dataset directory
            args: Arguments containing additional configuration
        """
        self.args = args
        self.data_dir = data_dir
        self.name = os.path.basename(data_dir)
        # Load ID mappings
        self._load_id_mappings()

        # Load KG triples
        self._load_kg_data()

        # Load alignment data
        self._load_alignment_data()

        # Load fold data (if exists)
        self._load_fold_data()
        if self.args.use_text_embeddings.lower() == "true":
            self.entity_text = self._load_entity_text()

    def _load_id_mappings(self):
        """Load entity and relation ID mappings"""
        mapping_file = os.path.join(self.data_dir, "id_mappings.pkl")
        if os.path.exists(mapping_file):
            with open(mapping_file, "rb") as f:
                mappings = pickle.load(f)
                self.entity2id_1 = mappings["entity2id_1"]
                self.entity2id_2 = mappings["entity2id_2"]
                self.relation2id_1 = mappings["relation2id_1"]
                self.relation2id_2 = mappings["relation2id_2"]
        else:
            raise FileNotFoundError(f"ID mappings file not found: {mapping_file}")

        self.id2entity_1 = {v: k for k, v in self.entity2id_1.items()}
        self.id2entity_2 = {v: k for k, v in self.entity2id_2.items()}
        self.id2relation_1 = {v: k for k, v in self.relation2id_1.items()}
        self.id2relation_2 = {v: k for k, v in self.relation2id_2.items()}

        self.entity_num_1 = len(self.entity2id_1)
        self.entity_num_2 = len(self.entity2id_2)
        self.relation_num_1 = len(self.relation2id_1)
        self.relation_num_2 = len(self.relation2id_2)

        self.relation_list_1 = list(self.id2relation_1)
        self.relation_list_2 = list(self.id2relation_2)
        self.entity_list_1 = list(self.id2entity_1)
        self.entity_list_2 = list(self.id2entity_2)

        # tensor化
        self.entity_list_1 = torch.tensor(self.entity_list_1, dtype=torch.long).to(
            self.args.device
        )
        self.entity_list_2 = torch.tensor(self.entity_list_2, dtype=torch.long).to(
            self.args.device
        )
        self.relation_list_1 = torch.tensor(self.relation_list_1, dtype=torch.long).to(
            self.args.device
        )
        self.relation_list_2 = torch.tensor(self.relation_list_2, dtype=torch.long).to(
            self.args.device
        )

    def _load_kg_data(self):
        """Load knowledge graph triples"""
        # Load KG1 triples
        kg1_file = os.path.join(self.data_dir, "triples_id_1.npy")
        if os.path.exists(kg1_file):
            self.kg1_triples = np.load(kg1_file)
        else:
            self.kg1_triples = np.empty((0, 3), dtype=np.int32)

        # Load KG2 triples
        kg2_file = os.path.join(self.data_dir, "triples_id_2.npy")
        if os.path.exists(kg2_file):
            self.kg2_triples = np.load(kg2_file)
        else:
            self.kg2_triples = np.empty((0, 3), dtype=np.int32)

    def _load_alignment_data(self):
        """Load entity alignment pairs"""
        alignment_file = os.path.join(self.data_dir, "alignment_pairs.npy")
        if os.path.exists(alignment_file):
            self.alignment_pairs = np.load(alignment_file)
        else:
            self.alignment_pairs = np.empty((0, 2), dtype=np.int32)

    def _load_entity_text(self):
        """
        读取 OpenEA 数据集对应的实体文本特征（合并后的 json）。

        参数:
            dataset_name: str
                例如 "D_W_15K_V1", "EN_DE_100K_V2"

        返回:
            ent2text: Dict[str, str]
                key   : 实体 URI / 名称
                value : 对应的文本描述
        """

        # src_EAFM/data_loader.py
        # 项目根目录 = src_EAFM/../
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        text_path = os.path.join(
            project_root,
            "datasets",
            "processed_data",
            "OpenEA",
            self.name,
            "ent_text_merged.json",
        )

        if not os.path.exists(text_path):
            raise FileNotFoundError(f"[OpenEA] Entity text file not found: {text_path}")

        with open(text_path, "r", encoding="utf-8") as f:
            ent2text = json.load(f)

        return ent2text

    def _load_fold_data(self):
        """Load fold data for train/valid/test splits"""
        self.fold_data = {}
        fold_dir = os.path.join(self.data_dir, "5fold")

        if os.path.exists(fold_dir):
            for fold_id in range(1, 6):  # 5 folds
                fold_path = os.path.join(fold_dir, str(fold_id))
                if os.path.exists(fold_path):
                    self.fold_data[fold_id] = {}

                    # Load train data
                    train_file = os.path.join(fold_path, "train_links.npy")
                    if os.path.exists(train_file):
                        self.fold_data[fold_id]["train"] = np.load(train_file)

                    # Load valid data
                    valid_file = os.path.join(fold_path, "valid_links.npy")
                    if os.path.exists(valid_file):
                        self.fold_data[fold_id]["valid"] = np.load(valid_file)

                    # Load test data
                    test_file = os.path.join(fold_path, "test_links.npy")
                    if os.path.exists(test_file):
                        self.fold_data[fold_id]["test"] = np.load(test_file)

                    # ===== 新增逻辑 =====
                    train_links = self.fold_data[fold_id].get("train", None)
                    valid_links = self.fold_data[fold_id].get("valid", None)
                    test_links = self.fold_data[fold_id].get("test", None)

                    if (
                        train_links is not None
                        and valid_links is not None
                        and test_links is not None
                    ):
                        total_links = (
                            len(train_links) + len(valid_links) + len(test_links)
                        )
                        train_ratio = len(train_links) / total_links

                        # 如果train超过30%
                        if (
                            train_ratio > float(self.args.ea_rate)
                            and float(self.args.ea_rate) < 0.3
                        ):
                            target_num = int(total_links * float(self.args.ea_rate))

                            # 随机采样
                            if target_num < len(train_links):
                                indices = np.random.choice(
                                    len(train_links), target_num, replace=False
                                )
                                self.fold_data[fold_id]["train"] = train_links[indices]

                                print(
                                    f"#########################[Fold {fold_id}] Downsample train: {len(train_links) + len(valid_links)} -> {target_num + len(valid_links)}"
                                )

    def get_kg_data(self, kg_id):
        """
        Get knowledge graph data

        Args:
            kg_id: Knowledge graph ID (1 or 2)

        Returns:
            numpy array of triples
        """
        if kg_id == 1:
            return self.kg1_triples
        elif kg_id == 2:
            return self.kg2_triples
        else:
            raise ValueError("kg_id must be 1 or 2")

    def get_alignment_data(self):
        """
        Get entity alignment pairs

        Returns:
            numpy array of alignment pairs [entity1_id, entity2_id]
        """
        return self.alignment_pairs

    def get_fold_data(self, fold_id, split):
        """
        Get train/valid/test data for a specific fold

        Args:
            fold_id: Fold ID (1-5)
            split: Data split ('train', 'valid', or 'test')

        Returns:
            numpy array of entity pairs
        """
        if fold_id in self.fold_data and split in self.fold_data[fold_id]:
            return self.fold_data[fold_id][split]
        else:
            return np.empty((0, 2), dtype=np.int32)

    def get_statistics(self):
        """
        Get dataset statistics

        Returns:
            dict with dataset statistics
        """
        stats_file = os.path.join(self.data_dir, "statistics.json")
        if os.path.exists(stats_file):
            with open(stats_file, "r") as f:
                return json.load(f)
        else:
            return {
                "kg1_entities": self.entity_num_1,
                "kg2_entities": self.entity_num_2,
                "kg1_relations": self.relation_num_1,
                "kg2_relations": self.relation_num_2,
                "kg1_triples": len(self.kg1_triples),
                "kg2_triples": len(self.kg2_triples),
                "alignment_pairs": len(self.alignment_pairs),
                "folds": len(self.fold_data),
            }


class EAKG:
    """Store one side of an entity-alignment dataset."""

    def __init__(self, data_loader, kg_id):
        """
        Knowledge Graph representation for Entity Alignment

        Args:
            data_loader: EADatasetLoader instance
            kg_id: Knowledge graph ID (1 or 2)
        """
        self.data_loader = data_loader
        self.kg_id = kg_id

        # Load triples
        self.triples = data_loader.get_kg_data(kg_id)

        # Load entity and relation mappings
        if kg_id == 1:
            self.entity2id = data_loader.entity2id_1
            self.relation2id = data_loader.relation2id_1
            self.id2entity = data_loader.id2entity_1
            self.id2relation = data_loader.id2relation_1
            self.entity_num = data_loader.entity_num_1
            # Original relation count
            self.original_relation_num = data_loader.relation_num_1
            # Will be set later based on global relation count
            self.relation_num = None
            self.relation_list = data_loader.relation_list_1
            self.entity_list = data_loader.entity_list_1
            self.name = data_loader.name.split("_")[0]
        else:
            self.entity2id = data_loader.entity2id_2
            self.relation2id = data_loader.relation2id_2
            self.id2entity = data_loader.id2entity_2
            self.id2relation = data_loader.id2relation_2
            self.entity_num = data_loader.entity_num_2
            # Original relation count
            self.original_relation_num = data_loader.relation_num_2
            # Will be set later based on global relation count
            self.relation_num = None
            self.relation_list = data_loader.relation_list_2
            self.entity_list = data_loader.entity_list_2
            self.name = data_loader.name.split("_")[1]

    def _add_inverse_triples(self, global_relation_offset):
        """Add inverse triples to the knowledge graph"""
        inverse_triples = []
        for h, r, t in self.triples:
            # Add inverse relation (r + global_relation_offset)
            inverse_triples.append((t, r + global_relation_offset, h))

        # Combine original and inverse triples
        self.triples = np.vstack([self.triples, inverse_triples])


class EADataLoader:
    """Build split-aware alignment mappings and graph tensors for one dataset."""

    def __init__(self, data_dir, fold_id=1, args=None):
        """
        Entity Alignment Data Loader

        Args:
            data_dir: Path to the processed dataset directory
            fold_id: Fold ID for train/valid/test splits (1-5)
            args: Arguments containing additional configuration
        """
        self.args = args
        self.reader = Reader(data_dir, args)
        self.fold_id = fold_id

        # Create KG representations
        self.kg1 = EAKG(self.reader, 1)
        self.kg2 = EAKG(self.reader, 2)

        # Set global relation count and update individual KGs
        self._setup_global_relations()

        # 合并两个KG的id2relation
        self.id2relation = {**self.kg1.id2relation, **self.kg2.id2relation}
        # 获取relation2inverse
        self.relation2inverse = {
            relation: relation + self.global_relation_offset
            for relation in self.id2relation
        }
        self.relation2inverse.update(
            {
                relation + self.global_relation_offset: relation
                for relation in self.id2relation
            }
        )

        # Add inverse triples to both KGs with global relation offsets
        self.kg1._add_inverse_triples(self.global_relation_offset)
        self.kg2._add_inverse_triples(self.global_relation_offset)

        # Update relation counts after adding inverse triples
        self.kg1.relation_num = self.global_relation_count
        self.kg2.relation_num = self.global_relation_count

        # Load train/valid/test splits
        self.train_links = self.reader.get_fold_data(fold_id, "train")
        self.valid_links = self.reader.get_fold_data(fold_id, "valid")
        self.test_links = self.reader.get_fold_data(fold_id, "test")
        if self.args.use_text_embeddings.lower() == "true":
            self.entity_text = self.reader.entity_text
        self.entity_num = (
            max(max(self.kg1.id2entity.keys()), max(self.kg2.id2entity.keys())) + 1
        )
        self.relation_num = (
            self.global_relation_count + 2
        )  # [-1]是self-loop， [-2]是对齐

        # All alignment pairs
        self.alignment_pairs = self.reader.get_alignment_data()

        # Build relation graphs
        tic = time.time()
        self.init_ea_relation_graph_cache()
        self.args.logger.info("Generate relation graph time:" + str(time.time() - tic))

        self.build_loader_fact_matrices()

        # Create entity alignment mapping for efficient lookup
        self._build_alignment_mapping()

        # Create global entity and relation mappings
        self._build_global_mappings()

        dataset_family = os.path.basename(os.path.dirname(os.path.normpath(data_dir)))
        dataset_name = os.path.basename(os.path.normpath(data_dir))
        self.name = dataset_family + "_" + dataset_name
        self.args.logger.info(
            "%s: %d entities, %d relations",
            self.name,
            self.entity_num,
            self.relation_num,
        )

        if self.args.use_text_embeddings.lower() == "true":
            mx_device = _mxnet_device_id(args.device)
            self.text_encoder = SentenceTransformer(
                getattr(args, "text_encoder_path", "BAAI/bge-large-en-v1.5"),
                trust_remote_code=True,
                device=mx_device,
            )

            # 冻结全部参数
            for param in self.text_encoder.parameters():
                param.requires_grad = False

            # 明确设为 eval 模式（关闭 dropout 等）
            self.text_encoder.eval()

            self.ent_embeddings = self.build_entity_text_embeddings_st(
                ent2text=self.entity_text,
                text_encoder=self.text_encoder,
                batch_size=128,
                max_length=64,  # 可选：建议 64 或 128
                normalize=False,  # 可选：如果后面用 cosine 相似度可设 True
            )

            self.ent_embeddings = self.ent_embeddings.to(args.device)

            self.args.logger.info("Encoded entity text for %s", self.name)

    @torch.no_grad()
    def build_entity_text_embeddings_st(
        self,
        ent2text,
        text_encoder,  # SentenceTransformer 实例
        batch_size: int = 64,
        max_length=None,
        normalize: bool = False,
    ):
        """
        用 SentenceTransformer 编码实体文本，按 entity id 顺序返回 entity_embeddings。
        缺失文本：entity.split('/')[-1]

        返回:
            entity_embeddings: torch.FloatTensor [num_entities, hidden_dim] (CPU tensor)
        """
        num_entities = len(self.id2entity)

        # 1) 按 id 顺序准备文本
        texts = []
        for eid in range(num_entities):
            entity = self.id2entity[eid]
            text = ent2text.get(entity, entity.split("/")[-1])
            texts.append(text)

        # 2) 如果你想限制最大长度（对 transformer 类模型有效）
        # SentenceTransformer exposes this attribute for transformer encoders.
        if max_length is not None and hasattr(text_encoder, "max_seq_length"):
            text_encoder.max_seq_length = int(max_length)

        # 3) 编码（SentenceTransformer 直接返回句向量）
        emb = text_encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,  # 直接返回 torch.Tensor
            normalize_embeddings=normalize,
        )

        # 4) 确保行号与 id 对齐
        assert emb.size(0) == num_entities, (
            f"embedding num mismatch: {emb.size(0)} vs {num_entities}"
        )

        # 按你的工程习惯：通常把 embedding 放 CPU，后续再 .to(device)
        return emb.detach().cpu()

    def _setup_global_relations(self):
        """Setup global relation count and offset for inverse relations"""
        # Total number of unique relations in both KGs
        self.global_relation_count = (
            max(
                max(self.kg1.id2relation.keys()) if self.kg1.id2relation.keys() else -1,
                max(self.kg2.id2relation.keys()) if self.kg2.id2relation.keys() else -1,
            )
            + 1
        )

        # Offset for inverse relations (relations + offset = inverse relations)
        self.global_relation_offset = self.global_relation_count

        # Double the relation count to account for inverse relations
        self.global_relation_count *= 2

    def _build_alignment_mapping(self):
        """Build mappings for efficient entity alignment lookup"""
        # Mapping from KG1 entities to KG2 entities
        self.kg1_to_kg2 = {}
        # Mapping from KG2 entities to KG1 entities
        self.kg2_to_kg1 = {}

        for kg1_entity, kg2_entity in self.train_links:
            self.kg1_to_kg2[kg1_entity] = kg2_entity
            self.kg2_to_kg1[kg2_entity] = kg1_entity

        self.kg1_to_kg2_test = {}
        self.kg2_to_kg1_test = {}
        for kg1_entity, kg2_entity in np.concatenate(
            (self.train_links, self.valid_links)
        ):
            self.kg1_to_kg2_test[kg1_entity] = kg2_entity
            self.kg2_to_kg1_test[kg2_entity] = kg1_entity

        # 合并到一起
        self.kg_to_kg = {**self.kg1_to_kg2, **self.kg2_to_kg1}
        self.kg_to_kg_test = {**self.kg1_to_kg2_test, **self.kg2_to_kg1_test}

    def _build_global_mappings(self):
        """
        Build global entity and relation mappings combining both KGs.
        For id2relation, when there are duplicate IDs, the value is a list.
        """
        # Combine entity mappings
        self.entity2id = {**self.kg1.entity2id, **self.kg2.entity2id}

        # Create reverse mapping for entities
        self.id2entity = {id_: entity for entity, id_ in self.entity2id.items()}

        # Combine relation mappings
        self.relation2id = {**self.kg1.relation2id, **self.kg2.relation2id}

        # Create reverse mapping for relations
        # When there are duplicate IDs, store as a list
        self.id2relation = {}
        for relation, id_ in self.relation2id.items():
            if id_ in self.id2relation:
                # If ID already exists, convert to list or append to existing list
                if not isinstance(self.id2relation[id_], list):
                    self.id2relation[id_] = [self.id2relation[id_]]
                self.id2relation[id_].append(relation)
            else:
                self.id2relation[id_] = relation

    def build_loader_fact_matrices(self):
        """Build the unified triple tensor and its head-incidence matrix."""
        triples = np.concatenate((self.kg1.triples, self.kg2.triples))
        if self.args.use_two_tower.lower() != "true":
            self.args.logger.info("Using alignment edges in the unified entity graph")
            align_triples = []
            for e1, e2 in self.train_links:
                align_triples.append([e1, self.relation_num - 2, e2])
                align_triples.append([e2, self.relation_num - 2, e1])
            triples = np.concatenate((triples, align_triples))

        # 转成 torch tensor
        graph = torch.LongTensor(triples).to(self.args.device)
        self.graph = graph
        fact_num = graph.size(0)
        entity_num = max(graph[:, 0].max().item(), graph[:, 2].max().item()) + 1
        # ===== 三元组 -> 头实体 的稀疏矩阵 M_sub =====
        indices_sub = torch.cat(
            [
                torch.arange(fact_num, device=self.args.device).long().unsqueeze(1),
                graph[:, 0].unsqueeze(1),  # 头实体 id
            ],
            dim=1,
        ).t()  # 形状 (2, nnz)
        values = torch.ones(fact_num, device=self.args.device)
        size = torch.Size([fact_num, entity_num])
        self.M_sub = torch.sparse_coo_tensor(indices_sub, values, size).to(
            self.args.device
        )

    def merge_kgs_and_build_relation_graph(
        self, kg1_triples, kg2_triples, align_pairs, device="cpu"
    ):
        """
        合并两套 KG，并构建关系图（HH/HT/TH/TT）。
        输入:
            kg1_triples: (n1, 3) numpy, [h, r, t]
            kg2_triples: (n2, 3) numpy, [h, r, t]
            align_pairs: (m, 2) numpy, [e1_in_kg1, e2_in_kg2]
        输出:
            merged_triples: (N, 3) torch.LongTensor
            relation_graph: {
                "num_relations": int,
                "edge_index":   (E, 2) torch.LongTensor,
                "edge_type":    (E,)   torch.LongTensor   # 0=HH,1=HT,2=TH,3=TT
            }
        """
        # ---------- 0. numpy -> torch ----------
        kg1 = torch.as_tensor(kg1_triples, dtype=torch.long, device=device).clone()
        kg2 = torch.as_tensor(kg2_triples, dtype=torch.long, device=device).clone()
        align = torch.as_tensor(align_pairs, dtype=torch.long, device=device)

        if self.args.use_merge_relation_graph.lower() == "true":
            # ---------- 1. 合并实体 ----------
            # KG1 实体范围
            max_ent_kg1 = torch.max(torch.stack([kg1[:, 0].max(), kg1[:, 2].max()]))
            # KG2 实体范围（原始 id）
            max_ent_kg2 = torch.max(torch.stack([kg2[:, 0].max(), kg2[:, 2].max()]))

            # 验证kg2中的实体id都大于kg1的最大实体id
            assert torch.all(kg2[:, 0] > max_ent_kg1)

            ent_offset = max_ent_kg1 + 1

            # KG2 原始实体 id -> 合并后实体 id
            mapping_kg2 = (
                torch.arange(max_ent_kg2 + 1, dtype=torch.long, device=device)
                + ent_offset
            )
            if align.numel() > 0:
                e1 = align[:, 0]  # KG1 实体
                e2 = align[:, 1]  # KG2 原始实体 id
                mapping_kg2[e2] = e1  # 对齐的直接映射到 KG1 的 id

            # 应用实体映射到 KG2 三元组
            kg2_h_orig = kg2[:, 0].clone()
            kg2_t_orig = kg2[:, 2].clone()
            kg2[:, 0] = mapping_kg2[kg2_h_orig]
            kg2[:, 2] = mapping_kg2[kg2_t_orig]

            merged_triples = torch.cat([kg1, kg2], dim=0)  # [N, 3]
        else:
            align_triples = []
            for e1, e2 in align_pairs:
                align_triples.append([e1, self.relation_num - 2, e2])
            align_triples = torch.LongTensor(align_triples).to(self.args.device)
            merged_triples = torch.cat([kg1, kg2, align_triples], dim=0)
        triples_np = merged_triples.cpu().numpy()
        h = triples_np[:, 0]
        r = triples_np[:, 1]
        t = triples_np[:, 2]

        num_ents = int(max(h.max(), t.max()) + 1)
        num_rels = int(r.max() + 1)

        # ---------- 4. 用稀疏矩阵构建 HH / HT / TH / TT ----------

        # 头实体-关系  incidence: H[e, r] = 出现在 (e, r, *) 的次数
        data = np.ones_like(r, dtype=np.int8)
        H = csr_matrix((data, (h, r)), shape=(num_ents, num_rels))
        # 尾实体-关系  incidence: T[e, r] = 出现在 (*, r, e) 的次数
        T = csr_matrix((data, (t, r)), shape=(num_ents, num_rels))

        # 四种关系邻接矩阵 (num_rels x num_rels)，只关心非零位置即可
        HH = H.T @ H  # 共享头实体
        TT = T.T @ T  # 共享尾实体
        HT = H.T @ T  # 头 == 尾
        TH = T.T @ H  # 尾 == 头

        # 去掉 HH/TT 的对角线（你之前在 HH/TT 上排除了 i == j）
        HH.setdiag(0)
        TT.setdiag(0)
        HH.eliminate_zeros()
        TT.eliminate_zeros()
        # HT/TH 不去对角线（当 h==t 且同一 triple，也可能 r->r，这是你原先逻辑允许的）

        # 从稀疏矩阵中取出边
        edges_list = []
        types_list = []

        def extract_edges(mat, etype):
            if mat.nnz == 0:
                return
            row, col = mat.nonzero()
            edges_list.append(np.stack([row, col], axis=1))
            types_list.append(np.full(row.shape[0], etype, dtype=np.int64))

        extract_edges(HH, 0)  # HH
        extract_edges(HT, 1)  # HT
        extract_edges(TH, 2)  # TH
        extract_edges(TT, 3)  # TT
        for r1, r2 in self.relation2inverse.items():
            edges_list.append([[r1, r2]])
            types_list.append([4])

        if not edges_list:
            rel_edge_index = torch.empty((0, 2), dtype=torch.long, device=device)
            rel_edge_type = torch.empty((0,), dtype=torch.long, device=device)
        else:
            edges = np.concatenate(edges_list, axis=0)  # [E, 2]
            edge_types = np.concatenate(types_list, axis=0)  # [E]

            # 去重: (r_i, r_j, type) 唯一
            edge_view = np.concatenate([edges, edge_types[:, None]], axis=1)
            edge_unique = np.unique(edge_view, axis=0)
            edges_unique = edge_unique[:, :2]
            types_unique = edge_unique[:, 2]

            rel_edge_index = torch.as_tensor(
                edges_unique, dtype=torch.long, device=device
            )
            rel_edge_type = torch.as_tensor(
                types_unique, dtype=torch.long, device=device
            )

        relation_graph = {
            "num_relations": num_rels,
            "edge_index": rel_edge_index,
            "edge_type": rel_edge_type,
        }

        return relation_graph

    def init_ea_relation_graph_cache(self):
        """
        基于当前 KG1/KG2 和对齐数据，预先构建并缓存两份 EA 关系图：
            - self.ea_relation_graph_train       : 只用训练对齐对
            - self.ea_relation_graph_train_valid : 训练 + 验证对齐对
        """
        self.ea_relation_graph_train = self.merge_kgs_and_build_relation_graph(
            self.kg1.triples, self.kg2.triples, self.train_links, self.args.device
        )

        # 使用训练 + 验证对齐对
        if hasattr(self, "valid_links") and self.valid_links is not None:
            all_align = np.concatenate([self.train_links, self.valid_links], axis=0)
        else:
            all_align = self.train_links

        self.ea_relation_graph_valid = self.merge_kgs_and_build_relation_graph(
            self.kg1.triples, self.kg2.triples, all_align, self.args.device
        )

    def get_ea_relation_graph(self, use_train_only: bool = False):
        """
        获取 EA 任务使用的关系图。

        参数：
            use_train_only:
                - True  : 返回只用训练对齐对构建的图 (ea_relation_graph_train)
                - False : 返回使用训练 + 验证对齐对的图 (ea_relation_graph_train_valid)
        """
        if use_train_only:
            return self.ea_relation_graph_train
        else:
            return self.ea_relation_graph_valid

    def get_relation_index(self, mode="train"):
        """Return the split-appropriate relation-graph edges and edge types."""
        if mode == "train":
            return self.ea_relation_graph_train[
                "edge_index"
            ], self.ea_relation_graph_train["edge_type"]
        else:
            return self.ea_relation_graph_valid[
                "edge_index"
            ], self.ea_relation_graph_valid["edge_type"]

    def get_batch(self, batch_idx, mode="train"):
        """
        Get a batch of data

        Args:
            batch_idx: Indices of samples to include in the batch
            mode: Data mode ('train', 'valid', or 'test')

        Returns:
            Batch data based on mode
        """
        if mode == "train":
            links = self.train_links[batch_idx]
        elif mode == "valid":
            links = self.valid_links[batch_idx]
        elif mode == "test":
            links = self.test_links[batch_idx]
        else:
            raise ValueError("Mode must be 'train', 'valid', or 'test'")

        # Return entity pairs as numpy arrays with int64 type for consistency
        return links[:, 0].astype(np.int64), links[:, 1].astype(
            np.int64
        )  # entity1_ids, entity2_ids

    def get_aligned_neighbors_in_kg(self, subs, nodes, hop_count=3):
        """Expand source nodes and return their known cross-KG counterparts."""
        nodes_source = [[i, subs[i]] for i in range(len(subs))]
        nodes_target = []
        for _ in range(hop_count):
            nodes, _, _ = self.get_neighbors(nodes)
        for nd in nodes.detach().cpu().numpy():
            # 映射到另一KG
            if nd[1] not in self.kg_to_kg:
                continue
            if subs[nd[0]] == nd[1]:
                continue
            nodes_target.append([nd[0], self.kg_to_kg[nd[1]]])
            nodes_source.append([nd[0], nd[1]])

        return nodes_source, nodes_target

    def get_neighbors(self, nodes):
        """Expand batched nodes by one hop in the unified entity graph."""
        # Accept NumPy nodes from the caller and tensors from iterative hops.
        if type(nodes) == np.ndarray:
            indices_ = torch.cat(
                [
                    torch.from_numpy(nodes[:, 1])
                    .to(self.args.device)
                    .long()
                    .unsqueeze(1),
                    torch.from_numpy(nodes[:, 0])
                    .to(self.args.device)
                    .long()
                    .unsqueeze(1),
                ],
                dim=1,
            ).t()
        elif type(nodes) == torch.Tensor:
            indices_ = torch.cat(
                [nodes[:, 1].unsqueeze(1), nodes[:, 0].unsqueeze(1)], dim=1
            ).t()
        node_1hot = torch.sparse_coo_tensor(
            indices_,
            torch.ones(len(nodes)).to(self.args.device),
            torch.Size([self.entity_num, nodes.shape[0]]),
        )

        edge_1hot = torch.sparse.mm(self.M_sub, node_1hot)  # edge_idx x batch_idx
        edges = edge_1hot.indices()

        selected_edges = torch.index_select(self.graph, 0, edges[0])
        sampled_edges = torch.cat([edges[1].unsqueeze(1), selected_edges], dim=1).long()

        if type(nodes) == np.ndarray:
            nodes_torch = torch.from_numpy(nodes).to(self.args.device).long()
        else:
            nodes_torch = nodes.to(self.args.device).long()
        self_loop_edges = torch.cat(
            [
                nodes_torch[:, 0].unsqueeze(1),
                nodes_torch[:, 1].unsqueeze(1),
                (self.relation_num - 1)
                * torch.ones((len(nodes), 1)).to(self.args.device).long(),
                nodes_torch[:, 1].unsqueeze(1),
            ],
            1,
        )
        sampled_edges = torch.cat([sampled_edges, self_loop_edges], 0)

        _, head_index = torch.unique(
            sampled_edges[:, [0, 1]], dim=0, sorted=False, return_inverse=True
        )
        tail_nodes, tail_index = torch.unique(
            sampled_edges[:, [0, 3]], dim=0, sorted=False, return_inverse=True
        )

        sampled_edges = torch.cat(
            [sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1
        )

        mask = sampled_edges[:, 2] == self.relation_num - 1  # self-loop edges
        _, old_idx = head_index[mask].sort()
        old_nodes_new_idx = tail_index[mask][old_idx]

        # sampled_edges: batch_id, sub, rel, obj, head_idx, tail_idx

        return tail_nodes, sampled_edges, old_nodes_new_idx
