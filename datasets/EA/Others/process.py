import os
import numpy as np
import pickle
import json


class SRPRSDatasetPreprocessor:
    def __init__(self, dataset_path, output_path):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.entities1 = set()
        self.entities2 = set()
        self.relations1 = set()
        self.relations2 = set()
        self.triples1 = []
        self.triples2 = []
        self.alignment_pairs = []  # All alignment pairs (supervised + reference)
        self.train_pairs = []
        self.valid_pairs = []
        self.test_pairs = []

    def load_dataset(self):
        """加载SRPRS数据集"""
        # SRPRS数据集的所有文件都在mapping/0_3目录下
        mapping_dir = self.dataset_path
        if not os.path.exists(mapping_dir):
            print(f"Warning: directory not found in {self.dataset_path}")
            return

        # 加载三元组
        self._load_triples('triples_1', 1)
        self._load_triples('triples_2', 2)

        # 加载实体对齐 (从sup_ent_ids和ref_ent_ids组合)
        if self._load_alignments_from_split_files():
            # 加载训练集、验证集和测试集
            self._load_fold_data()
        else:
            self._split_fold_data()

    def _load_triples(self, filename, kg_id):
        """加载三元组数据"""
        filepath = os.path.join(self.dataset_path, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found")
            return

        triples = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    subj, rel, obj = parts[0], parts[1], parts[2]
                    triples.append((subj, rel, obj))

                    # 收集实体和关系
                    if kg_id == 1:
                        self.entities1.add(subj)
                        self.entities1.add(obj)
                        self.relations1.add(rel)
                    else:
                        self.entities2.add(subj)
                        self.entities2.add(obj)
                        self.relations2.add(rel)
                elif line.strip():
                    print(f"Warning: Invalid triple format at line {line_num} in {filename}: {line.strip()}")

        if kg_id == 1:
            self.triples1 = triples
        else:
            self.triples2 = triples

        print(f"Loaded {len(triples)} triples from KG{kg_id}")

    def _load_alignments_from_split_files(self):
        """从分割的文件加载实体对齐数据 (sup_ent_ids + ref_ent_ids)"""
        mapping_dir = self.dataset_path
        # 加载sup_ent_ids作为部分对齐数据
        sup_file = os.path.join(mapping_dir, 'sup_ent_ids')
        if os.path.exists(sup_file):
            if os.path.exists(sup_file):
                with open(sup_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            self.alignment_pairs.append((parts[0], parts[1]))
                print(f"Loaded {len(self.alignment_pairs)} alignment pairs from sup_ent_ids")

            # 加载ref_ent_ids作为剩余对齐数据
            ref_file = os.path.join(mapping_dir, 'ref_ent_ids')
            if os.path.exists(ref_file):
                with open(ref_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            self.alignment_pairs.append((parts[0], parts[1]))
                print(f"Loaded additional alignment pairs from ref_ent_ids. Total: {len(self.alignment_pairs)}")
            return True
        else:
            align_file = os.path.join(mapping_dir, 'ent_ILLs')
            with open(align_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        self.alignment_pairs.append((parts[0], parts[1]))
            return False

    def _load_fold_data(self):
        """加载SRPRS的训练集、验证集和测试集"""
        mapping_dir = self.dataset_path
        if not os.path.exists(mapping_dir):
            print("Warning: mapping/0_3 directory not found")
            return

        # 加载训练集 (sup_ent_ids)
        train_file = os.path.join(mapping_dir, 'sup_ent_ids')
        if os.path.exists(train_file):
            with open(train_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        self.train_pairs.append((parts[0], parts[1]))
            print(f"Loaded {len(self.train_pairs)} training pairs")

        # 加载验证集和测试集 (ref_ent_ids)
        ref_file = os.path.join(mapping_dir, 'ref_ent_ids')
        if os.path.exists(ref_file):
            ref_pairs = []
            with open(ref_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        ref_pairs.append((parts[0], parts[1]))

            # 前10%为验证集，剩余为测试集
            split_idx = int(len(ref_pairs) * 0.1)
            self.valid_pairs = ref_pairs[:split_idx]
            self.test_pairs = ref_pairs[split_idx:]

            print(f"Loaded {len(self.valid_pairs)} validation pairs")
            print(f"Loaded {len(self.test_pairs)} test pairs")

    def _split_fold_data(self):
        import random
        """将数据集划分为训练集、验证集和测试集"""
        train_rate = 0.3
        valid_rate = 0.1
        test_rate = 0.6
        train_pairs = []
        valid_pairs = []
        test_pairs = []
        # 打乱顺序，然后严格按照比例划分
        idxs = list(range(len(self.alignment_pairs)))
        random.shuffle(idxs)
        for order, idx in enumerate(idxs):
            if order < train_rate * len(idxs):
                train_pairs.append(self.alignment_pairs[idx])
            elif order < (train_rate + valid_rate) * len(idxs):
                valid_pairs.append(self.alignment_pairs[idx])
            else:
                test_pairs.append(self.alignment_pairs[idx])
        self.train_pairs = train_pairs
        self.valid_pairs = valid_pairs
        self.test_pairs = test_pairs


    def create_id_mappings(self):
        """创建ID映射，处理跨语言关系共享"""
        # 实体映射：确保两个KG的ID不重叠
        self.entity2id_1 = {ent: idx for idx, ent in enumerate(sorted(self.entities1))}
        # 第二个KG的ID从第一个KG的最大ID+1开始
        max_id_1 = len(self.entities1) - 1 if self.entities1 else 0
        self.entity2id_2 = {ent: idx + max_id_1 + 1 for idx, ent in enumerate(sorted(self.entities2))}

        # 关系映射：处理跨语言关系共享
        # 首先识别共同关系
        common_relations = self.relations1.intersection(self.relations2)
        unique_relations_1 = self.relations1.difference(self.relations2)
        unique_relations_2 = self.relations2.difference(self.relations1)

        # 为共同关系分配相同ID
        self.relation2id_common = {rel: idx for idx, rel in enumerate(sorted(common_relations))}
        max_common_id = len(common_relations) - 1 if common_relations else -1

        # 为各自独有的关系分配ID
        self.relation2id_1 = {}
        self.relation2id_2 = {}

        # 共同关系使用相同ID
        for rel in common_relations:
            self.relation2id_1[rel] = self.relation2id_common[rel]
            self.relation2id_2[rel] = self.relation2id_common[rel]

        # 独有关系分配新ID
        for idx, rel in enumerate(sorted(unique_relations_1)):
            self.relation2id_1[rel] = max_common_id + 1 + idx

        for idx, rel in enumerate(sorted(unique_relations_2)):
            self.relation2id_2[rel] = max_common_id + 1 + len(unique_relations_1) + idx

        print(f"Created ID mappings with {len(common_relations)} common relations")

    def convert_triples_to_ids(self):
        """将三元组转换为ID格式"""
        # 转换KG1三元组
        id_triples1 = []
        for subj, rel, obj in self.triples1:
            if subj in self.entity2id_1 and obj in self.entity2id_1 and rel in self.relation2id_1:
                id_triples1.append((
                    self.entity2id_1[subj],
                    self.relation2id_1[rel],
                    self.entity2id_1[obj]
                ))

        # 转换KG2三元组
        id_triples2 = []
        for subj, rel, obj in self.triples2:
            if subj in self.entity2id_2 and obj in self.entity2id_2 and rel in self.relation2id_2:
                id_triples2.append((
                    self.entity2id_2[subj],
                    self.relation2id_2[rel],
                    self.entity2id_2[obj]
                ))

        self.id_triples1 = np.array(id_triples1, dtype=np.int32) if id_triples1 else np.empty((0, 3), dtype=np.int32)
        self.id_triples2 = np.array(id_triples2, dtype=np.int32) if id_triples2 else np.empty((0, 3), dtype=np.int32)

    def save_processed_data(self):
        """保存处理后的数据，与OpenEA格式保持一致"""
        os.makedirs(self.output_path, exist_ok=True)

        # 保存ID映射（pickle格式）
        mappings = {
            'entity2id_1': self.entity2id_1,
            'entity2id_2': self.entity2id_2,
            'relation2id_1': self.relation2id_1,
            'relation2id_2': self.relation2id_2,
            'relation2id_common': getattr(self, 'relation2id_common', {})
        }

        with open(os.path.join(self.output_path, 'id_mappings.pkl'), 'wb') as f:
            pickle.dump(mappings, f)

        # 保存三元组（numpy格式）
        np.save(os.path.join(self.output_path, 'triples_id_1.npy'), self.id_triples1)
        np.save(os.path.join(self.output_path, 'triples_id_2.npy'), self.id_triples2)

        # 保存完整的对齐对
        alignment_ids = []
        for ent1, ent2 in self.alignment_pairs:
            if ent1 in self.entity2id_1 and ent2 in self.entity2id_2:
                alignment_ids.append([self.entity2id_1[ent1], self.entity2id_2[ent2]])

        alignment_array = np.array(alignment_ids, dtype=np.int32) if alignment_ids else np.empty((0, 2), dtype=np.int32)
        np.save(os.path.join(self.output_path, 'alignment_pairs.npy'), alignment_array)

        # 保存训练集、验证集和测试集
        fold_dir = os.path.join(self.output_path, '5fold', '1')  # 模拟5fold结构中的第1组
        os.makedirs(fold_dir, exist_ok=True)

        # 保存训练集
        train_ids = []
        for ent1, ent2 in self.train_pairs:
            if ent1 in self.entity2id_1 and ent2 in self.entity2id_2:
                train_ids.append([self.entity2id_1[ent1], self.entity2id_2[ent2]])
        if train_ids:
            train_array = np.array(train_ids, dtype=np.int32)
            np.save(os.path.join(fold_dir, 'train_links.npy'), train_array)

        # 保存验证集
        valid_ids = []
        for ent1, ent2 in self.valid_pairs:
            if ent1 in self.entity2id_1 and ent2 in self.entity2id_2:
                valid_ids.append([self.entity2id_1[ent1], self.entity2id_2[ent2]])
        if valid_ids:
            valid_array = np.array(valid_ids, dtype=np.int32)
            np.save(os.path.join(fold_dir, 'valid_links.npy'), valid_array)

        # 保存测试集
        test_ids = []
        for ent1, ent2 in self.test_pairs:
            if ent1 in self.entity2id_1 and ent2 in self.entity2id_2:
                test_ids.append([self.entity2id_1[ent1], self.entity2id_2[ent2]])
        if test_ids:
            test_array = np.array(test_ids, dtype=np.int32)
            np.save(os.path.join(fold_dir, 'test_links.npy'), test_array)

        # 保存统计数据
        stats = {
            'kg1_entities': len(self.entities1),
            'kg2_entities': len(self.entities2),
            'kg1_relations': len(self.relations1),
            'kg2_relations': len(self.relations2),
            'common_relations': len(getattr(self, 'relation2id_common', {})),
            'kg1_triples': len(self.id_triples1),
            'kg2_triples': len(self.id_triples2),
            'alignment_pairs': len(alignment_array),
            'train_pairs': len(train_ids),
            'valid_pairs': len(valid_ids),
            'test_pairs': len(test_ids),
            'folds': 1  # SRPRS只有一个fold
        }

        with open(os.path.join(self.output_path, 'statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2)

        print("Data saved successfully:")
        print(f"  - ID mappings: id_mappings.pkl")
        print(f"  - KG1 triples: triples_id_1.npy ({len(self.id_triples1)} triples)")
        print(f"  - KG2 triples: triples_id_2.npy ({len(self.id_triples2)} triples)")
        print(f"  - Alignment pairs: alignment_pairs.npy ({len(alignment_array)} pairs)")
        print(f"  - Train pairs: {len(train_ids)}")
        print(f"  - Valid pairs: {len(valid_ids)}")
        print(f"  - Test pairs: {len(test_ids)}")
        print(f"  - Data saved in 5fold/1/ directory")


def main():
    # Process SRPRS datasets
    datasets = ['D_W_15K', 'D_Y_15K',
                'D_W_100K', 'D_Y_100K',
                'JA_EN', 'FR_EN', 'ZH_EN']
    base_path = ""  # 修正base_path
    output_base = "../../processed_data/Others"

    for dataset in datasets:
        print(f'==================== {dataset}')
        dataset_path = os.path.join(base_path, dataset)
        output_path = os.path.join(output_base, dataset)

        if os.path.exists(dataset_path):
            preprocessor = SRPRSDatasetPreprocessor(dataset_path, output_path)
            preprocessor.load_dataset()
            preprocessor.create_id_mappings()
            preprocessor.convert_triples_to_ids()
            preprocessor.save_processed_data()
        else:
            print(f"Dataset path not found: {dataset_path}")


if __name__ == "__main__":
    main()
