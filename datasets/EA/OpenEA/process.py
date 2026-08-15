import os
import numpy as np
import pickle
import json
from collections import defaultdict

'''
EN_DE_15K_V1_processed/
├── id_mappings.pkl          # ID映射（pickle格式）
├── triples_id_1.npy         # KG1三元组（numpy格式）
├── triples_id_2.npy         # KG2三元组（numpy格式）
├── alignment_pairs.npy      # 所有对齐对（numpy格式）
├── statistics.json          # 统计信息
└── 5fold/                   # 5-fold数据
    ├── 1/
    │   ├── train_links.npy  # 第1组训练集
    │   ├── valid_links.npy  # 第1组验证集
    │   └── test_links.npy   # 第1组测试集
    ├── 2/
    │   ├── train_links.npy  # 第2组训练集
    │   ├── valid_links.npy  # 第2组验证集
    │   └── test_links.npy   # 第2组测试集
    ├── 3/
    │   ├── train_links.npy  # 第3组训练集
    │   ├── valid_links.npy  # 第3组验证集
    │   └── test_links.npy   # 第3组测试集
    ├── 4/
    │   ├── train_links.npy  # 第4组训练集
    │   ├── valid_links.npy  # 第4组验证集
    │   └── test_links.npy   # 第4组测试集
    └── 5/
        ├── train_links.npy  # 第5组训练集
        ├── valid_links.npy  # 第5组验证集
        └── test_links.npy   # 第5组测试集
'''

class EntityAlignmentDatasetPreprocessor:
    def __init__(self, dataset_path, output_path):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.entities1 = set()
        self.entities2 = set()
        self.relations1 = set()
        self.relations2 = set()
        self.triples1 = []
        self.triples2 = []
        self.alignment_pairs = []
        self.fold_data = {}

    def load_dataset(self):
        """加载数据集"""
        # 加载三元组 - 尝试多种可能的文件名
        self._load_triples_with_fallback('rel_triples_1', 1)
        self._load_triples_with_fallback('rel_triples_2', 2)

        # 加载实体对齐 - 尝试多种可能的文件名
        self._load_alignments_with_fallback()

        # 加载5-fold数据
        self._load_fold_data()

    def _load_triples_with_fallback(self, base_filename, kg_id):
        """加载三元组数据，支持多种文件名"""
        possible_filenames = [base_filename, f"{base_filename}.txt", f"triples_{kg_id}", f"triples_{kg_id}.txt"]

        for filename in possible_filenames:
            filepath = os.path.join(self.dataset_path, filename)
            if os.path.exists(filepath):
                self._load_triples(filename, kg_id)
                return

        print(f"Warning: No triple file found for KG{kg_id} (tried: {possible_filenames})")

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
                elif line.strip():  # 忽略空行但警告格式错误
                    print(f"Warning: Invalid triple format at line {line_num} in {filename}: {line.strip()}")

        if kg_id == 1:
            self.triples1 = triples
        else:
            self.triples2 = triples

        print(f"Loaded {len(triples)} triples from KG{kg_id}")

    def _load_alignments_with_fallback(self):
        """加载实体对齐数据，支持多种文件名"""
        possible_filenames = ['ent_links', 'ent_links.txt', 'entity_links', 'entity_links.txt']

        for filename in possible_filenames:
            filepath = os.path.join(self.dataset_path, filename)
            if os.path.exists(filepath):
                self._load_alignments(filename)
                return

        print(f"Warning: No entity alignment file found (tried: {possible_filenames})")

    def _load_alignments(self, filename='ent_links'):
        """加载实体对齐数据"""
        alignment_file = os.path.join(self.dataset_path, filename)
        if not os.path.exists(alignment_file):
            print("Warning: ent_links not found")
            return

        with open(alignment_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    ent1, ent2 = parts[0], parts[1]
                    self.alignment_pairs.append((ent1, ent2))
                elif line.strip():  # 忽略空行但警告格式错误
                    print(f"Warning: Invalid alignment format at line {line_num}: {line.strip()}")

        print(f"Loaded {len(self.alignment_pairs)} entity alignment pairs")

    def _load_fold_data(self):
        """加载5-fold分组数据"""
        fold_dir = os.path.join(self.dataset_path, '721_5fold')
        if not os.path.exists(fold_dir):
            print("Warning: 721_5fold directory not found")
            return

        for i in range(1, 6):  # 5 folds
            fold_path = os.path.join(fold_dir, str(i))
            if not os.path.exists(fold_path):
                continue

            self.fold_data[i] = {}

            # 加载训练集
            train_file = os.path.join(fold_path, 'train_links')
            if os.path.exists(train_file):
                train_pairs = []
                with open(train_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            train_pairs.append((parts[0], parts[1]))
                        elif line.strip():  # 忽略空行但警告格式错误
                            print(f"Warning: Invalid train link format at line {line_num} in fold {i}: {line.strip()}")
                if train_pairs:
                    self.fold_data[i]['train'] = train_pairs

            # 加载验证集
            valid_file = os.path.join(fold_path, 'valid_links')
            if os.path.exists(valid_file):
                valid_pairs = []
                with open(valid_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            valid_pairs.append((parts[0], parts[1]))
                        elif line.strip():  # 忽略空行但警告格式错误
                            print(f"Warning: Invalid valid link format at line {line_num} in fold {i}: {line.strip()}")
                if valid_pairs:
                    self.fold_data[i]['valid'] = valid_pairs

            # 加载测试集
            test_file = os.path.join(fold_path, 'test_links')
            if os.path.exists(test_file):
                test_pairs = []
                with open(test_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            test_pairs.append((parts[0], parts[1]))
                        elif line.strip():  # 忽略空行但警告格式错误
                            print(f"Warning: Invalid test link format at line {line_num} in fold {i}: {line.strip()}")
                if test_pairs:
                    self.fold_data[i]['test'] = test_pairs

        print(f"Loaded fold data for {len(self.fold_data)} folds")

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
        """保存处理后的数据"""
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

        # 保存5-fold数据
        if self.fold_data:
            fold_dir = os.path.join(self.output_path, '5fold')
            os.makedirs(fold_dir, exist_ok=True)

            for fold_id, fold_data in self.fold_data.items():
                fold_path = os.path.join(fold_dir, str(fold_id))
                os.makedirs(fold_path, exist_ok=True)

                # 保存训练集
                if 'train' in fold_data:
                    train_ids = []
                    for ent1, ent2 in fold_data['train']:
                        if ent1 in self.entity2id_1 and ent2 in self.entity2id_2:
                            train_ids.append([self.entity2id_1[ent1], self.entity2id_2[ent2]])
                    if train_ids:
                        train_array = np.array(train_ids, dtype=np.int32)
                        np.save(os.path.join(fold_path, 'train_links.npy'), train_array)

                # 保存验证集
                if 'valid' in fold_data:
                    valid_ids = []
                    for ent1, ent2 in fold_data['valid']:
                        if ent1 in self.entity2id_1 and ent2 in self.entity2id_2:
                            valid_ids.append([self.entity2id_1[ent1], self.entity2id_2[ent2]])
                    if valid_ids:
                        valid_array = np.array(valid_ids, dtype=np.int32)
                        np.save(os.path.join(fold_path, 'valid_links.npy'), valid_array)

                # 保存测试集
                if 'test' in fold_data:
                    test_ids = []
                    for ent1, ent2 in fold_data['test']:
                        if ent1 in self.entity2id_1 and ent2 in self.entity2id_2:
                            test_ids.append([self.entity2id_1[ent1], self.entity2id_2[ent2]])
                    if test_ids:
                        test_array = np.array(test_ids, dtype=np.int32)
                        np.save(os.path.join(fold_path, 'test_links.npy'), test_array)

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
            'folds': len(self.fold_data)
        }

        with open(os.path.join(self.output_path, 'statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2)

        print("Data saved successfully:")
        print(f"  - ID mappings: id_mappings.pkl")
        print(f"  - KG1 triples: triples_id_1.npy ({len(self.id_triples1)} triples)")
        print(f"  - KG2 triples: triples_id_2.npy ({len(self.id_triples2)} triples)")
        print(f"  - Alignment pairs: alignment_pairs.npy ({len(alignment_array)} pairs)")
        if self.fold_data:
            print(f"  - 5-fold data saved in 5fold/ directory")


def main(dataset):
    # 使用示例
    dataset_path = dataset
    output_path = "../../processed_data/OpenEA/"
    output_path = os.path.join(output_path, dataset)

    preprocessor = EntityAlignmentDatasetPreprocessor(dataset_path, output_path)
    preprocessor.load_dataset()
    preprocessor.create_id_mappings()
    preprocessor.convert_triples_to_ids()
    preprocessor.save_processed_data()


if __name__ == "__main__":
    datasets = ['D_W_15K_V1', 'D_W_15K_V2', 'D_W_100K_V1', 'D_W_100K_V2',
                'D_Y_15K_V1', 'D_Y_15K_V2', 'D_Y_100K_V1', 'D_Y_100K_V2',
                'EN_DE_15K_V1', 'EN_DE_15K_V2', 'EN_DE_100K_V1', 'EN_DE_100K_V2',
                'EN_FR_15K_V1', 'EN_FR_15K_V2', 'EN_FR_100K_V1', 'EN_FR_100K_V2',]
    for dataset in datasets:
        print('====================', dataset)
        main(dataset)
