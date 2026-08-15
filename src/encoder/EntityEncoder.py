# ruff: noqa: N999
"""Relation- and entity-level encoders used by the IJCAI model."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_add


class GNNLayer(torch.nn.Module):
    """One attention-based message-passing layer over entity triples."""

    def __init__(self, in_dim, out_dim, attn_dim, act=lambda x: x):
        super().__init__()
        self.in_dim = in_dim

        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        # Kept in the module layout so published checkpoints remain loadable.
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(2 * attn_dim, 1)
        self.W_h = nn.Linear(in_dim, out_dim, bias=False)

        if act is not None:
            self.activation = act
        else:
            self.activation = lambda x: x

    def forward(self, hidden, edges, n_node, rel_embeddings, loader):
        """
        Forward pass for entity graph encoding

        Args:
            hidden: hidden states of nodes
            edges: edge information
            n_node: number of nodes
            rel_embeddings: relation embeddings
            loader: data loader
        """
        rel_embeddings = rel_embeddings.view(-1, self.in_dim)
        hidden_proj = self.Ws_attn(hidden)  # [N_node, attn_dim]
        rel_proj_wr = self.Wr_attn(rel_embeddings)  # [N_rel_total, attn_dim]

        batch_idx = edges[:, 0]
        sub = edges[:, 4]
        rel = edges[:, 2]
        obj = edges[:, 5]

        relation_count = loader.relation_num
        rel_offset = rel + batch_idx * relation_count
        hs = hidden.index_select(0, sub)
        hr = rel_embeddings.index_select(0, rel_offset)

        hs_attn = hidden_proj.index_select(0, sub)
        hr_attn = rel_proj_wr.index_select(0, rel_offset)

        # 4. In-place addition to reduce peak memory
        alpha_input = torch.cat([hr_attn, hs_attn], dim=-1)
        # alpha_input.add_(h_qr)

        # 5. Activation
        alpha = self.w_alpha(
            F.leaky_relu(alpha_input)
        )  # Changed from rrelu to leaky_relu for stability
        alpha = torch.sigmoid(alpha)

        # Compute messages
        message = hs + hr
        # Clamp obj indices to prevent out of bounds
        hidden_agg = scatter_add(alpha * message, index=obj, dim=0, dim_size=n_node)

        hidden = self.W_h(hidden_agg)
        hidden = self.activation(hidden)

        return hidden


class RelationGNNLayer(torch.nn.Module):
    """One attention-based message-passing layer over relation graphs."""

    def __init__(self, in_dim, out_dim, act=lambda x: x):
        super().__init__()
        self.out_dim = out_dim

        self.W_r = nn.Linear(in_dim, out_dim, bias=False)
        self.W_msg = nn.Linear(in_dim, out_dim, bias=False)
        self.W_alpha = nn.Linear(in_dim * 2, 1, bias=False)
        self.prototype_embeddings = nn.Embedding(5, in_dim)

        if act is not None:
            self.activation = act
        else:
            self.activation = lambda x: x

    def forward(self, n, rel_embeddings, edge_index, edge_type):
        """
        Forward pass for relation graph encoding

        Args:
            rel_embeddings: [num_relations, dim] - relation embeddings
            edge_index: [2, num_edges] - edge connections in relation graph
            edge_type: [num_edges] - edge types (0-4)
        """
        # Convert numpy arrays to tensors if needed
        if isinstance(edge_index, np.ndarray):
            edge_index = torch.from_numpy(edge_index)
        if isinstance(edge_type, np.ndarray):
            edge_type = torch.from_numpy(edge_type)

        # Move tensors to the same device as rel_embeddings and ensure correct dtypes
        edge_index = edge_index.to(rel_embeddings.device).long()
        edge_type = edge_type.to(rel_embeddings.device).long()

        num_relations = rel_embeddings.size(1)
        _, _, embedding_dim = rel_embeddings.shape

        sub_idx = edge_index[:, :, 0]  # [128, 1668]
        obj_idx = edge_index[:, :, 1]  # [128, 1668]

        sub_embeddings = torch.gather(
            rel_embeddings,
            dim=1,
            index=sub_idx.unsqueeze(-1).expand(-1, -1, embedding_dim),
        )  # [128, 1668, 32]

        obj_embeddings = torch.gather(
            rel_embeddings,
            dim=1,
            index=obj_idx.unsqueeze(-1).expand(-1, -1, embedding_dim),
        )  # [128, 1668, 32]
        proto_emb = self.prototype_embeddings(edge_type)

        sub_embeddings = sub_embeddings + proto_emb
        alpha_input = torch.cat(
            [sub_embeddings, obj_embeddings], dim=-1
        )  # [num_edges, 2*dim]
        alpha = torch.sigmoid(self.W_alpha(alpha_input))  # [num_edges, 1]

        messages = self.W_msg(sub_embeddings)  # [num_edges, out_dim]
        weighted_messages = alpha * messages  # [num_edges, out_dim]

        weighted_messages = weighted_messages.view(-1, self.out_dim)
        obj_idx = (
            torch.arange(obj_idx.size(0), device=obj_idx.device).unsqueeze(-1)
            * num_relations
            + obj_idx
        )
        obj_idx = obj_idx.view(-1)

        aggregated = scatter_add(
            weighted_messages, index=obj_idx, dim=0, dim_size=n * num_relations
        )
        aggregated = aggregated.view(n, num_relations, self.out_dim)

        updated_rel_emb = self.W_r(rel_embeddings) + aggregated
        updated_rel_emb = self.activation(updated_rel_emb)

        updated_rel_emb = updated_rel_emb.view(n, num_relations, self.out_dim)

        return updated_rel_emb


class EntityEncoder(torch.nn.Module):
    """Encode relation-aware entity representations and alignment scores."""

    def __init__(self, params):
        super().__init__()
        self.args = params
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        acts = {"relu": nn.ReLU(), "tanh": torch.tanh, "idd": lambda x: x}
        act = acts[params.act]

        # Relation encoder for relation graph
        self.relation_gnn_layers = nn.ModuleList(
            [
                RelationGNNLayer(self.hidden_dim, self.hidden_dim, act=act)
                for _ in range(self.args.n_relation_encoder_layer)
            ]
        )

        # Entity encoder layers
        self.entity_gnn_layers = nn.ModuleList(
            [
                GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, act=act)
                for _ in range(self.n_layer)
            ]
        )

        self.layer_norms_rel = nn.ModuleList(
            [
                nn.LayerNorm(self.hidden_dim)
                for _ in range(self.args.n_relation_encoder_layer)
            ]
        )
        self.layer_norms_ent = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim) for _ in range(self.n_layer)]
        )

        self.hidden_map = nn.Linear(1024, self.hidden_dim)
        self.hidden_map2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.dropout = nn.Dropout(params.dropout)
        self.W_final = nn.Linear(self.hidden_dim * 2, 1, bias=False)
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)

        # Retained for compatibility with checkpoints produced by earlier runs.
        self.similarity_layer = nn.Linear(self.hidden_dim, 1, bias=False)

        self.loop_and_align_embeddings = nn.Embedding(2, self.hidden_dim)

    def encode_relations(self, n, sub_neighbor_rels, edge_index, edge_type, loader):
        """
        Encode relations using relation graph

        Args:
            edge_index: [2, num_edges] - edge connections in relation graph
            edge_type: [num_edges] - edge types (0-4)
            loader: data loader containing relation information
        """
        if type(edge_index) == np.ndarray:
            edge_index = torch.from_numpy(edge_index).to(self.args.device)
            edge_type = torch.from_numpy(edge_type).to(self.args.device)

        rel_embeddings = torch.zeros(n, loader.relation_num, self.hidden_dim).to(
            self.args.device
        )
        rel_embeddings[:, -2:, :] = self.loop_and_align_embeddings.weight
        batch_idx = sub_neighbor_rels[:, 0].long()
        rel_idx = sub_neighbor_rels[:, 1].long()

        # 要写入的向量，[580, 32]
        ones_vec = torch.ones(
            len(sub_neighbor_rels),
            rel_embeddings.size(-1),
            device=rel_embeddings.device,
            dtype=rel_embeddings.dtype,
        )

        # 修改 rel_embeddings 中对应位置的值
        rel_embeddings[batch_idx, rel_idx] = ones_vec

        edge_index = edge_index.unsqueeze(0).expand(n, -1, -1)
        edge_type = edge_type.unsqueeze(0).expand(n, -1)
        for i in range(self.args.n_relation_encoder_layer):
            rel_embeddings = self.relation_gnn_layers[i](
                n, rel_embeddings, edge_index, edge_type
            )
            rel_embeddings = self.layer_norms_rel[i](rel_embeddings)
            rel_embeddings = self.dropout(rel_embeddings)

        return rel_embeddings

    def encode_entities_with_alignment(self, subs, loader, objs, mode="train"):
        """
        Encode entities using GNN with alignment information

        Args:
            subs: subject entities
            loader: data loader
            objs: aligned target entities

        Returns:
            entity representations and relation embeddings
        """
        target_kg = loader.kg2
        n = len(subs)

        q_sub = torch.from_numpy(subs).to(self.args.device).long()
        nodes = torch.cat(
            [torch.arange(n).unsqueeze(1).to(self.args.device), q_sub.unsqueeze(1)], 1
        )
        edge_index, edge_type = loader.get_relation_index(mode=mode)
        if self.args.use_anchor_conditioned_initialize.lower() == "true":
            _, edges, _ = loader.get_neighbors(nodes.data.cpu().numpy())
            sub_neighbor_rels = edges[:, [0, 2]]
        else:
            sub_neighbor_rels = torch.cat(
                [
                    torch.arange(n).unsqueeze(1),
                    torch.zeros(n).unsqueeze(1) + loader.relation_num - 2,
                ],
                dim=-1,
            ).to(self.args.device)
        rel_embeddings_full = self.encode_relations(
            n, sub_neighbor_rels, edge_index, edge_type, loader
        )

        if self.args.use_two_tower.lower() == "true":
            source_nei_align, target_nei_align = loader.get_aligned_neighbors_in_kg(
                subs, nodes, int(self.args.K)
            )
            target_nei_align = torch.LongTensor(target_nei_align).to(self.args.device)
            source_nei_align = torch.LongTensor(source_nei_align).to(self.args.device)
            nodes = torch.cat([source_nei_align, target_nei_align])
        else:
            nodes = torch.cat(
                [torch.arange(n).unsqueeze(1).to(self.args.device), q_sub.unsqueeze(1)],
                1,
            )

        if self.args.use_text_embeddings.lower() == "true":
            ent_embeddings = self.hidden_map(loader.ent_embeddings).clone()
            hidden_ = ent_embeddings[nodes[:, 1]]
        else:
            hidden_ = torch.ones(nodes.size(0), self.hidden_dim).to(self.args.device)

        hidden = torch.zeros(n, loader.entity_num, self.hidden_dim).to(self.args.device)
        hidden[nodes[:, 0], nodes[:, 1]] = hidden_
        hidden = hidden.view(-1, self.hidden_dim)
        triples_ = loader.graph
        subs_tensor = torch.from_numpy(subs).to(self.args.device)
        objs_tensor = torch.from_numpy(objs).to(self.args.device)
        subs_merge_tensor = torch.cat([subs_tensor, objs_tensor], dim=0)
        objs_merge_tensor = torch.cat([objs_tensor, subs_tensor], dim=0)
        if self.args.use_two_tower.lower() != "true":
            mask_k = triples_[:, 1] == loader.relation_num - 2

            # Remove the queried alignment edge from the unified graph.
            idx = mask_k.nonzero(as_tuple=True)[0]
            A_k = triples_[idx]  # [N_k, 3]

            drop_k = torch.isin(A_k[:, 0], subs_merge_tensor) & torch.isin(
                A_k[:, 2], objs_merge_tensor
            )

            keep = torch.ones(
                triples_.size(0), dtype=torch.bool, device=triples_.device
            )
            keep[idx] = ~drop_k

            triples = triples_[keep]
        else:
            triples = triples_

        batch_ids = (
            torch.arange(n)
            .to(self.args.device)
            .unsqueeze(1)
            .expand(n, triples.size(0))
            .reshape(-1, 1)
        )

        triple_edges = triples.unsqueeze(0).expand(n, triples.size(0), 3).reshape(-1, 3)
        offset = (
            torch.arange(n).to(self.args.device).unsqueeze(1).expand(n, triples.size(0))
            * loader.entity_num
        )
        edges = torch.cat(
            [
                batch_ids,
                triple_edges,
                triple_edges[:, 0].unsqueeze(1) + offset.view(-1, 1),
                triple_edges[:, 2].unsqueeze(1) + offset.view(-1, 1),
            ],
            dim=-1,
        )

        for i in range(self.n_layer):
            hidden_ = hidden
            hidden = self.entity_gnn_layers[i](
                hidden,
                edges,
                n * loader.entity_num,
                rel_embeddings_full,
                loader,
            )

            hidden = self.layer_norms_ent[i](hidden)
            hidden = hidden_ + self.dropout(hidden)
            hidden = hidden.squeeze(0)

        if self.args.use_text_embeddings.lower() == "true":
            hidden_ = ent_embeddings
            hidden_ = self.hidden_map2(hidden_).unsqueeze(0)
            hidden = hidden.reshape(n, loader.entity_num, self.hidden_dim) + hidden_

        batch_id = torch.arange(n).to(self.args.device)
        subs_torch = torch.from_numpy(subs).to(self.args.device).long()
        hidden = hidden.reshape(-1, self.hidden_dim)
        if self.args.use_interact:
            subs_emb_nodes = hidden[batch_id * loader.entity_num + subs_torch]
            subs_emb_nodes = (
                subs_emb_nodes.unsqueeze(1)
                .expand(n, loader.entity_num, self.hidden_dim)
                .reshape(-1, self.hidden_dim)
            )
            hidden = torch.cat([torch.abs(subs_emb_nodes - hidden), hidden], dim=1)
        else:
            hidden = torch.cat([hidden, hidden], dim=1)

        scores = self.W_final(hidden).squeeze(-1)
        scores_all = scores.reshape(n, -1)
        target_kg_entities = target_kg.entity_list
        source_kg = loader.kg1
        source_kg_entities = source_kg.entity_list

        return scores_all, source_kg_entities, target_kg_entities, hidden
