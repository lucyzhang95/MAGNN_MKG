import numpy as np
import torch
import torch.nn as nn

from .base_MAGNN import MAGNN_ctr_ntype_specific


# for link prediction task
class MAGNN_lp_layer(nn.Module):
    def __init__(
        self,
        num_metapaths_list,
        num_edge_type,
        etypes_lists,
        in_dim,
        out_dim,
        num_heads,
        attn_vec_dim,
        rnn_type="gru",
        attn_drop=0.5,
    ):
        super(MAGNN_lp_layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        # etype-specific parameters
        # num_edge_type // 2, relationships like edge and reverse edge share a common parameter
        # they are symmetric
        r_vec = None
        if rnn_type == "TransE0":
            r_vec = nn.Parameter(
                torch.empty(size=(num_edge_type // 2, in_dim))
            )
        elif rnn_type == "TransE1":
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type, in_dim)))
        elif rnn_type == "RotatE0":
            r_vec = nn.Parameter(
                torch.empty(size=(num_edge_type // 2, in_dim // 2, 2))
            )
        elif rnn_type == "RotatE1":
            r_vec = nn.Parameter(
                torch.empty(size=(num_edge_type, in_dim // 2, 2))
            )
        if r_vec is not None:
            nn.init.xavier_normal_(r_vec.data, gain=1.414)

        # ctr_ntype-specific layers
        self.microbe_layer = MAGNN_ctr_ntype_specific(
            num_metapaths_list[0],
            etypes_lists[0],
            in_dim,
            num_heads,
            attn_vec_dim,
            rnn_type,
            r_vec,
            attn_drop,
            use_minibatch=True,
        )
        self.disease_layer = MAGNN_ctr_ntype_specific(
            num_metapaths_list[1],
            etypes_lists[1],
            in_dim,
            num_heads,
            attn_vec_dim,
            rnn_type,
            r_vec,
            attn_drop,
            use_minibatch=True,
        )
        self.metabolite_layer = MAGNN_ctr_ntype_specific(
            num_metapaths_list[2],
            etypes_lists[2],
            in_dim,
            num_heads,
            attn_vec_dim,
            rnn_type,
            r_vec,
            attn_drop,
            use_minibatch=True,
        )

        # note that the actual input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        self.fc_microbe = nn.Linear(in_dim * num_heads, out_dim, bias=True)
        self.fc_disease = nn.Linear(in_dim * num_heads, out_dim, bias=True)
        self.fc_metabolite = nn.Linear(in_dim * num_heads, out_dim, bias=True)

        nn.init.xavier_normal_(self.fc_microbe.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc_disease.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc_metabolite.weight, gain=1.414)

    def forward(self, inputs):
        (
            g_lists,
            features,
            type_mask,
            edge_metapath_indices_lists,
            target_idx_lists,
        ) = inputs

        # ctr_ntype-specific layers
        h_microbe = self.microbe_layer(
            (
                g_lists[0],
                features,
                type_mask,
                edge_metapath_indices_lists[0],
                target_idx_lists[0],
            )
        )
        h_disease = self.disease_layer(
            (
                g_lists[1],
                features,
                type_mask,
                edge_metapath_indices_lists[1],
                target_idx_lists[1],
            )
        )
        h_metabolite = self.metabolite_layer(
            (
                g_lists[2],
                features,
                type_mask,
                edge_metapath_indices_lists[2],
                target_idx_lists[2],
            )
        )

        logits_microbe = self.fc_microbe(h_microbe)
        logits_disease = self.fc_disease(h_disease)
        logits_metabolite = self.fc_metabolite(h_metabolite)

        return (
            [logits_microbe, logits_disease, logits_metabolite],
            [h_microbe, h_disease, h_metabolite],
        )


class MAGNN_lp(nn.Module):
    def __init__(
        self,
        num_metapaths_list,  # [4, 4, 4] for 3 node types
        num_edge_type,
        etypes_lists,
        feats_dim_list,  # Input feature dimensions per node type
        hidden_dim,
        out_dim,
        num_heads,
        attn_vec_dim,
        rnn_type="gru",
        dropout_rate=0.5,
    ):
        super(MAGNN_lp, self).__init__()
        self.hidden_dim = hidden_dim

        # ntype-specific transformation layer for microbe, disease, metabolite
        self.fc_list = nn.ModuleList(
            [
                nn.Linear(feats_dim, hidden_dim, bias=True)
                for feats_dim in feats_dim_list
            ]
        )

        # feature dropout after transformation
        # if dropout_rate > 0:
        #     self.feat_drop = nn.Dropout(dropout_rate)
        # else:
        #     self.feat_drop = lambda x: x
        self.feat_drop = (
            nn.Dropout(dropout_rate) if dropout_rate > 0 else lambda x: x
        )

        # initialization of fc layers
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        # MAGNN_lp layers
        self.layer1 = MAGNN_lp_layer(
            num_metapaths_list,
            num_edge_type,
            etypes_lists,
            hidden_dim,
            out_dim,
            num_heads,
            attn_vec_dim,
            rnn_type,
            attn_drop=dropout_rate,
        )

    def forward(self, inputs):
        (
            g_lists,
            features_list,
            type_mask,
            edge_metapath_indices_lists,
            target_idx_lists,
        ) = inputs

        # ntype-specific transformation
        transformed_features = torch.zeros(
            type_mask.shape[0], self.hidden_dim, device=features_list[0].device
        )
        for i, fc in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i)[0]
            transformed_features[node_indices] = fc(features_list[i])
        transformed_features = self.feat_drop(transformed_features)

        # hidden layers: process microbe, disease, and metabolite
        (
            [logits_microbe, logits_disease, logits_metabolite],
            [h_microbe, h_disease, h_metabolite],
        ) = self.layer1(
            (
                g_lists,
                transformed_features,
                type_mask,
                edge_metapath_indices_lists,
                target_idx_lists,
            )
        )

        return (
            [logits_microbe, logits_disease, logits_metabolite],
            [h_microbe, h_disease, h_metabolite],
        )
