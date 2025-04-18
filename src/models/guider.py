import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from geomloss import SamplesLoss # Keep this import

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss
from utils.utils import build_sim, compute_normalized_laplacian


class GUIDER(GeneralRecommender):
    def __init__(self, config, dataset):
        super(GUIDER, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.knn_k = config['knn_k']
        self.lambda_coeff = config['lambda_coeff']
        self.cf_model = config['cf_model']
        self.n_layers = config['n_mm_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.reg_weight = config['reg_weight'] # Keep reg_weight if needed elsewhere, otherwise can be removed
        self.build_item_graph = True
        self.mm_image_weight = config['mm_image_weight']
        self.dropout = config['dropout']
        self.degree_ratio = config['degree_ratio']
        # Use getattr for safe access with default
        self.hash_bits = getattr(config, 'hash_bits', 64)

        self.n_nodes = self.n_users + self.n_items

        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(float)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.masked_adj, self.mm_adj = None, None
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices, self.edge_values = self.edge_indices.to(self.device), self.edge_values.to(self.device)
        self.edge_full_indices = torch.arange(self.edge_values.size(0)).to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        mm_adj_file = os.path.join(dataset_path, 'mm_adj_GUIDERdsp_{}_{}.pt'.format(self.knn_k, int(10*self.mm_image_weight)))

        # Initialize hash layers only if features exist
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            self.image_hash_layer = nn.Linear(self.feat_embed_dim, self.hash_bits)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            self.text_hash_layer = nn.Linear(self.feat_embed_dim, self.hash_bits)

        # Initialize user hash layer only if features exist (needed for similarity weighting)
        if self.v_feat is not None or self.t_feat is not None:
             self.user_hash_layer = nn.Linear(self.embedding_dim, self.hash_bits)

        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)
        else:
            # Simplified MM adj calculation (assuming both features exist if either exists for weighting)
            if self.v_feat is not None and self.t_feat is not None:
                _, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                _, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj, image_adj
                torch.save(self.mm_adj, mm_adj_file)
            elif self.v_feat is not None:
                 _, self.mm_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                 torch.save(self.mm_adj, mm_adj_file)
            elif self.t_feat is not None:
                 _, self.mm_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                 torch.save(self.mm_adj, mm_adj_file)
            # Handle case where neither feature exists if necessary, though mm_adj might not be used then

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # Return indices as well if needed elsewhere, otherwise just the adj
        # return indices, self.compute_normalized_laplacian(indices, adj_size)
        return indices, self.compute_normalized_laplacian(indices, adj_size)


    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]] # Assuming symmetry for D^-1/2 A D^-1/2
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=float)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def pre_epoch_processing(self):
        if self.dropout <= .0:
            self.masked_adj = self.norm_adj
            return
        degree_len = int(self.edge_values.size(0) * (1. - self.dropout))
        # Ensure edge_values are positive for multinomial sampling if they aren't already
        positive_edge_values = F.relu(self.edge_values) + 1e-8 # Add small epsilon
        # Normalize if needed, though multinomial handles unnormalized weights
        # positive_edge_values = positive_edge_values / positive_edge_values.sum()

        # Check if positive_edge_values sum is zero before sampling
        if positive_edge_values.sum() == 0:
             # Handle the case where all edge values are zero or negative
             # Maybe select random indices or use uniform distribution
             print("Warning: Edge values sum to zero, using uniform sampling for dropout.")
             degree_idx = torch.randint(0, self.edge_values.size(0), (degree_len,), device=self.device)
        else:
            degree_idx = torch.multinomial(positive_edge_values, degree_len, replacement=False) # Use replacement=False if possible

        keep_indices = self.edge_indices[:, degree_idx]
        # Recalculate normalized values for the kept edges
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
        all_values = torch.cat((keep_values, keep_values))

        # Adjust indices for the full adjacency matrix
        keep_indices_adjusted = keep_indices.clone()
        keep_indices_adjusted[1] += self.n_users # User-Item edges
        flipped_indices = torch.flip(keep_indices, [0])
        flipped_indices[0] += self.n_users # Item-User edges

        all_indices = torch.cat((keep_indices_adjusted, flipped_indices), 1)
        self.masked_adj = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj.shape).coalesce().to(self.device)


    def _normalize_adj_m(self, indices, adj_size):
        # Ensure indices are valid before creating sparse tensor
        if indices.numel() == 0:
            return torch.tensor([], device=self.device) # Return empty tensor if no indices

        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0].float()), adj_size) # Use float for ones_like
        # Use sparse sum and handle potential division by zero
        row_sum = torch.sparse.sum(adj, dim=1).to_dense() + 1e-7
        col_sum = torch.sparse.sum(adj, dim=0).to_dense() + 1e-7 # Sum across rows for column sums

        r_inv_sqrt = torch.pow(row_sum, -0.5)
        c_inv_sqrt = torch.pow(col_sum, -0.5)

        # Clamp infinite values resulting from 1/sqrt(0) if epsilon wasn't enough
        r_inv_sqrt = torch.nan_to_num(r_inv_sqrt, nan=0.0, posinf=0.0, neginf=0.0)
        c_inv_sqrt = torch.nan_to_num(c_inv_sqrt, nan=0.0, posinf=0.0, neginf=0.0)


        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = c_inv_sqrt[indices[1]]

        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values

    def forward(self, adj):
        # Initialize h only if mm_adj exists and n_layers > 0
        h = None
        if hasattr(self, 'mm_adj') and self.mm_adj is not None and self.n_layers > 0:
            h = self.item_id_embedding.weight
            for i in range(self.n_layers):
                h = torch.sparse.mm(self.mm_adj, h)

        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        # Check if adj is valid before sparse mm
        if adj is not None and adj._nnz() > 0: # Check if adj is not None and has non-zero elements
            for i in range(self.n_ui_layers):
                # Ensure adj and ego_embeddings are compatible for sparse.mm
                try:
                    side_embeddings = torch.sparse.mm(adj, ego_embeddings)
                    ego_embeddings = side_embeddings # F.dropout(side_embeddings, self.dropout, training=self.training) # Optional dropout
                    all_embeddings += [ego_embeddings]
                except RuntimeError as e:
                    print(f"Error during sparse.mm in forward layer {i}: {e}")
                    print(f"adj shape: {adj.shape}, ego_embeddings shape: {ego_embeddings.shape}")
                    # Decide how to handle the error: break, continue, use previous embeddings?
                    break # Example: stop propagation if error occurs
        else:
             print("Warning: Adjacency matrix 'adj' is None or empty in forward pass.")


        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        # Add multimodal features only if they were computed
        if h is not None:
            i_g_embeddings = i_g_embeddings + h

        return u_g_embeddings, i_g_embeddings

    # Keep the original BPR loss function for direct calculation when needed
    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)
        return mf_loss

    # Removed contrastive_loss_hashing function

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        # Ensure masked_adj is generated before forward pass
        if self.masked_adj is None:
             self.pre_epoch_processing() # Generate masked_adj if it wasn't created

        # Check if masked_adj is valid before calling forward
        if self.masked_adj is None or self.masked_adj._nnz() == 0:
             print("Warning: masked_adj is None or empty in calculate_loss. Using norm_adj.")
             # Fallback to norm_adj or handle appropriately
             current_adj = self.norm_adj
             if current_adj is None or current_adj._nnz() == 0:
                 print("Error: Both masked_adj and norm_adj are invalid.")
                 # Return a zero loss or raise an error if forward pass cannot proceed
                 return torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            current_adj = self.masked_adj


        ua_embeddings, ia_embeddings = self.forward(current_adj)
        # self.build_item_graph = False # This flag seems unused, can be removed if not needed

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        # Calculate element-wise BPR terms
        pos_scores = torch.sum(torch.mul(u_g_embeddings, pos_i_g_embeddings), dim=1)
        neg_scores = torch.sum(torch.mul(u_g_embeddings, neg_i_g_embeddings), dim=1)
        bpr_terms = F.logsigmoid(pos_scores - neg_scores) # Shape: (batch_size,)

        # Initialize weights to 1
        weights = torch.ones_like(bpr_terms)

        # Calculate hash similarity weights only if features and hash layers exist
        if hasattr(self, 'text_trs') and hasattr(self, 'image_trs') and \
           hasattr(self, 'user_hash_layer') and hasattr(self, 'text_hash_layer') and \
           hasattr(self, 'image_hash_layer') and \
           self.t_feat is not None and self.v_feat is not None:

            text_feats = self.text_trs(self.text_embedding(pos_items))
            image_feats = self.image_trs(self.image_embedding(pos_items))

            # Use Sinkhorn distance to find potentially noisy pairs
            sinkhorn_loss_fn = SamplesLoss(loss="sinkhorn", p=1, blur=0.1, backend="auto")
            # Add small epsilon to avoid division by zero if text/image feats are identical
            sinkhorn_dist = sinkhorn_loss_fn(
                text_feats.unsqueeze(1),
                image_feats.unsqueeze(1) + 1e-8
            )

            # Normalize Sinkhorn distance (handle potential NaN if max_dist == min_dist)
            min_dist = torch.min(sinkhorn_dist)
            max_dist = torch.max(sinkhorn_dist)
            if max_dist == min_dist:
                 norm_sinkhorn_dist = torch.zeros_like(sinkhorn_dist)
            else:
                 norm_sinkhorn_dist = (sinkhorn_dist - min_dist) / (max_dist - min_dist + 1e-7)


            similarity_threshold = 0.7 # Threshold for considering pairs "clean"
            clean_mask = norm_sinkhorn_dist < similarity_threshold

            if clean_mask.any():
                # Calculate hash similarity only for clean samples
                u_clean = u_g_embeddings[clean_mask]
                text_clean = text_feats[clean_mask]
                image_clean = image_feats[clean_mask]

                # Ensure clean embeddings are not empty before hashing
                if u_clean.shape[0] > 0:
                    user_hash = torch.sign(self.user_hash_layer(u_clean))
                    text_hash = torch.sign(self.text_hash_layer(text_clean))
                    image_hash = torch.sign(self.image_hash_layer(image_clean))

                    # Calculate Hamming similarity (normalized dot product of signs)
                    pos_sim_text = torch.sum(user_hash * text_hash, dim=1) / self.hash_bits
                    pos_sim_image = torch.sum(user_hash * image_hash, dim=1) / self.hash_bits

                    # Normalize similarity to [0, 1] range
                    norm_pos_sim_text = (pos_sim_text + 1) / 2
                    norm_pos_sim_image = (pos_sim_image + 1) / 2

                    # Average similarity from both modalities
                    avg_hash_similarity_clean = (norm_pos_sim_text + norm_pos_sim_image) / 2

                    # Clamp weights to avoid potential issues (e.g., negative weights if normalization fails)
                    avg_hash_similarity_clean = torch.clamp(avg_hash_similarity_clean, min=0.0, max=1.0)


                    # Update weights for clean samples
                    weights[clean_mask] = avg_hash_similarity_clean

        # Calculate final weighted BPR loss
        # Add small epsilon to weights to prevent potential log(0) if weights become exactly 0
        # weighted_bpr_terms = (weights + 1e-8) * bpr_terms # Apply weights element-wise
        weighted_bpr_terms = weights * bpr_terms # Apply weights element-wise
        final_loss = -torch.mean(weighted_bpr_terms) # Average over the batch

        # Add regularization loss if needed (e.g., L2 on embeddings)
        # reg_loss = self.reg_weight * (l2_loss(u_g_embeddings) + l2_loss(pos_i_g_embeddings) + l2_loss(neg_i_g_embeddings))
        # final_loss += reg_loss

        return final_loss


    def full_sort_predict(self, interaction):
        user = interaction[0]

        # Use norm_adj for prediction as it represents the full graph
        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        # Calculate scores
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

# Helper function for L2 loss if needed
# def l2_loss(*embeddings):
#     loss = 0
#     for emb in embeddings:
#         loss += torch.sum(emb.pow(2)) / 2
#     return loss / embeddings[0].shape[0] # Normalize by batch size



