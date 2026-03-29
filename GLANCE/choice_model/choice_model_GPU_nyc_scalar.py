import numpy as np
import math
import torch
from torch.nn.functional import softmax
from torch.autograd import grad, Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from entmax import sparsemax, entmax15, entmax_bisect, normmax_bisect, budget_bisect

############################## Choice Model
class ChoiceModelScalar(nn.Module):
    def __init__(self, data, H, device):
        super(ChoiceModelScalar, self).__init__()
        self.data = data
        self.device = device
        self.M, self.unique_tuples_list, self.unique_t, self.unique_precinct_idx = self.count_possible_choice()
        self.SUSP_RACE_list, self.SUSP_SEX_list = self.get_features()
        
        self.H = H
        self.d_model = 32
        self.d_h = 16
        
        self.patch_linear_A = nn.Linear(in_features=1, out_features=self.d_model, bias=False).to(self.device)
        self.patch_linear_B = nn.Linear(in_features=1, out_features=self.d_model, bias=False).to(self.device)
        
        self.W_a_H = nn.ModuleList([nn.Linear(in_features=self.d_model, out_features=self.d_h, bias=False).to(self.device) for i in range(self.H)])
        self.W_b_H = nn.ModuleList([nn.Linear(in_features=self.d_model, out_features=self.d_h, bias=False).to(self.device) for i in range(self.H)])
        
        ## position vector, used for temporal/spatial position encoding
        self.position_vec = torch.tensor([math.pow(10000.0, 2.0 * (i // 2) / self.d_model) for i in range(self.d_model)], device=self.device)

                
        ## claim learnable alpha
        # self.alpha = nn.Parameter(torch.empty(self.H).uniform_(1.25, 1.45)) # .to(self.device)
        self.alpha = nn.Parameter(torch.empty(self.H).uniform_(1.5, 1.5)).to(self.device)
        self.relu = nn.ReLU()
        
        self.U = nn.Parameter(torch.empty(self.H, self.M).uniform_(1.20, 1.50)) # .to(self.device)
        
        self.pi = nn.Parameter(torch.randn(self.H)) # .to(self.device)
        
    def count_possible_choice(self):
        pair_list = []
        for id in list(self.data.keys()):
            sample = self.data[id]
            ## for each "pair", it is a tuple with 4 elements
            ## the 1-st item is time_grid_center
            ## the 2-nd item is precinct_idx
            pair = sample["time_location_idx_pair"]
            pair_list.append(pair)

        ## remove repeated item
        unique_tuples_list = list(set(pair_list))
        ## count unique pair
        unique_tuple_count = len(unique_tuples_list)
        
        t_0 = [t[0] for t in unique_tuples_list] ## time_grid_center
        t_1 = [t[1] for t in unique_tuples_list] ## precinct_idx
             
        ## turn to tensor and make their shape
        unique_time_grid_center_tensor = torch.tensor([t_0]).float() # .t() # .to(torch.long)
        unique_precinct_index_tensor = torch.tensor([t_1]).float() # .t() # .to(torch.long)
        
        ## move to same device
        # unique_tuples_list = unique_tuples_list.to(self.device)
        unique_time_grid_center_tensor = unique_time_grid_center_tensor.to(self.device)
        unique_precinct_index_tensor = unique_precinct_index_tensor.to(self.device)
        
        return unique_tuple_count, unique_tuples_list, unique_time_grid_center_tensor, unique_precinct_index_tensor
    
    def get_features(self):
        SUSP_RACE_list = []
        SUSP_SEX_list = []
        for idx in range((len(list(self.data.keys())))):
            sample = self.data[idx]
            SUSP_RACE_list.append(sample["SUSP_RACE"])
            SUSP_SEX_list.append(sample["SUSP_SEX"])
        
        SUSP_RACE_list = list(set(SUSP_RACE_list))
        SUSP_SEX_list = list(set(SUSP_SEX_list))
        
        return SUSP_RACE_list, SUSP_SEX_list
    
    def patch_embedding(self, input):
        patch_embed_A = self.patch_linear_A(input.t()).t()
        patch_embed_B = self.patch_linear_B(input.t()).t()
        patch_embed_A = patch_embed_A.t()
        patch_embed_B = patch_embed_B.t()
        return patch_embed_A, patch_embed_B
    
    def position_embedding(self, input):
        pos_embed = input.unsqueeze(-1) / self.position_vec
        pos_embed[:, :, 0::2] = torch.sin(pos_embed[:, :, 0::2])
        pos_embed[:, :, 1::2] = torch.cos(pos_embed[:, :, 1::2])
        pos_embed = pos_embed.squeeze()
        return pos_embed
    
    def embedding(self):
        ## spatial embedding
        spatial_pos_embed = self.position_embedding(self.unique_precinct_idx)
        spatial_patch_embed_A, spatial_patch_embed_B = self.patch_embedding(self.unique_precinct_idx)
        
        ## temporal embedding
        temporal_pos_embed = self.position_embedding(self.unique_t)

        embed_A = spatial_pos_embed + spatial_patch_embed_A + temporal_pos_embed
        embed_B = spatial_pos_embed + spatial_patch_embed_B + temporal_pos_embed
        
        A_H_list = []
        B_H_list = []
        for layer in self.W_a_H:
            A_H_list.append(layer(embed_A))
        for layer in self.W_b_H:
            B_H_list.append(layer(embed_B))
            
        A_H = torch.stack(A_H_list)
        B_H = torch.stack(B_H_list)

        return A_H, B_H
    
    def gating_function(self, A_H, B_H):
    
        B_H_transposed = B_H.permute(0, 2, 1)

        matrix_H = torch.bmm(A_H, B_H_transposed)
        z_H = torch.sum(matrix_H, dim=1)        
        ## normalization for each z_h
        z_H = torch.nn.functional.normalize(z_H, p=2, dim=1)    
        ## compute g_res_H
        g_res_H = torch.empty(0, self.M).to(self.device)
        for h in range(self.H):
            alpha_h = self.alpha[h]
            z_h = z_H[h].unsqueeze(dim=0)
            g_res_h = entmax_bisect(z_h, alpha_h)
            g_res_H = torch.cat((g_res_H, g_res_h), dim=0)

        return matrix_H, g_res_H

    def prob_loc_time_pair(self, g_res_H):   

        ## normalization for the mixing coefficient
        pi_clone = self.pi.clone()
        normalized_pi = F.softmax(pi_clone, dim=0)
        normalized_pi = normalized_pi.unsqueeze(1).repeat(1, self.M)
        upper = torch.mul(g_res_H, torch.exp(self.U))
        upper = self.relu(upper) # to avoid numerical issue

        lower = torch.sum(torch.mul(g_res_H, torch.exp(self.U)), dim=1)
        
        lower = lower.unsqueeze(1).repeat(1, self.M)
        P_M = torch.sum(normalized_pi * (upper / lower), dim=0)
        
        return P_M
 
    def log_likelihood(self, time, precinct, sex, race):
        
        ##### turn time-location pair to one-hot matrix
        batch_size = time.shape[0]
        
        time_location_pairs = list(zip(time.tolist(), precinct.tolist()))

        tuple_index_dict = {item: index for index, item in enumerate(self.unique_tuples_list)}

        indices = [tuple_index_dict[pair] for pair in time_location_pairs]

        one_hot = torch.zeros(batch_size, self.M).to(self.device)

        indices_tensor = torch.LongTensor(indices).to(self.device)
        one_hot.scatter_(1, indices_tensor.unsqueeze(1), 1)

        ##### compute log-likelihood
        A_H, B_H = self.embedding()
        matrix_H, g_res_H = self.gating_function(A_H, B_H)

        P_M = self.prob_loc_time_pair(g_res_H)
    
        P_M = P_M.repeat(batch_size, 1)  

        batch_ll = torch.log(torch.sum(P_M * one_hot, dim=1))

        batch_sum_ll = torch.sum(batch_ll)

        return batch_sum_ll, matrix_H, g_res_H

