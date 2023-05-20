from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import torch


class HinTop(nn.Module):

    def __init__(self,num_k=10):
        super(HinTop, self).__init__()
        self.crit = nn.MSELoss()
        self.num_k = num_k
        

    def forward(self, att_s, att_t,v_s,v_t,t=1,module=None,module_t=None):
        att_s = F.softmax(att_s/t, dim=-1)
        att_t = F.softmax(att_t/t, dim=-1)
        B, H, P, _, = att_t.shape
        H_s = att_s.shape[1]
        att_s = att_s.reshape(B*H_s*P,P)
        att_t = att_t.reshape(B*H*P,P)
        
        margin_t = torch.topk(
                    att_t, self.num_k, dim=-1
                )[0][:, -1]
        bool_topk_pos_t = att_t >= margin_t.unsqueeze(-1)

        margin_s = torch.topk(
                    att_s, self.num_k, dim=-1
                )[0][:, -1]
        bool_topk_pos_s = att_s >= margin_s.unsqueeze(-1)

        att_s = (att_s * bool_topk_pos_s).reshape(B,H_s,P,P)
        norm_s = att_s.sum(-1)

        att_t = (att_t * bool_topk_pos_t).reshape(B,H,P,P)
        norm_t = att_t.sum(-1)

        att_s = (att_s @ v_s)  / ( norm_s.unsqueeze(-1))
        att_t = (att_t @ v_t)  / ( norm_t.unsqueeze(-1))
        if module is not None:
            att_s = att_s.transpose(1, 2).reshape(B, P, -1)
            att_s = module(att_s)
            att_s = att_s.reshape(B,P,H,-1).transpose(1, 2)
        if module_t is not None:
            att_t = att_t.transpose(1, 2).reshape(B, P, -1)
            att_t = module_t(att_t)
            att_t = att_t.reshape(B,P,H,-1).transpose(1, 2)

        loss = self.crit(att_s, att_t)  

        return loss