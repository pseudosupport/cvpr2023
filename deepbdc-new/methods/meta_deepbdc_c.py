import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .template import MetaTemplate
from .bdc_module import BDC

class MetaDeepBDC(MetaTemplate):
    def __init__(self, params, model_func, n_way, n_support):
        super(MetaDeepBDC, self).__init__(params, model_func, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.fake_n = params.fake
        
        reduce_dim = params.reduce_dim
        self.feat_dim = int(reduce_dim * (reduce_dim+1) / 2)
        self.dcov = BDC(is_vec=True, input_dim=self.feature.feat_dim, dimension_reduction=reduce_dim)

    def feature_forward(self, x):
        out = self.dcov(x)
        return out

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)
        z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        #scores         = self.metric(z_query, z_proto)
        dists_1        = self.metric(z_query, z_proto)
        dists_1_t      = torch.transpose(dists_1, dim0=0, dim1=1)
        c = self.fake_n
        if c == 0:
            return dists_1
        values,indices = torch.topk(dists_1_t,c,largest=True)
        for i in range(5):
            fake_support = torch.index_select(z_query,dim=0,index=indices[i]).view(1,c,-1)
            r_support = z_support[i].view(1,self.n_support,-1)
            r_f_support_single = torch.cat((fake_support,r_support),dim=1)
            if i == 0:
                r_f_support = r_f_support_single
            else:
                r_f_support = torch.cat((r_f_support,r_f_support_single),dim=0)
        z_proto_new = r_f_support.contiguous().view(self.n_way, self.n_support+c, -1 ).mean(1)
        dists_1_new = self.metric(z_query, z_proto_new)
        scores = dists_1_new
        return scores

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())
        y_label = np.repeat(range(self.n_way), self.n_query)
        scores = self.set_forward(x)
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_label)

        return float(top1_correct), len(y_label), self.loss_fn(scores, y_query), scores

    def metric(self, x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        if self.n_support > 1:
            dist = torch.pow(x - y, 2).sum(2)
            score = -dist
        else:
            score = (x * y).sum(2)
        return score
