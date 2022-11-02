import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .template import MetaTemplate


class ProtoNet(MetaTemplate):
    def __init__(self, params, model_func, n_way, n_support):
        super(ProtoNet, self).__init__(params, model_func, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fake_n = params.fake
        self.true_n = params.true

    def feature_forward(self, x):
        out = self.avgpool(x).view(x.size(0),-1)
        return out

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)
        z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        #scores = self.euclidean_dist(z_query, z_proto)
        dists_1        = self.euclidean_dist(z_query, z_proto)
        dists_1_t      = torch.transpose(dists_1, dim0=0, dim1=1)
        c = self.fake_n
        t = self.true_n
        if c == 0:
            return dists_1
        values,indices = torch.topk(dists_1_t,20,largest=True)
        y_label = np.repeat(range(self.n_way), self.n_query)
        indices_add_fu = np.zeros((self.n_way,t+c))
        c_add_fu = 0
        c_add_zheng = 0
        for i in range(5):
            for j in range(20):
                if y_label[indices[i][j]] == i:
                    if c_add_zheng == t:
                        pass
                    else:
                        indices_add_fu[i][c_add_zheng] = indices[i][j]
                        c_add_zheng = c_add_zheng + 1
                else:
                    if c_add_fu == c:
                        pass
                    else:
                        indices_add_fu[i][t+c_add_fu] = indices[i][j]
                        c_add_fu = c_add_fu + 1
                    #rint(c_add_fu,c)
                if c_add_fu == c and c_add_zheng == t:
                    c_add_fu = 0
                    c_add_zheng = 0
                    break
        indices_add_fu = torch.tensor(indices_add_fu).long().cuda(0)
        for i in range(5):
            fake_support = torch.index_select(z_query,dim=0,index=indices_add_fu[i]).view(1,t+c,-1)
            r_support = z_support[i].view(1,self.params.n_shot,-1)
            r_f_support_single = torch.cat((fake_support,r_support),dim=1)
            if i == 0:
                r_f_support = r_f_support_single
            else:
                r_f_support = torch.cat((r_f_support,r_f_support_single),dim=0)
        z_proto_new = r_f_support.contiguous().view(self.n_way, self.params.n_shot+t+c, -1 ).mean(1)
        #z_proto = z_support.contiguous().view(self.params.val_n_way, self.params.n_shot, -1).mean(1)
        dists_1_new = self.euclidean_dist(z_query, z_proto_new)
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

    def euclidean_dist(self, x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        score = -torch.pow(x - y, 2).sum(2)
        return score
