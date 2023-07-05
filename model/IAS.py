import torch
import torch.nn as nn
import torch.nn.functional as F


class Model_IAS(nn.Module):

    def __init__(self, img_dims, att_dims, scale=1, bias=False):
        super(Model_IAS, self).__init__()
        # todo:取消bias
        self.L1 = nn.Linear(att_dims, int(img_dims/2), bias=bias)
        self.L2 = nn.Linear(int(img_dims/2), img_dims, bias=bias)
        self.ReLU1 = torch.nn.LeakyReLU()
        self.ReLU2 = torch.nn.LeakyReLU()
        self.att_dims = att_dims
        #self.ReLU = torch.nn.ReLU()
        #self.img_guaid = nn.Linear(img_dims, att_dims, bias=True)

    def forward(self, x, AttM, pre_attri, att_weight):
        bs = pre_attri.shape[0]
        nc = AttM.shape[0]
        '''
        att = self.img_guaid(x)
        att = self.ReLU(att)
        att = F.softmax(att / 5).reshape(-1, 1, self.att_dims)
        att = att + torch.ones_like(att)
        att = att.repeat(1, nc, 1)
        '''
        AttM = AttM.reshape(1, -1, self.att_dims)
        # 1 nc na
        AttM = AttM.repeat(bs, 1, 1)
        # bs nc na
        pre_attri = pre_attri.reshape(-1, 1, self.att_dims)
        # # bs nc na
        pre_attri = pre_attri.repeat(1, nc, 1)

        AttM = AttM + att_weight * pre_attri
        # AttM = 0.9 * AttM + 0.1 * pre_attri
        #AttM = AttM * att

        W1 = self.L1(AttM)
        W1 = self.ReLU1(W1)
        W2 = self.L2(W1)
        classifier = self.ReLU2(W2)
        classifier = F.normalize(classifier, p=2, dim=-1, eps=1e-12)
        # bs nc dim * dim bs -> bs nc

        # out = torch.matmul(classifier, x.t())
        # # bs nc bs
        # out = out.permute(1, 0, 2)
        # out = torch.diagonal(out, offset=0, dim1=1, dim2=2)
        # out = out.t()
        out = torch.einsum('ijk,ik->ij', classifier, x)

        return out