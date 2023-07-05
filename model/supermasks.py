import torch.nn as nn
import torch
from torch.distributions import Bernoulli, RelaxedBernoulli
SMOOTH = 1e-6
from itertools import combinations


class SuperMask(nn.Module):
    def __init__(self, args, act_size, init_setting="random_uniform", init_scalar=1):
        super().__init__()
        
        self.args = args
        self.domain_num = args.num_classes
        self.act_size = act_size
        self.init_setting = init_setting
        self.init_scalar = init_scalar
        self.domain_list = args.domain_list

        # Define the super mask logits
        if self.init_setting == "random_uniform":
            # self.super_mask_logits = nn.ParameterDict(
            #     {
            #         str(x): nn.init.xavier_normal_(nn.Parameter(torch.empty(self.act_size, requires_grad=True)))
            #         # nn.Parameter(torch.rand(self.act_size, requires_grad=True))
            #         for x in args.seen_c
            #     }
            # )
            # self.super_mask_logits = nn.init.xavier_normal_(nn.Parameter(torch.empty(self.act_size, requires_grad=True)))
            self.super_mask_logits = nn.ParameterList(
                {
                    nn.init.xavier_normal_(nn.Parameter(torch.empty(self.act_size, requires_grad=True)))
                    for x in range(len(self.domain_list))
                }
            )
        elif self.init_setting == "scalar":
            param_tensor = torch.ones(self.act_size, requires_grad=True)
            param_tensor = param_tensor.new_tensor(
                [self.init_scalar] * self.act_size, requires_grad=True
            )
            self.super_mask_logits = nn.ParameterList(
                {nn.Parameter(param_tensor.clone()) for x in range(self.domain_num)}
            )
        elif self.init_setting == 'encoder':
            self.encoder_weight = nn.init.xavier_normal_(
                nn.Parameter(torch.empty([act_size[0], act_size[1], args.att_emb_dim], requires_grad=True)))

    def forward(self, activation, targets, mode="sample", conv_mode=False):
        # Mask repeated along channel dimensions if conv_mode == True
        activation = activation.permute(1, 0, 2) #bs na dim
        # todo: K个mask，先把target映射到具体domain中。后续根据domain_label挑选domain_mask
        domain_label = torch.zeros_like(targets)
        for i, label_in_domain in enumerate(self.domain_list):
            for l in label_in_domain:
                domain_label[targets == l] = i
        if mode == "sample":
            # todo: replace domains by domain_label
            probs = [torch.sigmoid((self.super_mask_logits[x])) for x in domain_label]
            probs = torch.stack(probs)
            # probs = torch.sigmoid()(self.super_mask_logits)
            mask_dist = Bernoulli(probs)
            hard_mask = mask_dist.sample()
            soft_mask = probs
            mask = (hard_mask - soft_mask).detach() + soft_mask
            if conv_mode and len(activation.shape) > 2:
                apply_mask = mask.view(mask.shape[0], mask.shape[1], 1, 1)
                apply_mask = apply_mask.repeat(
                    1, 1, activation.shape[2], activation.shape[3]
                )
                activation = apply_mask * activation
            else:
                'bs*na*c X na*c'
                activation = mask * activation
        elif mode == "greedy":
            probs = [torch.sigmoid((self.super_mask_logits[x])) for x in domain_label]
            probs = torch.stack(probs)
            hard_mask = (probs > 0.5).float()
            soft_mask = probs
            mask = (hard_mask - soft_mask).detach() + soft_mask
            if conv_mode and len(activation.shape) > 2:
                apply_mask = mask.view(mask.shape[0], mask.shape[1], 1, 1)
                apply_mask = apply_mask.repeat(
                    1, 1, activation.shape[2], activation.shape[3]
                )
                activation = apply_mask * activation
            else:
                activation = mask * activation
        elif mode == "softscale":
            probs = [torch.sigmoid((self.super_mask_logits[x])) for x in domain_label]
            probs = torch.stack(probs)
            # probs = torch.sigmoid()(self.super_mask_logits)
            soft_mask = probs
            if conv_mode and len(activation.shape) > 2:
                apply_mask = soft_mask.view(
                    soft_mask.shape[0], soft_mask.shape[1], 1, 1
                )
                apply_mask = apply_mask.repeat(
                    1, 1, activation.shape[2], activation.shape[3]
                )
                activation = apply_mask * activation
            else:
                activation = soft_mask * activation
        elif mode == "avg_mask_softscale":
            # Average all the source domain masks
            # instead of combining them
            '''
            sf = torch.from_numpy(self.args.sf)
            seen_sf = sf[self.args.seen_c, :]
            idx1, idx2 = torch.nonzero(seen_sf, as_tuple=True)
            avg = seen_sf[idx1, idx2].mean()
            # pos_index = torch.where(sf > avg, 1, 0)
            # pos_index[self.args.unseen_c, :] = 0.0
            # 取消pos_index中不可见类的选择下标
            # all_probs = torch.stack([self.super_mask_logits[x] for x in range(sf.shape[0])])
            pos_index = torch.where(seen_sf > (2 * avg), 1, 0)
            all_probs = torch.stack([self.super_mask_logits[str(x.item())] for x in self.args.seen_c])
            # 拼接所有类的mask得到nc*na*dim的张量
            mean_probs = [torch.mean(torch.sigmoid(all_probs[pos_index[:, x], x, :]), dim=0) for x in range(sf.shape[1])]
            # 每个属性根据pos_index找到出现多的属性mask
            '''
            all_probs = [torch.sigmoid(self.super_mask_logits[x]) for x in range(len(self.domain_list))]
            all_probs = torch.mean(torch.stack(all_probs), 0)
            mean_mask = [all_probs for x in targets]
            # mean_probs = torch.stack(mean_probs).squeeze()
            # mean_mask = [mean_probs for x in domains]
            mean_mask = torch.stack(mean_mask)
            soft_mask = mean_mask.squeeze()
            if conv_mode and len(activation.shape) > 2:
                apply_mask = soft_mask.view(
                    soft_mask.shape[0], soft_mask.shape[1], 1, 1
                )
                apply_mask = apply_mask.repeat(
                    1, 1, activation.shape[2], activation.shape[3]
                )
                activation = apply_mask * activation
            else:
                activation = soft_mask * activation
        elif mode == 'encoder':
            activation = torch.einsum('ijk,jkl->ijl', activation, self.encoder_weight)
        # na*bs*dim
        return activation.permute(1, 0, 2)

    def sparsity(self, mask):
        return torch.mean(mask, dim=1)

    def sparsity_penalty(self):
        sparse_pen = 0
        for mask in self.super_mask_logits:
            # 原本是torch.sum
            sparse_pen += torch.mean(torch.sigmoid(mask))
        return sparse_pen / len(self.super_mask_logits)

    def overlap_penalty(self):
        overlap_pen = 0
        if len(self.domain_list) == 1:
            return overlap_pen
        domain_pairs = list(combinations(range(len(self.domain_list)), 2))
        for pair in domain_pairs:
            dom1, dom2 = pair
            mask1 = torch.sigmoid((self.super_mask_logits[dom1]))
            mask2 = torch.sigmoid((self.super_mask_logits[dom2]))
            intersection = torch.sum(mask1 * mask2)
            union = torch.sum(mask1 + mask2 - mask1 * mask2)
            iou = (intersection + SMOOTH) / (union + SMOOTH)
            overlap_pen += iou
        overlap_pen /= len(domain_pairs)
        return overlap_pen
