import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_with_label_vectors(logits, label_vectors):
    probs = logits.log_softmax(dim=1)
    return torch.mean(torch.sum(-label_vectors * probs, dim=1))


class MLP(nn.Module):
    def __init__(self, inp_dim, out_dim, num_layers=1, relu=True, bias=True):
        super(MLP, self).__init__()
        mod = []
        for L in range(num_layers-1):
            mod.append(nn.Linear(inp_dim, inp_dim, bias=bias))
            mod.append(nn.ReLU(True))

        mod.append(nn.Linear(inp_dim, out_dim, bias=bias))
        if relu:
            mod.append(nn.ReLU(True))

        self.mod = nn.Sequential(*mod)

    def forward(self, x):
        output = self.mod(x)
        return output


class CrossFactorPredictorModule(nn.Module):
    def __init__(self, latent_dim, num_sources, output_dim_list):
        super().__init__()
        self.num_sources = num_sources
        self.output_dim_list = output_dim_list
        self.num_outputs = len(self.output_dim_list)

        self.clf_head_dict = nn.ModuleDict()
        for i_src in range(self.num_sources):
            self.clf_head_dict[str(i_src)] = nn.ModuleDict()
            for i_tgt in range(self.num_outputs):
                if i_src != i_tgt:
                    self.clf_head_dict[str(i_src)][str(i_tgt)] = \
                        MLP(latent_dim, self.output_dim_list[i_tgt], 2, relu=False)

    def forward(self, z_list, labels, against_uniform=False):
        total_loss = 0

        aux_dict = {}
        for i_src in self.clf_head_dict.keys():
            for i_tgt in self.clf_head_dict[i_src].keys():
                logits = self.clf_head_dict[i_src][i_tgt](z_list[int(i_src)])

                if against_uniform:
                    label_vectors = torch.ones_like(logits) / logits.shape[1]
                    loss = cross_entropy_with_label_vectors(logits, label_vectors)
                else:
                    loss = F.cross_entropy(logits, labels[:, int(i_tgt)])
                    # label_vectors = F.one_hot(labels[:, int(i_tgt)], num_classes=logits.shape[1])
                    # cross_entropy_with_label_vectors(logits, label_vectors)

                total_loss += (loss / len(self.clf_head_dict[i_src]))

                # log
                aux_dict['cross_cls_acc_{}-{}'.format(i_src, i_tgt)] = \
                    (logits.argmax(dim=1) == labels[:, int(i_tgt)]).float().mean().item()

        return total_loss, aux_dict
