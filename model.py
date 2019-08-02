import torch
import torch.nn as nn
import numpy as np

class RankNet(nn.Module):
    def __init__(self, layers):
        super(RankNet, self).__init__()
        self.model = nn.Sequential(*layers)

    def forward(self, batch_ranking=None, batch_stds_labels=None, sigma=1.0):
        s_batch = self.model(batch_ranking)
        pred_diff = s_batch - s_batch.view(1, s_batch.size(0))
        row_inds, col_inds = np.triu_indices(batch_ranking.size()[0], k=1)
        si_sj = pred_diff[row_inds, col_inds]
        std_diffs = batch_stds_labels.view(batch_stds_labels.size(0), 1) - batch_stds_labels.view(1, batch_stds_labels.size(0))
        Sij = torch.clamp(std_diffs, min=-1, max=1)
        Sij = Sij[row_inds, col_inds]
        batch_loss_1st = 0.5 * sigma * si_sj * (1.0 - Sij)  # cf. the 1st equation in page-3
        batch_loss_2nd = torch.log(torch.exp(-sigma * si_sj) + 1.0)  # cf. the 1st equation in page-3
        batch_loss = torch.sum(batch_loss_1st + batch_loss_2nd)
        return batch_loss

    def predict(self, x):
        return self.model(x)
