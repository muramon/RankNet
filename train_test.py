import torch
import math
from ndcg import ndcg_score

def train_step(model, l2r_dataset, optimizer):
    epoch_loss_ls = []
    for batch_rankings, batch_std_labels in l2r_dataset:
        loss = model(batch_ranking=batch_rankings, batch_stds_labels=batch_std_labels, sigma=1.0)
        epoch_loss_ls.append(loss.item())
        model.zero_grad()
        loss.backward()
        optimizer.step()
    return sum(epoch_loss_ls) / len(epoch_loss_ls)



def test_step(model, test_ds):
    results = {}
    for k in [1, 3, 5, 10]:
        ndcg_ls = []
        for batch_rankings, labels in test_ds:
            pred = model.predict(batch_rankings)
            pred_ar = pred.squeeze(1).detach()
            label_ar = labels.detach()
            _, argsort = torch.sort(pred_ar, descending=True, dim=0)
            pred_ar_sorted = label_ar[argsort]
            if len(pred_ar_sorted) >= k:
                ndgc_s = ndcg_score(label_ar, pred_ar_sorted, k=k)
                if not math.isnan(ndgc_s):
                    ndcg_ls.append(ndgc_s)
        results[k] = sum(ndcg_ls) / len(ndcg_ls)

    return results
