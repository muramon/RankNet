import torch
import numpy as np
def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    def dcg_score(y_score, k=k, gains="exponential"):
        y_score_k = y_score[:k]

        if gains == "exponential":
            gains = torch.pow(2.0, y_score_k) - 1.0
            gains = gains.type(torch.FloatTensor)
        elif gains == "linear":
            gains = y_score
        else:
            raise ValueError("Invalid gains option.")
        discounts = torch.log2(torch.arange(k).type(torch.FloatTensor) + 2)
        return torch.sum(gains / discounts)

    best = dcg_score(y_true, k, gains)
    actual = dcg_score(y_score, k, gains)
    result = actual / best
    return result.item()

if __name__ == '__main__':
    sys_sorted_labels = [1, 1, 0, 1, 0, 1, 0, 0]
    ideal_sorted_labels=[1, 1, 1, 1, 0, 0, 0, 0]
    sys_sorted_labels = torch.from_numpy(np.asarray(sys_sorted_labels))
    ideal_sorted_labels = torch.from_numpy(np.asarray(ideal_sorted_labels))
    for k in [1, 3, 4, 8, 10]:
        if len(sys_sorted_labels) >= k:
            print(ndcg_score(ideal_sorted_labels, sys_sorted_labels, k=k))
