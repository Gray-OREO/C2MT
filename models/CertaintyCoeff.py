import torch
import time


def mini_CertaintyCoeff(preds, labels):
    """
    Performing pairwise comparisons between patches and increasing the weights in
    special cases of inconsistencies in different patches from different
    images, and the differences of special patches in the image will be
    accumulated in the iterable local credibility calculation.
    """
    K, C = preds.shape[0], torch.zeros_like(preds)
    for i in range(K):
        if i + 1 != K:
            for j in range(i + 1, K):
                if (labels[i] - labels[j]) * (preds[i] - preds[j]) > 0:
                    C[i] += 1
                    C[j] += 1
        else:
            return C / (K - 1)


if __name__ == '__main__':
    preds = torch.tensor([2.7, 6.5, 2.3, 7.2, 1.0, 6.5, 1.2, 0.6, 4.2, 3.1], device='cuda')
    labels = torch.tensor([3., 7., 2., 3., 2., 8., 1., 1., 7., 3.], device='cuda')
    patch_indices = torch.tensor([[0, 2], [3, 3], [7, 1], [0, 0], [5, 1],
                                  [6, 2], [11, 0], [13, 2], [3, 0], [0, 1]], device='cuda')
    t0 = time.time()
    mcc = mini_CertaintyCoeff(preds, labels)
    print(mcc)
    t1 = time.time()
    print('mcc_time:{:.4f}'.format(t1 - t0))
