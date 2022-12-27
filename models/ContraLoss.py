import torch
import torch.nn.functional as F


def SRmClassifier(labels):
    labels = labels.detach().cpu().numpy().astype(int)
    # mcls = list(set(labels))
    mcls = set(labels)
    mcls_dict = {}
    for cls in mcls:
        # print(cls)
        tmp_dict = {cls: []}
        mcls_dict.update(tmp_dict)
    i = 0
    for c in labels:
        res = mcls_dict[c]
        res.append(i)
        tmp_dict = {labels[i]: res}
        mcls_dict.update(tmp_dict)
        i += 1
    return mcls_dict


def mini_ContraLoss(samples, labels, tau=0.5):
    B = samples.shape[0]
    # CVQRs [batch, embed_dim]

    samples = F.normalize(samples, p=2, dim=1)
    sim = torch.mm(samples, samples.T) / tau  # sim matrix
    # print(type(sim))
    sim = torch.exp(sim)  # e^sim
    sim = sim - torch.diag(torch.diag(sim))  # 0-diag

    cls_dict = SRmClassifier(labels)
    RankIndices = list(cls_dict.values())
    Loss_c = 0
    for b in range(B):
        positive_indices = []
        for i in range(len(RankIndices)):
            if b in RankIndices[i]:
                positive_indices = RankIndices[i]
        num_positive = len(positive_indices)
        denominator = torch.sum(sim[b])
        numerator = denominator
        # To avert the 'inf' value where the positive example is not exist.
        if num_positive != 1:
            numerator = 0
            for j in positive_indices:
                numerator += sim[b, j]
        l = - torch.log(numerator / denominator)
        Loss_c += l / num_positive
    return Loss_c / B


if __name__ == '__main__':
    samples = torch.randn([10, 576]).to('cuda')
    labels = torch.Tensor([3., 7., 2., 7., 2., 8., 1., 1., 4., 3.]).to('cuda')
    # l = list(set(labels.numpy()))
    # print(labels)
    # dict = SRmClassifier(labels)
    # print(dict)
    # print(list(dict.values()))
    loss = mini_ContraLoss(samples, labels, tau=100)
    print(loss)
