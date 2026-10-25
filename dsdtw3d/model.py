"""The upstream CNN/GRU and triplet objective with a nine-channel XYZ input."""
import torch
from torch import nn
from torch.nn import functional as F
from soft_dtw_cuda import SoftDTW


class DsDTW3D(nn.Module):
    def __init__(self, softdtw="cpu"):
        super().__init__()
        if softdtw not in ("cpu", "cuda"):
            raise ValueError("softdtw must be cpu or cuda")
        self.conv = nn.Sequential(nn.Conv1d(9, 64, 7, padding=3), nn.MaxPool1d(2, 2, ceil_mode=True),
                                  nn.ReLU(inplace=True), nn.Conv1d(64, 128, 3, padding=1),
                                  nn.ReLU(inplace=True), nn.Dropout(0.1))
        self.rnn = nn.GRU(128, 128, 2, batch_first=True, dropout=0.1)
        self.linear = nn.Linear(128, 64, bias=False)
        for layer in range(2):
            with torch.no_grad():
                getattr(self.rnn, f"bias_hh_l{layer}")[128:256].fill_(-1e10)
                getattr(self.rnn, f"bias_ih_l{layer}")[128:256].fill_(-1e10)
        nn.init.kaiming_normal_(self.linear.weight, a=1)
        for index in (0, 3):
            nn.init.kaiming_normal_(self.conv[index].weight, a=0)
            nn.init.zeros_(self.conv[index].bias)
        self.dtw = SoftDTW(softdtw == "cuda", gamma=5, normalize=False, bandwidth=0.1)

    def forward(self, signatures):
        output = self.conv(signatures.transpose(1, 2)).transpose(1, 2)
        # Prepared trajectories have equal length; dynamic h0 supports any batch size.
        h0 = output.new_zeros(2, output.shape[0], 128)
        output, _ = self.rnn(output, h0)
        output = self.linear(output)
        if self.training:
            output = F.avg_pool1d(output.transpose(1, 2), 2, 2).transpose(1, 2)
        return output

    def triplet_loss(self, output, tasks=4, genuine=5, forged=10):
        step = 1 + genuine + forged
        if len(output) != tasks * step:
            raise ValueError("Batch does not match the task structure")
        standard, hard, variance = [], [], []
        length = output.shape[1]
        for task in range(tasks):
            group = output[task * step:(task + 1) * step]
            anchor = group[0:1]
            positive = torch.stack([self.dtw(anchor, sample[None]).reshape(()) / (2 * length)
                                    for sample in group[1:1 + genuine]])
            negative = torch.stack([self.dtw(anchor, sample[None]).reshape(()) / (2 * length)
                                    for sample in group[1 + genuine:]])
            triplets = F.relu(positive[:, None] - negative[None, :] + 1)
            standard.append(triplets.mean())
            hard.append(triplets.sum() / (torch.count_nonzero(triplets) + 1))
            variance.append(positive.mean())
        return torch.stack(standard).mean(), torch.stack(hard).mean(), torch.stack(variance).mean()
