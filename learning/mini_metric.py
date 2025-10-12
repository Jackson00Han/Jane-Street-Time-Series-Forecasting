import torch
from torch.nn.utils.rnn import pack_padded_sequence
from pytorch_forecasting.metrics import MultiHorizonMetric

class HorizonMAE_PF(MultiHorizonMetric):
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction=reduction)
    def loss(self, y_pred, target):
        if y_pred.dim() == 3 and y_pred.size(2) == 1:
            y_pred = y_pred.squeeze(2)
        return (y_pred - target).abs()

# ---- 构造一个小批次：B=2, T=4，长度分别为3和2 ----
y_pred  = torch.tensor([[1.,  2.,  3.,  4.],
                        [10., 20., 30., 40.]]).unsqueeze(-1)   # [B,T,1]
target  = torch.tensor([[1.,  1.,  1.,   1.],
                        [11., 19., 33., 100.]])                # [B,T]
lengths = torch.tensor([3, 2], dtype=torch.long)
weight  = torch.tensor([[1., 2., 1., 999.],
                        [3., 0.5, 7., 888.]])

# ✨ 关键：用 PackedSequence 提供有效长度
packed_target = pack_padded_sequence(target, lengths.cpu(), batch_first=True, enforce_sorted=False)

# ✨ 关键：把 y_pred 截到 max_len=3（与解包后的 target 对齐）
y_pred3 = y_pred[:, :3, :]   # [B,3,1]

# 1) mean 规约：应得到 1.5
m_mean = HorizonMAE_PF(reduction="mean")
m_mean.update(y_pred3, (packed_target, weight[:, :3]))
print("mean =", float(m_mean.compute()))  # 期望 1.5

# 2) none 规约：应返回 [B,3]，被 mask 的步为 NaN（这里 T=3，都有效）
m_none = HorizonMAE_PF(reduction="none")
m_none.update(y_pred3, (packed_target, weight[:, :3]))
print("none matrix =\n", m_none.compute())
