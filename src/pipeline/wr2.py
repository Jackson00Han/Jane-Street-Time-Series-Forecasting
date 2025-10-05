# metrics_weighted_r2.py
import torch
from torch.nn.utils import rnn
from pytorch_forecasting.metrics import MultiHorizonMetric
from pytorch_forecasting.utils import unpack_sequence, create_mask

class WeightedR2(MultiHorizonMetric):
    """
    Zero-mean Weighted R^2 for decoder horizon:
        WR2 = 1 - (Σ w * (y - ŷ)^2) / (Σ w * y^2)

    - 支持 target 为 Tensor 或 PackedSequence
    - 支持 (target, weight)，其中 weight 可以是 [B, T]（逐步权重）
    - 仅在有效 decoder 步上累计（通过 length mask 或权重=0 达成）
    """
    def __init__(self, eps: float = 1e-8, **kwargs):
        # reduction 给父类随便传个值（我们不使用父类的 losses/lengths 聚合）
        super().__init__(reduction="mean", **kwargs)
        self.eps = eps
        self.add_state("num", default=torch.tensor(0.0), dist_reduce_fx="sum")  # Σ w*(y-ŷ)^2
        self.add_state("den", default=torch.tensor(0.0), dist_reduce_fx="sum")  # Σ w*y^2

    def update(self, y_pred, target):
        # 1) 取预测（与 PF 约定一致）
        yhat = self.to_prediction(y_pred)          # [B,T] 或 [B,T,1]
        if yhat.ndim == 3 and yhat.size(-1) == 1:
            yhat = yhat.squeeze(-1)                # -> [B,T]

        # 2) 拆出 target / weight
        if isinstance(target, (list, tuple)) and not isinstance(target, rnn.PackedSequence):
            y, w = target
        else:
            y, w = target, None

        # 3) 变长与长度
        if isinstance(y, rnn.PackedSequence):
            y, lengths = unpack_sequence(y)        # y: [B,T]
        else:
            lengths = torch.full((y.size(0),), y.size(1), dtype=torch.long, device=y.device)

        B, T = y.shape
        if yhat.shape[:2] != (B, T):
            raise ValueError(f"Shape mismatch: pred {tuple(yhat.shape[:2])} vs target {(B, T)}")

        # 4) 有效步 mask（True=有效）
        valid_bool = create_mask(T, lengths, inverse=True)     # [B,T] bool
        valid = valid_bool.to(y.dtype)                         # 1/0

        # 5) 逐步权重（若你已有 [B,T] 且无效步已置0，这里会自然等价）
        if w is None:
            w_eff = valid
        else:
            # 若你保证无效步权重=0，则乘 valid 不会改变；保留这步以防未来变长
            if w.ndim == 1:
                w_eff = w.view(-1, 1).expand(B, T) * valid
            else:
                w_eff = w * valid

        # 6) 累计分子/分母（零均值分母：Σ w*y^2）
        sse  = (w_eff * (y - yhat) ** 2).sum()
        sst0 = (w_eff * (y ** 2)).sum().clamp_min(self.eps)

        self.num += sse
        self.den += sst0

    def compute(self):
        return 1.0 - (self.num / self.den)
