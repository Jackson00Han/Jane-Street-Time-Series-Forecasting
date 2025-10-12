# metrics_wr2.py（修订后的最终版）
import torch
from pytorch_forecasting.metrics import MultiHorizonMetric

class WeightedCorrSq(MultiHorizonMetric):
    higher_is_better = True

    def __init__(self, eps: float = 1e-9, name: str = "WR2", **kwargs):
        super().__init__(name=name, **kwargs)   # 关键：把 name 传给父类；不要自定义 @property name
        self.eps = eps
        self.add_state("w_sum",   default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("wx_sum",  default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("wy_sum",  default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("wxx_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("wyy_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("wxy_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, y_pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None, **kwargs):
        x = self.to_prediction(y_pred)
        y = self.to_prediction(target)
        w = torch.ones_like(y) if weights is None else self.to_prediction(weights)

        mask = torch.isfinite(x) & torch.isfinite(y) & torch.isfinite(w)
        if not torch.any(mask):
            return
        x, y, w = x[mask], y[mask], torch.clamp(w[mask], min=0.0)

        self.w_sum   += w.sum()
        self.wx_sum  += (w * x).sum()
        self.wy_sum  += (w * y).sum()
        self.wxx_sum += (w * x * x).sum()
        self.wyy_sum += (w * y * y).sum()
        self.wxy_sum += (w * x * y).sum()

    def compute(self) -> torch.Tensor:
        eps = self.eps
        w = torch.clamp(self.w_sum, min=eps)
        mx, my = self.wx_sum / w, self.wy_sum / w
        cov  = self.wxy_sum / w - mx * my
        varx = self.wxx_sum / w - mx * mx
        vary = self.wyy_sum / w - my * my
        return (cov * cov) / (torch.clamp(varx, min=eps) * torch.clamp(vary, min=eps) + eps)
