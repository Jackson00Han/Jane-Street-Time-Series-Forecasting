import torch
from torch import nn

class MiniTFT_Step5(nn.Module):
    """
    Step5：为 encoder/decoder 各自独立做变量选择（简化版 VSN），
           并提供静态上下文占位接口（后续用于 LSTM 初始态）。
    """
    def __init__(self, hidden_size: int, output_size: int, x_reals: list[str]):
        super().__init__()
        self.H = hidden_size
        self.O = output_size
        self.x_reals = x_reals
        F = len(x_reals)

        # 每列连续特征的投影层
        self.prescalers = nn.ModuleDict({name: nn.Linear(1, self.H) for name in x_reals})

        # TODO 1: 定义两套“列打分层”（不带 bias），形状 F->F
        self.vsn_score_enc = nn.Linear(F, F, bias=False)  # <- 你来填（提示：nn.Linear(F, F, bias=False)）
        self.vsn_score_dec = nn.Linear(F, F, bias=False)  # <- 你来填

        # 输出头（只对 decoder 段）
        self.head = nn.Linear(self.H, self.O)

    @staticmethod
    def expand_static_context(context: torch.Tensor, timesteps: int) -> torch.Tensor:
        # 占位：把 (B, H) 扩展为 (B, T, H)
        return context[:, None, :].expand(-1, timesteps, -1)

    def _project_with_vsn(self, x_cont: torch.Tensor, which: str) -> torch.Tensor:
        B, T, F = x_cont.shape
        assert F == len(self.x_reals)

        # 逐列投影并堆叠 -> (B, T, H, F)
        outs = [self.prescalers[name](x_cont[..., j:j+1]) for j, name in enumerate(self.x_reals)]
        stacked = torch.stack(outs, dim=-1)  # (B, T, H, F)

        # TODO 2: 根据 which 选择打分层（"enc" or "dec"）
        scorer = self.vsn_score_enc if which == "enc" else self.vsn_score_dec  # <- 你来填（提示：三元表达式）

        # 列权重 -> (B, T, F)
        scores  = scorer(x_cont)
        weights = torch.softmax(scores, dim=-1)

        # TODO 3: 用“矩阵法”聚合到 (B, T, H)
        h = (stacked @ weights.unsqueeze(-1)).squeeze(-1)  # <- 你来填（提示：矩阵乘法）

        return h

    def forward(self, x: dict) -> dict:
        enc = x["encoder_cont"]  # (B, Te, F)
        dec = x["decoder_cont"]  # (B, Td, F)

        enc_h = self._project_with_vsn(enc, which="enc")  # (B, Te, H)  先不使用
        dec_h = self._project_with_vsn(dec, which="dec")  # (B, Td, H)

        pred = self.head(dec_h)  # (B, Td, O)
        return {"prediction": pred}


if __name__ == "__main__":
    B, Te, Td, F = 2, 5, 3, 4
    x = {
        "encoder_cont": torch.randn(B, Te, F),
        "decoder_cont": torch.randn(B, Td, F),
    }
    model = MiniTFT_Step5(hidden_size=8, output_size=2, x_reals=[f"f{i}" for i in range(F)])
    # 先不运行 forward，等 TODO 2/3 也填完再跑
    print("构造成功")
    out = model(x)
    print("pred shape:", out["prediction"].shape)  # 期望: (2, 3, 2)
    
