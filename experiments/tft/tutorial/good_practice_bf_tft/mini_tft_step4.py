import torch
from torch import nn

class MiniTFT_Step4(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, x_reals: list[str]):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.x_reals = x_reals

        # 每列一个投影层: (B,T,1) -> (B,T,H)
        self.prescalers = nn.ModuleDict({
            name: nn.Linear(1, hidden_size) for name in x_reals
        })
        # 列打分层（最小实现）：(B,T,F) -> (B,T,F)
        self.vsn_score = nn.Linear(len(x_reals), len(x_reals), bias=False)

        self.head = nn.Linear(hidden_size, output_size)

    def project_reals(self, x_cont: torch.Tensor) -> torch.Tensor:
        B, T, F = x_cont.shape
        assert F == len(self.x_reals)

        # 逐列投影并堆叠: (B,T,H,F)
        outs = [self.prescalers[name](x_cont[..., j:j+1]) for j, name in enumerate(self.x_reals)]
        stacked = torch.stack(outs, dim=-1)  # (B, T, H, F)

        # TODO 1: 计算列权重 weights，形状 (B,T,F)，并在列维做 softmax
        # 提示：scores = self.vsn_score(x_cont)  # (B,T,F)
        scores = self.vsn_score(x_cont)  # (B,T,F)
        weights = torch.softmax(scores, dim=-1)  # (B,T,F)

        # TODO 2（下一步再做）: 用 weights 对 stacked 做加权和 -> (B,T,H)
        # h = (stacked * weights.unsqueeze(2)).sum(dim=-1)
        # return h
        h = (stacked * weights.unsqueeze(2)).sum(dim=-1)  

        
        return h

    def forward(self, x: dict) -> dict:
        dec_h = self.project_reals(x["decoder_cont"])  # (B, Td, H)
        pred = self.head(dec_h)                        # (B, Td, O)
        return {"prediction": pred}


if __name__ == "__main__":
    B, Te, Td, F = 2, 5, 3, 4
    x = {
        "encoder_cont": torch.randn(B, Te, F),
        "decoder_cont": torch.randn(B, Td, F),
    }
    model = MiniTFT_Step4(hidden_size=8, output_size=2, x_reals=[f"f{i}" for i in range(F)])
    out = model(x)
    print("pred shape:", out["prediction"].shape)  # 期望: (2, 3, 2)
    
