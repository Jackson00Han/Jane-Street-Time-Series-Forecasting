import torch
from torch import nn

class MiniTFT_Step3(nn.Module):
    """
    Step3：加入 prescalers，把每个连续特征投影到隐藏维，再只对 decoder 段输出
    """
    def __init__(self, hidden_size: int, output_size: int, x_reals: list[str]):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.x_reals = x_reals

        # TODO 1: 每个连续特征一层 Linear(1 -> H)
        self.prescalers = nn.ModuleDict({
            name: nn.Linear(1, hidden_size) for name in x_reals  # name: 对应的线性层
        })

        # 小输出头 H -> O
        self.head = nn.Linear(hidden_size, output_size)

    def project_reals(self, x_cont: torch.Tensor) -> torch.Tensor:
        """
        x_cont: (B, T, F)  ->  (B, T, H)
        """
        B, T, F = x_cont.shape
        assert F == len(self.x_reals), f"列数 {F} 与 x_reals {len(self.x_reals)} 不一致"

        outs = []
        for j, name in enumerate(self.x_reals):
            col = x_cont[..., j:j+1]              # (B,T,1)
            # TODO 2: 通过 prescalers[name] 投影到 (B,T,H)
            out = self.prescalers[name](col)  # (B,T,H)
            outs.append(out)

        # TODO 3: 把各列投影相加为 (B,T,H)
        h = torch.stack(outs, dim=-1).sum(dim=-1)  # (B,T,H)
        return h

    def forward(self, x: dict) -> dict:
        dec = x["decoder_cont"]                   # (B, Td, F)
        dec_h = self.project_reals(dec)           # (B, Td, H)
        pred = self.head(dec_h)                   # (B, Td, O)
        return {"prediction": pred}


if __name__ == "__main__":
    B, Te, Td, F = 2, 5, 3, 4
    x = {
        "encoder_cont": torch.randn(B, Te, F),
        "decoder_cont": torch.randn(B, Td, F),
    }
    model = MiniTFT_Step3(hidden_size=8, output_size=2, x_reals=[f"f{i}" for i in range(F)])
    out = model(x)
    print("pred shape:", out["prediction"].shape)  # 期望: (2, 3, 2)
