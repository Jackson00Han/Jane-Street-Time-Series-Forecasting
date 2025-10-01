import torch
from torch import nn

class MiniTFT_Step6(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, x_reals: list[str], static_size: int):
        super().__init__()
        self.H = hidden_size
        self.O = output_size
        self.x_reals = x_reals
        F = len(x_reals)
        self.prescalers = nn.ModuleDict({name: nn.Linear(1, self.H) for name in x_reals})
        self.vsn_score_enc = nn.Linear(F, F, bias=False)
        self.vsn_score_dec = nn.Linear(F, F, bias=False)
        self.head = nn.Linear(self.H, self.O)
        self.static_size = static_size

        # TODO 1: 新增两层，把静态特征 S 投到隐藏维 H，分别作为 h0/c0 的生成器
        self.init_h = nn.Linear(static_size, self.H)  # 例如：nn.Linear(static_size, self.H)
        self.init_c = nn.Linear(static_size, self.H)  # 例如：nn.Linear(static_size, self.H)

    @staticmethod
    def expand_static_context(context: torch.Tensor, timesteps: int) -> torch.Tensor:
        # (B,H) -> (B,T,H)
        return context[:, None, :].expand(-1, timesteps, -1)

    def _project_with_vsn(self, x_cont: torch.Tensor, which: str) -> torch.Tensor:
        B, T, F = x_cont.shape
        outs = [self.prescalers[name](x_cont[..., j:j+1]) for j, name in enumerate(self.x_reals)]
        stacked = torch.stack(outs, dim=-1)                 # (B,T,H,F)
        scorer  = self.vsn_score_enc if which == "enc" else self.vsn_score_dec
        weights = torch.softmax(scorer(x_cont), dim=-1)     # (B,T,F)
        h = torch.matmul(stacked, weights.unsqueeze(-1)).squeeze(-1)  # (B,T,H)
        return h

    def forward(self, x: dict) -> dict:
        enc, dec = x["encoder_cont"], x["decoder_cont"]      # (B,Te,F), (B,Td,F)
        B, Te, _ = enc.shape

        # TODO 2: 构造静态占位（(B,S)），可以用全零或随机；注意设备对齐 enc.device
        static = torch.zeros(B, self.static_size , device=enc.device)  # (B,S)

        # TODO 3: 通过 init_h/init_c 得到 (B,H) 的 h0/c0；再把 h0 扩展到 (B,Te,H)
        h0 = self.init_h(static)  # (B,H)
        c0 = self.init_c(static)  # (B,H)
        h0_time = self.expand_static_context(h0, Te)  # 用 expand_static_context(h0, Te)

        # 正常的 decoder 路径（沿用 Step5）
        dec_h = self._project_with_vsn(dec, which="dec")     # (B,Td,H)
        pred  = self.head(dec_h)                             # (B,Td,O)
        return {"prediction": pred, "h0_time": h0_time, "h0": h0, "c0": c0}


if __name__ == "__main__":
    B, Te, Td, F, S = 2, 5, 3, 4, 6
    x = {
        "encoder_cont": torch.randn(B, Te, F),
        "decoder_cont": torch.randn(B, Td, F),
    }
    model = MiniTFT_Step6(hidden_size=8, output_size=2, x_reals=[f"f{i}" for i in range(F)], static_size=S)
    out = model(x)
    print("pred:", out["prediction"].shape)  # 期望: (2, 3, 2)
    print("h0:",   out["h0"].shape)          # 期望: (2, 8)
    print("h0_time:", out["h0_time"].shape)  # 期望: (2, 5, 8)
