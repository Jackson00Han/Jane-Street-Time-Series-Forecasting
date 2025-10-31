import torch
from torch import nn

# ========== 小组件：门控 & AddNorm（留空给你实现） ==========
class SimpleGating(nn.Module):
    """
    目标：实现一个最小版的 GLU 风格门控。
    输入: x (B, T, H)
    输出: u (B, T, H)，形状与输入一致
    提示：用两条并行的 Linear(H->H)：一条做 Z，一条做 G（过 sigmoid），再逐点相乘。
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj_z = nn.Linear(hidden_size, hidden_size)
        self.proj_g = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ===== TODO(你来写): 计算 z, g，并返回 u = sigmoid(g) * z =====
        # z = ...
        # g = ...
        # u = ...
        z = self.proj_z(x)
        g = self.proj_g(x)
        u = torch.sigmoid(g) * z
        return u


class AddNorm(nn.Module):
    """
    目标：实现最小版 AddNorm：LayerNorm(x + skip)
    输入: x, skip 均为 (B, T, H)
    输出: y (B, T, H)
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # ===== TODO(你来写): 先做残差相加，再做 LayerNorm =====
        sum = x + skip
        return self.norm(sum)


# ========== 主模型：在 Step7 的基础上增加 post-LSTM 的 Gate + AddNorm ==========
class MiniTFT_Step8(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, x_reals: list[str], static_size: int):
        super().__init__()
        self.H = hidden_size
        self.O = output_size
        self.x_reals = x_reals
        F = len(x_reals)
        self.static_size = static_size

        # ---- 和 Step7 一致的部分 ----
        self.prescalers = nn.ModuleDict({name: nn.Linear(1, self.H) for name in x_reals})
        self.vsn_score_enc = nn.Linear(F, F, bias=False)
        self.vsn_score_dec = nn.Linear(F, F, bias=False)

        self.init_h = nn.Linear(static_size, self.H)
        self.init_c = nn.Linear(static_size, self.H)

        self.lstm_enc = nn.LSTM(input_size=self.H, hidden_size=self.H, batch_first=True)
        self.lstm_dec = nn.LSTM(input_size=self.H, hidden_size=self.H, batch_first=True)

        # ---- Step8 新增：post-LSTM（decoder分支）----
        self.post_lstm_gate_dec = SimpleGating(self.H)   # 门控 (dec_out -> u)
        self.post_lstm_addnorm_dec = AddNorm(self.H)     # AddNorm(u, dec_h) -> y

        # 输出头
        self.head = nn.Linear(self.H, self.O)

    @staticmethod
    def expand_static_context(context: torch.Tensor, timesteps: int) -> torch.Tensor:
        return context[:, None, :].expand(-1, timesteps, -1)

    def _project_with_vsn(self, x_cont: torch.Tensor, which: str) -> torch.Tensor:
        B, T, F = x_cont.shape
        outs = [self.prescalers[name](x_cont[..., j:j+1]) for j, name in enumerate(self.x_reals)]
        stacked = torch.stack(outs, dim=-1)                          # (B,T,H,F)
        scorer  = self.vsn_score_enc if which == "enc" else self.vsn_score_dec
        weights = torch.softmax(scorer(x_cont), dim=-1)              # (B,T,F)
        h = torch.matmul(stacked, weights.unsqueeze(-1)).squeeze(-1) # (B,T,H)
        return h

    def forward(self, x: dict) -> dict:
        enc, dec = x["encoder_cont"], x["decoder_cont"]      # (B,Te,F), (B,Td,F)
        B, Te, _ = enc.shape
        Td = dec.size(1)

        # 1) 静态 -> 初态
        static = torch.zeros(B, self.static_size, device=enc.device)
        h0 = self.init_h(static)           # (B,H)
        c0 = self.init_c(static)           # (B,H)

        # 2) VSN 投影
        enc_h = self._project_with_vsn(enc, which="enc")     # (B,Te,H)
        dec_h = self._project_with_vsn(dec, which="dec")     # (B,Td,H)

        # 3) LSTM 编解码
        h0_l, c0_l = h0.unsqueeze(0), c0.unsqueeze(0)        # (1,B,H)
        enc_out, (hT, cT) = self.lstm_enc(enc_h, (h0_l, c0_l))
        dec_out, _ = self.lstm_dec(dec_h, (hT, cT))          # (B,Td,H)

        # 4) Step8 关键：post-LSTM 的 Gate + AddNorm（与源码对齐：skip=dec_h）
        # ===== TODO(你来写) =====
        # u = self.post_lstm_gate_dec( ... )        # 门控：输入 dec_out
        # y = self.post_lstm_addnorm_dec( ..., ... )# 残差归一化：输入 u 和 dec_h
        u = self.post_lstm_gate_dec(dec_out)
        y = self.post_lstm_addnorm_dec(u, dec_h)
        print("u shape:", u.shape)  # (B,Td,H)

        # 5) 输出头（只在 decoder 段出结果）
        # ===== TODO(你来写) =====
        # pred = self.head( ... )   # 输入应是 y，形状 (B,Td,H) -> (B,Td,O)
        pred = self.head(y)

        # 6) （可选）形状校验—你可以自己加 assert/print
        # 例：assert pred.shape == (B, Td, self.O)

        return {
            "prediction": pred,   # ===== TODO: 返回 pred =====
            "h0": h0, "c0": c0,
        }


if __name__ == "__main__":
    B, Te, Td, F, S = 2, 5, 3, 4, 6
    x = {
        "encoder_cont": torch.randn(B, Te, F),
        "decoder_cont": torch.randn(B, Td, F),
    }
    model = MiniTFT_Step8(hidden_size=8, output_size=2, x_reals=[f"f{i}" for i in range(F)], static_size=S)
    out = model(x)
    # 你可以在填完 TODO 后，打印：
    print("pred:", out["prediction"].shape)  # 期望: (2, 3, 2)
