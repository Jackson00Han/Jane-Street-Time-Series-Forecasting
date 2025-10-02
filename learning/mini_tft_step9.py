import torch
from torch import nn

# 你已有的 SimpleGating / AddNorm 可沿用
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
        # 计算 z, g，并返回 u = sigmoid(g) * z =====
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
        # 先做残差相加，再做 LayerNorm =====
        sum = x + skip
        return self.norm(sum)
    
class MiniTFT_Step9(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, x_reals: list[str], static_size: int, num_heads: int = 2):
        super().__init__()
        self.H = hidden_size
        self.O = output_size
        self.x_reals = x_reals
        F = len(x_reals)
        self.static_size = static_size

        # ----- 与 Step8 一致 -----
        self.prescalers = nn.ModuleDict({name: nn.Linear(1, self.H) for name in x_reals})
        self.vsn_score_enc = nn.Linear(F, F, bias=False)
        self.vsn_score_dec = nn.Linear(F, F, bias=False)

        self.init_h = nn.Linear(static_size, self.H)
        self.init_c = nn.Linear(static_size, self.H)

        self.lstm_enc = nn.LSTM(input_size=self.H, hidden_size=self.H, batch_first=True)
        self.lstm_dec = nn.LSTM(input_size=self.H, hidden_size=self.H, batch_first=True)

        self.post_lstm_gate_dec = SimpleGating(self.H)
        self.post_lstm_addnorm_dec = AddNorm(self.H)

        # ----- Step9：注意力与后处理 -----
        self.mha = nn.MultiheadAttention(self.H, num_heads=num_heads, batch_first=True)
        self.post_attn_addnorm = AddNorm(self.H)   # 对应 PF 的 GateAddNorm(简化)
        self.pos_wise_ff = nn.Sequential(          # 对应 PF 的 GRN(简化版前馈)
            nn.Linear(self.H, self.H),
            nn.ReLU(),
            nn.Linear(self.H, self.H),
        )
        self.pre_output_addnorm = AddNorm(self.H)  # 对应 PF 的 pre_output_gate_norm(简化)

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

    def _build_causal_attn_mask(self, Te: int, Td: int, device) -> torch.Tensor:
        """
        目标：返回形状 (Td, Te+Td) 的布尔掩码，True 表示“禁止关注”。
        规则：
        - 左半 (Td, Te)：全 False（decoder 可看所有 encoder）
        - 右半 (Td, Td)：上三角（严格未来）为 True，包含当前步和过去为 False
        提示：用 torch.triu / torch.zeros 拼接。
        """
        # ===== TODO(你来写) =====
        # left = ...
        # right = ...
        # mask = torch.cat([left, right], dim=1)  # (Td, Te+Td)
        # return mask
        left = torch.zeros((Td, Te), dtype=torch.bool, device=device)
        right = torch.triu(torch.ones((Td, Td), dtype=torch.bool, device=device), diagonal=1)
        mask = torch.cat([left, right], dim=1)  # (Td, Te+Td)
        return mask

    def forward(self, x: dict) -> dict:
        enc, dec = x["encoder_cont"], x["decoder_cont"]      # (B,Te,F), (B,Td,F)
        B, Te, _ = enc.shape
        Td = dec.size(1)

        # 1) 静态 -> 初态
        static = torch.zeros(B, self.static_size, device=enc.device)
        h0 = self.init_h(static)           # (B,H)
        c0 = self.init_c(static)           # (B,H)

        # 2) VSN
        enc_h = self._project_with_vsn(enc, which="enc")     # (B,Te,H)
        dec_h = self._project_with_vsn(dec, which="dec")     # (B,Td,H)

        # 3) LSTM
        h0_l, c0_l = h0.unsqueeze(0), c0.unsqueeze(0)
        enc_out, (hT, cT) = self.lstm_enc(enc_h, (h0_l, c0_l))
        dec_out, _ = self.lstm_dec(dec_h, (hT, cT))          # (B,Td,H)

        # 4) Step8：post-LSTM（你已经实现）
        u = self.post_lstm_gate_dec(dec_out)                 # (B,Td,H)
        y = self.post_lstm_addnorm_dec(u, dec_h)             # (B,Td,H)

        # 5) Step9：拼接成注意力的 K/V 序列；Q 只用 decoder 段
        attn_input = torch.cat([enc_out, y], dim=1)          # (B,Te+Td,H)
        q = y                                                # (B,Td,H)
        k = attn_input                                       # (B,Te+Td,H)
        v = attn_input                                       # (B,Te+Td,H)

        # 6) 因果 mask（关键）
        attn_mask = self._build_causal_attn_mask(Te, Td, device=enc.device)  # (Td, Te+Td)
        # 注：nn.MultiheadAttention(batch_first=True) 期望 attn_mask 形状 (T_q, T_k)
        # True = 禁止关注

        # 7) 注意力前向
        attn_out, attn_weights = self.mha(q, k, v, attn_mask=attn_mask)  # attn_out: (B,Td,H)

        # 8) post-attn 残差归一化（skip 用 y）
        y_attn = self.post_attn_addnorm(attn_out, y)         # (B,Td,H)

        # 9) position-wise FF + pre-output 残差归一化
        ff = self.pos_wise_ff(y_attn)                        # (B,Td,H)
        fused = self.pre_output_addnorm(ff, y_attn)          # (B,Td,H)

        # 10) 输出头（decoder 段）
        pred = self.head(fused)                              # (B,Td,O)

        return {
            "prediction": pred,
            "attn_weights": attn_weights,  # 方便你后面可视化
        }


if __name__ == "__main__":
    B, Te, Td, F, S = 2, 5, 3, 4, 6
    x = {
        "encoder_cont": torch.randn(B, Te, F),
        "decoder_cont": torch.randn(B, Td, F),
    }
    model = MiniTFT_Step9(hidden_size=8, output_size=2, x_reals=[f"f{i}" for i in range(F)], static_size=S)
    out = model(x)
    #打印：
    print("pred:", out["prediction"].shape) 
