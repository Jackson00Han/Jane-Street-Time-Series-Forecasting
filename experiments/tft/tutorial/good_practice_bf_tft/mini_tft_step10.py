import torch
from torch import nn

# ========== 小组件：门控 & AddNorm ==========
class SimpleGating(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj_z = nn.Linear(hidden_size, hidden_size)
        self.proj_g = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj_z(x)
        g = torch.sigmoid(self.proj_g(x))
        return g * z


class AddNorm(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.norm(x + skip)


# ========== 主模型：Step10（加入 key_padding_mask） ==========
class MiniTFT_Step10(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, x_reals: list[str], static_size: int, num_heads: int = 2):
        super().__init__()
        self.H = hidden_size
        self.O = output_size
        self.x_reals = x_reals
        self.static_size = static_size
        F = len(x_reals)

        # prescalers + 两套 VSN
        self.prescalers = nn.ModuleDict({name: nn.Linear(1, self.H) for name in x_reals})
        self.vsn_score_enc = nn.Linear(F, F, bias=False)
        self.vsn_score_dec = nn.Linear(F, F, bias=False)

        # 静态 -> LSTM 初态
        self.init_h = nn.Linear(static_size, self.H)
        self.init_c = nn.Linear(static_size, self.H)

        # 编解码 LSTM
        self.lstm_enc = nn.LSTM(self.H, self.H, batch_first=True)
        self.lstm_dec = nn.LSTM(self.H, self.H, batch_first=True)

        # post-LSTM（decoder）：门控 + AddNorm（skip=dec_h）
        self.post_lstm_gate_dec = SimpleGating(self.H)
        self.post_lstm_addnorm_dec = AddNorm(self.H)

        # 注意力与后处理
        self.mha = nn.MultiheadAttention(self.H, num_heads=num_heads, batch_first=True)
        self.post_attn_addnorm = AddNorm(self.H)
        self.pos_wise_ff = nn.Sequential(nn.Linear(self.H, self.H), nn.ReLU(), nn.Linear(self.H, self.H))
        self.pre_output_addnorm = AddNorm(self.H)

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

    def _build_causal_attn_mask(self, Te: int, Td: int, device) -> torch.Tensor:
        """
        (Td, Te+Td) 布尔掩码；True=禁止关注。
          左半 (Td, Te)：全 False；右半 (Td, Td)：严格未来 True。
        """
        left  = torch.zeros((Td, Te), dtype=torch.bool, device=device)
        right = torch.triu(torch.ones((Td, Td), dtype=torch.bool, device=device), diagonal=1)
        return torch.cat([left, right], dim=1)

    def _build_key_padding_mask(
        self,
        encoder_lengths: torch.Tensor,  # (B,)
        decoder_lengths: torch.Tensor,  # (B,)
        Te: int, Td: int, device
    ) -> torch.Tensor:
        """
        ⭐ TODO：实现 (B, Te+Td) 的布尔掩码，True=该样本该位置是 padding（禁止被关注）。
        规则：每行 [0:len) 有效(False)，[len:T) 为 padding(True)。
        小提示：用 arange + 比较：
            enc_idx = torch.arange(Te, device=device)[None, :].expand(B, -1)
            enc_pad = enc_idx >= encoder_lengths[:, None]    # (B,Te)
            dec_idx = torch.arange(Td, device=device)[None, :].expand(B, -1)
            dec_pad = dec_idx >= decoder_lengths[:, None]    # (B,Td)
        """
        B = encoder_lengths.size(0)
        # ===== TODO(你来写) =====
        enc_idx = torch.arange(Te, device=device)[None, :].expand(B, -1)
        enc_pad = enc_idx >= encoder_lengths[:, None]    # (B,Te)
        dec_idx = torch.arange(Td, device=device)[None, :].expand(B, -1)
        dec_pad = dec_idx >= decoder_lengths[:, None]    # (B,Td)
        mask = torch.cat([enc_pad, dec_pad], dim=1)  # (B, Te+Td)
        return mask
        

    def forward(self, x: dict) -> dict:
        """
        期望的 x 键：
        - encoder_cont: (B, Te, F)
        - decoder_cont: (B, Td, F)
        - encoder_lengths: (B,)
        - decoder_lengths: (B,)
        """
        enc, dec = x["encoder_cont"], x["decoder_cont"]
        enc_len, dec_len = x["encoder_lengths"], x["decoder_lengths"]
        B, Te, _ = enc.shape
        Td = dec.size(1)

        # 1) 静态 -> 初态
        static = torch.zeros(B, self.static_size, device=enc.device)
        h0, c0 = self.init_h(static), self.init_c(static)

        # 2) VSN
        enc_h = self._project_with_vsn(enc, "enc")  # (B,Te,H)
        dec_h = self._project_with_vsn(dec, "dec")  # (B,Td,H)

        # 3) LSTM（本步不做 pack/pad）
        enc_out, (hT, cT) = self.lstm_enc(enc_h, (h0.unsqueeze(0), c0.unsqueeze(0)))  # (B,Te,H)
        dec_out, _         = self.lstm_dec(dec_h, (hT, cT))                            # (B,Td,H)

        # 4) post-LSTM
        u = self.post_lstm_gate_dec(dec_out)           # (B,Td,H)
        y = self.post_lstm_addnorm_dec(u, dec_h)       # (B,Td,H)

        # 5) 注意力输入
        attn_input = torch.cat([enc_out, y], dim=1)    # (B,Te+Td,H)
        q, k, v = y, attn_input, attn_input

        # 6) 两类 mask
        attn_mask = self._build_causal_attn_mask(Te, Td, device=enc.device)  # (Td, Te+Td)
        key_padding_mask = self._build_key_padding_mask(enc_len, dec_len, Te, Td, enc.device)  # (B,Te+Td)

        # 7) 注意力前向
        # ⭐ TODO：把 key_padding_mask 传进去（并保留 attn_mask）
        # 形状要求：
        #   - q: (B,Td,H), k: (B,Te+Td,H), v 同 k
        #   - attn_mask: (Td, Te+Td) 布尔，True=禁止
        #   - key_padding_mask: (B, Te+Td) 布尔，True=该样本该位置禁止
        # attn_out, attn_weights = self.mha(q, k, v, attn_mask=..., key_padding_mask=...)
        # return 后面要用 attn_out
        attn_out, attn_weights = self.mha(q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        # 8) post-attn 残差归一化 & 9) 前馈 + pre-output AddNorm & 10) head
        y_attn = self.post_attn_addnorm(attn_out, y)  # (B,Td,H)
        ff     = self.pos_wise_ff(y_attn)
        fused  = self.pre_output_addnorm(ff, y_attn)
        pred   = self.head(fused)
        return {"prediction": pred}
        
        

# ========== 自测（填完 TODO 后再跑） ==========
if __name__ == "__main__":
    torch.manual_seed(0)
    B, Te, Td, F, S, H, O = 2, 5, 3, 4, 6, 8, 2

    x = {
        "encoder_cont": torch.randn(B, Te, F),
        "decoder_cont": torch.randn(B, Td, F),
        # 两条样本长度不同：第 2 条在 enc/dec 各少 1 步
        "encoder_lengths": torch.tensor([Te, Te-1]),
        "decoder_lengths": torch.tensor([Td, Td-1]),
    }

    model = MiniTFT_Step10(hidden_size=H, output_size=O, x_reals=[f"f{i}" for i in range(F)], static_size=S, num_heads=2)
    # 你可以在完成 TODO 后取消注释：
    out = model(x)
    print("pred shape:", out["prediction"].shape)  # 期望: (2, 3, 2)
