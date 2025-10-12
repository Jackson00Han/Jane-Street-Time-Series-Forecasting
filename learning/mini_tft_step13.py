import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ========== 小组件 ==========
class AddNorm(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.norm(x + skip)

class GateAddNorm(nn.Module):
    """
    更贴近 PF 的 GateAddNorm（简化版）：
      z = Wz(x)
      g = sigmoid(Wg(x))
      u = dropout(g * z)
      y = LayerNorm(u + skip)
    输入/输出: (B,T,H) 或 (B,H)（两者形状相容时都可）
    """
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.proj_z = nn.Linear(hidden_size, hidden_size)
        self.proj_g = nn.Linear(hidden_size, hidden_size)
        self.drop   = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.norm   = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # ⭐ TODO：补齐 GateAddNorm 的 4 步
        # 1) z = self.proj_z(x)
        # 2) g = torch.sigmoid(self.proj_g(x))
        # 3) u = self.drop(g * z)
        # 4) y = self.norm(u + skip)
        # return y
        z = self.proj_z(x)
        g = torch.sigmoid(self.proj_g(x))
        u = self.drop(g * z)
        y = self.norm(u + skip)
        return y


# ========== 主模型：Step13（GateAddNorm 取代门控+AddNorm） ==========
class MiniTFT_Step13(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        x_reals: list[str],
        static_size: int,
        num_heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.H = hidden_size
        self.O = output_size
        self.x_reals = x_reals
        self.static_size = static_size
        F = len(x_reals)
        self.dropout = dropout

        # prescalers + 两套 VSN
        self.prescalers = nn.ModuleDict({name: nn.Linear(1, self.H) for name in x_reals})
        self.vsn_score_enc = nn.Linear(F, F, bias=False)
        self.vsn_score_dec = nn.Linear(F, F, bias=False)

        # 静态 -> 初态
        self.init_h = nn.Linear(static_size, self.H)
        self.init_c = nn.Linear(static_size, self.H)

        # 编解码 LSTM
        self.lstm_enc = nn.LSTM(self.H, self.H, batch_first=True)
        self.lstm_dec = nn.LSTM(self.H, self.H, batch_first=True)

        # —— 静态上下文（沿用 Step12 的最小 GRN 思想，但为简洁，这里直接加线性占位）——
        self.static_ctx = nn.Linear(self.H, self.H)   # h0 -> s_ctx（占位）
        self.static_fuse = GateAddNorm(self.H, dropout=dropout)

        # 注意力 & 后处理
        self.mha = nn.MultiheadAttention(self.H, num_heads=num_heads, batch_first=True)
        self.post_attn_gan   = GateAddNorm(self.H, dropout=dropout)   # 替代 post_attn AddNorm
        self.pos_wise_ff     = nn.Sequential(nn.Linear(self.H, self.H), nn.ReLU(), nn.Linear(self.H, self.H))
        self.pre_output_gan  = GateAddNorm(self.H, dropout=dropout)   # 替代 pre_output AddNorm

        # —— 关键替换：LSTM 后（decoder）的 GateAddNorm —— #
        self.post_lstm_gan = GateAddNorm(self.H, dropout=dropout)     # 替代 SimpleGating + AddNorm

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
        left  = torch.zeros((Td, Te), dtype=torch.bool, device=device)
        right = torch.triu(torch.ones((Td, Td), dtype=torch.bool, device=device), diagonal=1)
        return torch.cat([left, right], dim=1)

    def _build_key_padding_mask(self, encoder_lengths, decoder_lengths, Te, Td, device):
        B = encoder_lengths.size(0)
        enc_idx = torch.arange(Te, device=device)[None, :].expand(B, -1)
        dec_idx = torch.arange(Td, device=device)[None, :].expand(B, -1)
        enc_pad = enc_idx >= encoder_lengths[:, None]
        dec_pad = dec_idx >= decoder_lengths[:, None]
        return torch.cat([enc_pad, dec_pad], dim=1)

    # LSTM with lengths（沿用 Step11）
    def _run_lstm_with_lengths(self, lstm, x_seq, lengths, h0, c0):
        pack = pack_padded_sequence(x_seq, lengths=lengths, batch_first=True, enforce_sorted=False)
        out_packed, (h, c) = lstm(pack, (h0.unsqueeze(0), c0.unsqueeze(0)))
        out, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=x_seq.size(1))
        return out, (h, c)

    def forward(self, x: dict) -> dict:
        """
        需要的 x：
          - encoder_cont: (B, Te, F)
          - decoder_cont: (B, Td, F)
          - encoder_lengths: (B,)
          - decoder_lengths: (B,)
          - （可选）static_cont: (B, S)
        """
        enc, dec = x["encoder_cont"], x["decoder_cont"]
        enc_len, dec_len = x["encoder_lengths"], x["decoder_lengths"]
        B, Te, _ = enc.shape
        Td = dec.size(1)
        device = enc.device

        # 1) 静态
        if "static_cont" in x:
            static_raw = x["static_cont"]
        else:
            static_raw = torch.zeros(B, self.static_size, device=device)
        h0 = self.init_h(static_raw)   # (B,H)
        c0 = self.init_c(static_raw)   # (B,H)
        s_ctx = self.static_ctx(h0)    # (B,H)  占位“精炼”静态

        # 2) VSN
        enc_h = self._project_with_vsn(enc, "enc")   # (B,Te,H)
        dec_h = self._project_with_vsn(dec, "dec")   # (B,Td,H)

        # 3) LSTM (pack/pad)
        enc_out, (hT, cT) = self._run_lstm_with_lengths(self.lstm_enc, enc_h, enc_len, h0, c0)
        dec_out, _        = self._run_lstm_with_lengths(self.lstm_dec, dec_h, dec_len, hT.squeeze(0), cT.squeeze(0))

        # 4) post-LSTM（decoder）：GateAddNorm(dec_out, skip=dec_h)
        y = self.post_lstm_gan(dec_out, dec_h)                 # (B,Td,H)

        # 5) 静态富集：把 s_ctx 注入 y（时间上平铺），再做一次 GateAddNorm 融合
        s_ctx_tiled = s_ctx[:, None, :].expand(-1, Td, -1)     # (B,Td,H)
        y = self.static_fuse(y, s_ctx_tiled)                   # (B,Td,H)

        # 6) 注意力
        attn_input = torch.cat([enc_out, y], dim=1)            # (B,Te+Td,H)
        q, k, v = y, attn_input, attn_input
        attn_mask = self._build_causal_attn_mask(Te, Td, device)
        key_padding_mask = self._build_key_padding_mask(enc_len, dec_len, Te, Td, device)
        attn_out, _ = self.mha(q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask)  # (B,Td,H)

        # 7) post-attn：GateAddNorm(attn_out, skip=y)
        y_attn = self.post_attn_gan(attn_out, y)               # (B,Td,H)

        # 8) FF + pre-output GateAddNorm
        ff    = self.pos_wise_ff(y_attn)                       # (B,Td,H)
        fused = self.pre_output_gan(ff, y_attn)                # (B,Td,H)

        # 9) head
        pred = self.head(fused)                                 # (B,Td,O)
        return {"prediction": pred}


# ========== 自测（填完 GateAddNorm.forward 后再跑） ==========
if __name__ == "__main__":
    torch.manual_seed(0)
    B, Te, Td, F, S, H, O = 2, 5, 3, 4, 6, 8, 2
    x = {
        "encoder_cont": torch.randn(B, Te, F),
        "decoder_cont": torch.randn(B, Td, F),
        "encoder_lengths": torch.tensor([Te, Te-1]),
        "decoder_lengths": torch.tensor([Td, Td-1]),
        # "static_cont": torch.randn(B, S),  # 可打开试试
    }
    model = MiniTFT_Step13(hidden_size=H, output_size=O, x_reals=[f"f{i}" for i in range(F)],
                           static_size=S, num_heads=2, dropout=0.0)
    out = model(x)   # <- 先实现 GateAddNorm.forward 再运行
    print("pred shape:", out["prediction"].shape)  # 期望: (2, 3, 2)
