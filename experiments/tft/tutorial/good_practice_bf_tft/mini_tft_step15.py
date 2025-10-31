import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ========== 基础小件 ==========
class GateAddNorm(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.proj_z = nn.Linear(hidden_size, hidden_size)
        self.proj_g = nn.Linear(hidden_size, hidden_size)
        self.drop   = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.norm   = nn.LayerNorm(hidden_size)
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        z = self.proj_z(x)
        g = torch.sigmoid(self.proj_g(x))
        u = self.drop(g * z)
        return self.norm(u + skip)

# ========== Pinball（Quantile）Loss ==========
def pinball_loss(pred_q: torch.Tensor, y: torch.Tensor, quantiles: list[float]) -> torch.Tensor:
    """
    pred_q: (B, Td, Q)
    y:      (B, Td) 或 (B, Td, 1)
    返回标量损失（对 B,Td,Q 平均）
    """
    Q = len(quantiles)
    if y.dim() == 2:
        y = y.unsqueeze(-1)                        # (B,Td,1)
    assert pred_q.shape[-1] == Q

    # ⭐ TODO #1：实现 pinball（逐分位）
    # 提示：
    #   e = y - pred_q
    #   对每个 q：loss_q = max(q*e, (q-1)*e) = torch.maximum(q*e, (q-1)*e)
    #   汇总后对 (B,Td,Q) 取 mean
    #   注意：quantiles 是 Python 列表，可做形状对齐：torch.tensor(quantiles, device=...)[None,None,:]
    e = y - pred_q
    loss_q = torch.maximum(torch.tensor(quantiles, device=pred_q.device)[None, None, :] * e,
                           (torch.tensor(quantiles, device=pred_q.device)[None, None, :] - 1) * e)
    return loss_q.mean()
    

# ========== Step15 主模型：多分位输出 ==========
class MiniTFT_Step15(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        quantiles: list[float],            # e.g. [0.1, 0.5, 0.9]
        x_reals: list[str],
        static_size: int,
        num_heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.H = hidden_size
        self.Q = len(quantiles)
        self.quantiles = quantiles
        self.x_reals = x_reals
        self.static_size = static_size
        F = len(x_reals)

        # prescalers + 两套 VSN 打分层
        self.prescalers = nn.ModuleDict({name: nn.Linear(1, self.H) for name in x_reals})
        self.vsn_score_enc = nn.Linear(F, F, bias=False)
        self.vsn_score_dec = nn.Linear(F, F, bias=False)

        # 静态 -> 初态
        self.init_h = nn.Linear(static_size, self.H)
        self.init_c = nn.Linear(static_size, self.H)

        # 编解码 LSTM
        self.lstm_enc = nn.LSTM(self.H, self.H, batch_first=True)
        self.lstm_dec = nn.LSTM(self.H, self.H, batch_first=True)

        # post-LSTM（decoder）
        self.post_lstm_gan = GateAddNorm(self.H, dropout=dropout)

        # 注意力与后处理
        self.mha = nn.MultiheadAttention(self.H, num_heads=num_heads, batch_first=True)
        self.post_attn_gan  = GateAddNorm(self.H, dropout=dropout)
        self.pos_wise_ff    = nn.Sequential(nn.Linear(self.H, self.H), nn.ReLU(), nn.Linear(self.H, self.H))
        self.pre_output_gan = GateAddNorm(self.H, dropout=dropout)

        # 输出头：H -> Q（分位点数）
        self.head = nn.Linear(self.H, self.Q)

    @staticmethod
    def expand_static_context(context: torch.Tensor, timesteps: int) -> torch.Tensor:
        return context[:, None, :].expand(-1, timesteps, -1)

    def _project_with_vsn(self, x_cont: torch.Tensor, which: str, return_weights: bool = False):
        B, T, F = x_cont.shape
        outs = [self.prescalers[name](x_cont[..., j:j+1]) for j, name in enumerate(self.x_reals)]
        stacked = torch.stack(outs, dim=-1)                          # (B,T,H,F)
        scorer  = self.vsn_score_enc if which == "enc" else self.vsn_score_dec
        weights = torch.softmax(scorer(x_cont), dim=-1)              # (B,T,F)
        h = torch.matmul(stacked, weights.unsqueeze(-1)).squeeze(-1) # (B,T,H)
        if return_weights:
            return h, weights
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

    # LSTM with lengths
    def _run_lstm_with_lengths(self, lstm, x_seq, lengths, h0, c0):
        pack = pack_padded_sequence(x_seq, lengths=lengths, batch_first=True, enforce_sorted=False)
        out_packed, (h, c) = lstm(pack, (h0.unsqueeze(0), c0.unsqueeze(0)))
        out, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=x_seq.size(1))
        return out, (h, c)

    def forward(self, x: dict) -> dict:
        """
        x 需要：
          encoder_cont: (B, Te, F)
          decoder_cont: (B, Td, F)
          encoder_lengths: (B,)
          decoder_lengths: (B,)
          target (可选，用于直接算损失): (B, Td)
        """
        enc, dec = x["encoder_cont"], x["decoder_cont"]
        enc_len, dec_len = x["encoder_lengths"], x["decoder_lengths"]
        B, Te, _ = enc.shape
        Td = dec.size(1)
        device = enc.device

        # 1) 静态 → 初态
        static_raw = x.get("static_cont", torch.zeros(B, self.static_size, device=device))
        h0, c0 = self.init_h(static_raw), self.init_c(static_raw)

        # 2) VSN
        enc_h = self._project_with_vsn(enc, "enc")
        dec_h = self._project_with_vsn(dec, "dec")

        # 3) LSTM（pack/pad）
        enc_out, (hT, cT) = self._run_lstm_with_lengths(self.lstm_enc, enc_h, enc_len, h0, c0)
        dec_out, _        = self._run_lstm_with_lengths(self.lstm_dec, dec_h, dec_len, hT.squeeze(0), cT.squeeze(0))

        # 4) post-LSTM（decoder）
        y = self.post_lstm_gan(dec_out, dec_h)                     # (B,Td,H)

        # 5) 注意力
        attn_input = torch.cat([enc_out, y], dim=1)                # (B,Te+Td,H)
        q, k, v = y, attn_input, attn_input
        attn_mask = self._build_causal_attn_mask(Te, Td, device)
        key_padding_mask = self._build_key_padding_mask(enc_len, dec_len, Te, Td, device)
        attn_out, attn_w = self.mha(q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask, average_attn_weights=False)
        # attn_w: (B, num_heads, Td, Te+Td)

        # 6) post-attn & FF & pre-output
        y_attn = self.post_attn_gan(attn_out, y)                   # (B,Td,H)
        ff     = self.pos_wise_ff(y_attn)
        fused  = self.pre_output_gan(ff, y_attn)                   # (B,Td,H)

        # 7) 分位输出
        pred_q = self.head(fused)                                  # (B,Td,Q)

        out = {
            "prediction": pred_q,          # 分位预测 (B,Td,Q)
            "attn_weights": attn_w,        # (B, num_heads, Td, Te+Td)
            "attn_mask": attn_mask,
            "key_padding_mask": key_padding_mask,
        }

        # 8) 若传入 target，就计算 pinball 损失
        if "target" in x:
            # ⭐ TODO #2：调用 pinball_loss 计算损失，并放到 out["loss"]
            # 例：out["loss"] = pinball_loss(pred_q, x["target"], self.quantiles)
            out["loss"] = pinball_loss(pred_q, x["target"], self.quantiles)
        return out


# ========== 自测（填完两个 TODO 后再跑） ==========
if __name__ == "__main__":
    torch.manual_seed(0)
    B, Te, Td, F, S, H = 2, 5, 3, 4, 6, 8
    Qs = [0.1, 0.5, 0.9]
    x = {
        "encoder_cont": torch.randn(B, Te, F),
        "decoder_cont": torch.randn(B, Td, F),
        "encoder_lengths": torch.tensor([Te, Te-1]),
        "decoder_lengths": torch.tensor([Td, Td-1]),
        "target": torch.randn(B, Td),
    }
    model = MiniTFT_Step15(hidden_size=H, quantiles=Qs, x_reals=[f"f{i}" for i in range(F)],
                           static_size=S, num_heads=2, dropout=0.1)
    out = model(x)   # <- 先补 TODO
    print("pred shape:", out["prediction"].shape)  # 期望: (2, 3, 3)
    print("loss:", out["loss"].item())
