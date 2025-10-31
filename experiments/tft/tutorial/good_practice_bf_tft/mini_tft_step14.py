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

# ========== Step14 主模型：加“解释性输出” ==========
class MiniTFT_Step14(nn.Module):
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

        # 输出头
        self.head = nn.Linear(self.H, self.O)

    @staticmethod
    def expand_static_context(context: torch.Tensor, timesteps: int) -> torch.Tensor:
        return context[:, None, :].expand(-1, timesteps, -1)

    def _project_with_vsn(self, x_cont: torch.Tensor, which: str, return_weights: bool = False):
        """
        连续特征：逐列 Linear(1->H) 投影 -> 堆叠 (B,T,H,F) -> 按列 softmax 权重 (B,T,F) -> 加权合成 (B,T,H)
        若 return_weights=True，额外返回权重，便于解释。
        """
        B, T, F = x_cont.shape
        outs = [self.prescalers[name](x_cont[..., j:j+1]) for j, name in enumerate(self.x_reals)]
        stacked = torch.stack(outs, dim=-1)                          # (B,T,H,F)
        scorer  = self.vsn_score_enc if which == "enc" else self.vsn_score_dec
        weights = torch.softmax(scorer(x_cont), dim=-1)              # (B,T,F)
        h = torch.matmul(stacked, weights.unsqueeze(-1)).squeeze(-1) # (B,T,H)
        # ⭐ TODO #1：当 return_weights=True，返回 (h, weights)，否则只返回 h
        # 提示：保持兼容：if return_weights: return h, weights; else: return h
        if return_weights:
            return h, weights
        else:
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
        h0 = self.init_h(static_raw); c0 = self.init_c(static_raw)

        # 2) VSN（本步要求：拿到时序隐表示 + 列权重）
        # enc_h, enc_vars = ...
        # dec_h, dec_vars = ...
        # ⭐ TODO #2 (part A)：调用 _project_with_vsn(..., return_weights=True) 拿到 (h, weights)
        enc_h, enc_vars = self._project_with_vsn(enc, "enc", return_weights=True)   # (B,Te,H), (B,Te,F)
        dec_h, dec_vars = self._project_with_vsn(dec, "dec", return_weights=True)   # (B,Td,H), (B,Td,F)

        # 3) LSTM
        enc_out, (hT, cT) = self._run_lstm_with_lengths(self.lstm_enc, enc_h, enc_len, h0, c0)
        dec_out, _        = self._run_lstm_with_lengths(self.lstm_dec, dec_h, dec_len, hT.squeeze(0), cT.squeeze(0))

        # 4) post-LSTM（decoder）
        y = self.post_lstm_gan(dec_out, dec_h)

        # 5) 注意力
        attn_input = torch.cat([enc_out, y], dim=1)
        q, k, v = y, attn_input, attn_input
        attn_mask = self._build_causal_attn_mask(Te, Td, device)
        key_padding_mask = self._build_key_padding_mask(enc_len, dec_len, Te, Td, device)

        # 我们希望拿到“逐头”的注意力（更利于解释），因此需要 average_attn_weights=False
        # ⭐ TODO #2 (part B)：把 average_attn_weights=False 传入，获取 per-head 权重
        # 期望形状：attn_w ~ (B, num_heads, Td, Te+Td)
        attn_out, attn_w = self.mha(q, k, v, attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                average_attn_weights=False)  # attn_out: (B,Td,H); attn_w: (B, num_heads, Td, Te+Td)

        # 6) post-attn & FF
        y_attn = self.post_attn_gan(attn_out, y)
        ff     = self.pos_wise_ff(y_attn)
        fused  = self.pre_output_gan(ff, y_attn)

        # 7) head
        pred = self.head(fused)

        # 8) 返回解释性输出（最小版）
        return {
            "prediction": pred,                 # (B, Td, O)
            "encoder_variables": enc_vars,      # (B, Te, F)
            "decoder_variables": dec_vars,      # (B, Td, F)
            "attn_weights": attn_w,            # (B, num_heads, Td, Te+Td)
            "attn_mask": attn_mask,            # (Td, Te+Td)
            "key_padding_mask": key_padding_mask,  # (B, Te+Td)
        }


# ========== 自测（你填完两个 TODO 后再跑） ==========
if __name__ == "__main__":
    torch.manual_seed(0)
    B, Te, Td, F, S, H, O = 2, 5, 3, 4, 6, 8, 2
    x = {
        "encoder_cont": torch.randn(B, Te, F),
        "decoder_cont": torch.randn(B, Td, F),
        "encoder_lengths": torch.tensor([Te, Te-1]),
        "decoder_lengths": torch.tensor([Td, Td-1]),
    }
    model = MiniTFT_Step14(hidden_size=H, output_size=O, x_reals=[f"f{i}" for i in range(F)],
                           static_size=S, num_heads=2, dropout=0.1)
    out = model(x)  # <- 填完 TODO 再运行
    print(out["prediction"].shape, out["encoder_variables"].shape, out["decoder_variables"].shape)
