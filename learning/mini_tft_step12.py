import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ========== 小组件 ==========
class SimpleGating(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj_z = nn.Linear(hidden_size, hidden_size)
        self.proj_g = nn.Linear(hidden_size, hidden_size)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj_g(x)) * self.proj_z(x)

class AddNorm(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.norm(x + skip)

# 最小 GRN：带可选“上下文注入”的前馈（简化版）
class MiniGRN(nn.Module):
    """
    y = AddNorm( FFN(x + C(context)), x )
    这里 C(context) 先线性到 H，再与 x 相加后过 ReLU+Linear。
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.ctx = nn.Linear(hidden_size, hidden_size)
        self.f1  = nn.Linear(hidden_size, hidden_size)
        self.f2  = nn.Linear(hidden_size, hidden_size)
        self.addnorm = AddNorm(hidden_size)
        

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        x: (B,T,H) 或 (B,H)；context: 同 H 维（(B,T,H) 或 (B,H)）
        要求：在进行逐元素相加前确保形状对齐（必要时扩展时间维）。
        """
        # ⭐ TODO #1：实现 GRN 主体
        # 提示：
        #  - 若 context 是 (B,H) 而 x 是 (B,T,H)，先扩展成 (B,T,H)
        #  - z = x + ctx(context)
        #  - h = f2( ReLU( f1(z) ) )
        #  - y = AddNorm(h, x)
        if context.dim() + 1 == x.dim():  # (B,H) -> (B,T,H)
            context = context[:, None, :].expand(-1, x.size(1), -1)
            
        z = x + self.ctx(context)
        h = self.f2(torch.relu(self.f1(z)))
        y = self.addnorm(h, x)
        return y

# ========== 主模型：Step12（静态上下文富集 + 现有全部功能） ==========
class MiniTFT_Step12(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        x_reals: list[str],
        static_size: int,
        num_heads: int = 2,
    ):
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

        # 静态 -> 初态（占位：线性）
        self.init_h = nn.Linear(static_size, self.H)
        self.init_c = nn.Linear(static_size, self.H)

        # 编解码 LSTM
        self.lstm_enc = nn.LSTM(self.H, self.H, batch_first=True)
        self.lstm_dec = nn.LSTM(self.H, self.H, batch_first=True)

        # post-LSTM（decoder）：门控 + AddNorm（skip=dec_h）
        self.post_lstm_gate_dec    = SimpleGating(self.H)
        self.post_lstm_addnorm_dec = AddNorm(self.H)

        # —— Step12：静态上下文富集 —— #
        self.static_ctx_grn  = MiniGRN(self.H)  # s -> s_ctx（(B,H) 或 (B,T,H)）
        self.static_fuse_grn = MiniGRN(self.H)  # 融入到时序表示（对 encoder/decoder 均可用）

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

    # LSTM with lengths（延用你 Step11 写法）
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
        - （可选）static_cont: (B, S)  # 若不给，我们用 0 向量占位
        """
        enc, dec = x["encoder_cont"], x["decoder_cont"]
        enc_len, dec_len = x["encoder_lengths"], x["decoder_lengths"]
        B, Te, _ = enc.shape
        Td = dec.size(1)
        device = enc.device

        # 1) 静态向量（占位或使用传入）
        if "static_cont" in x:
            static_raw = x["static_cont"]                          # (B, S)
        else:
            static_raw = torch.zeros(B, self.static_size, device=device)
        h0 = self.init_h(static_raw)                               # (B,H)
        c0 = self.init_c(static_raw)                               # (B,H)

        # —— Step12：做一个静态上下文 —— #
        # s_ctx: (B,H) 先不过时间维
        # ⭐ TODO #2：用 self.static_ctx_grn 把 h0 “精炼”成 s_ctx（注意形状要求）
        # 提示：MiniGRN 可以接 (B,H)，此时不需要扩时间维；也可以你先让它接 (B,H)，后面再 expand。
        s_ctx = self.static_ctx_grn(h0, h0)                       # (B,H)

        # 2) VSN
        enc_h = self._project_with_vsn(enc, "enc")                 # (B,Te,H)
        dec_h = self._project_with_vsn(dec, "dec")                 # (B,Td,H)

        # 3) LSTM（pack/pad）
        enc_out, (hT, cT) = self._run_lstm_with_lengths(self.lstm_enc, enc_h, enc_len, h0, c0)
        dec_out, _        = self._run_lstm_with_lengths(self.lstm_dec, dec_h, dec_len, hT.squeeze(0), cT.squeeze(0))

        # 4) post-LSTM（decoder）
        u = self.post_lstm_gate_dec(dec_out)                       # (B,Td,H)
        y = self.post_lstm_addnorm_dec(u, dec_h)                   # (B,Td,H)

        # —— Step12：把 s_ctx 融入到时序表示（这里先对 decoder 融入）——
        # 做法：把 (B,H) 的 s_ctx 扩到 (B,Td,H)，再用 static_fuse_grn 融合到 y
        # 最终：y_se = static_fuse_grn(y, s_ctx_tiled)  # 形状仍 (B,Td,H)
        s_ctx_tiled = s_ctx[:, None, :].expand(-1, Td, -1)         # (B,Td,H)
        y_se = self.static_fuse_grn(y, s_ctx_tiled)                # (B,Td,H)

        # 5) 注意力（Q 用融合后的 y_se；K/V 仍用 enc_out 与 y_se 拼接）
        attn_input = torch.cat([enc_out, y_se], dim=1)             # (B,Te+Td,H)
        q, k, v = y_se, attn_input, attn_input

        attn_mask = self._build_causal_attn_mask(Te, Td, device)
        key_padding_mask = self._build_key_padding_mask(enc_len, dec_len, Te, Td, device)
        attn_out, _ = self.mha(q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        # 6) post-attn & FFN
        y_attn = self.post_attn_addnorm(attn_out, y_se)            # (B,Td,H)
        ff     = self.pos_wise_ff(y_attn)
        fused  = self.pre_output_addnorm(ff, y_attn)
        pred   = self.head(fused)                                   # (B,Td,O)
        return {"prediction": pred}


# ========== 自测（填完两个 TODO 后再跑） ==========
if __name__ == "__main__":
    torch.manual_seed(0)
    B, Te, Td, F, S, H, O = 2, 5, 3, 4, 6, 8, 2
    x = {
        "encoder_cont": torch.randn(B, Te, F),
        "decoder_cont": torch.randn(B, Td, F),
        "encoder_lengths": torch.tensor([Te, Te-1]),
        "decoder_lengths": torch.tensor([Td, Td-1]),
        # 也可试试提供静态： "static_cont": torch.randn(B, S),
    }
    model = MiniTFT_Step12(hidden_size=H, output_size=O, x_reals=[f"f{i}" for i in range(F)], static_size=S, num_heads=2)
    out = model(x)
    print("pred:", out["prediction"].shape)  # 期望: (2, 3, 2)
