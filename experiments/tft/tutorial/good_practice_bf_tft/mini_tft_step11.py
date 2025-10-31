import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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


# ========== 主模型：Step11（LSTM pack/pad + 注意力 masks） ==========
class MiniTFT_Step11(nn.Module):
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

        # prescalers + 两套 VSN 打分层
        self.prescalers = nn.ModuleDict({name: nn.Linear(1, self.H) for name in x_reals})
        self.vsn_score_enc = nn.Linear(F, F, bias=False)
        self.vsn_score_dec = nn.Linear(F, F, bias=False)

        # 静态 -> 初态（简化为线性）
        self.init_h = nn.Linear(static_size, self.H)
        self.init_c = nn.Linear(static_size, self.H)

        # 编解码 LSTM
        self.lstm_enc = nn.LSTM(self.H, self.H, batch_first=True)
        self.lstm_dec = nn.LSTM(self.H, self.H, batch_first=True)

        # post-LSTM（decoder 分支）：门控 + AddNorm（skip=dec_h）
        self.post_lstm_gate_dec = SimpleGating(self.H)
        self.post_lstm_addnorm_dec = AddNorm(self.H)

        # 注意力与后处理
        self.mha = nn.MultiheadAttention(self.H, num_heads=num_heads, batch_first=True)
        self.post_attn_addnorm = AddNorm(self.H)  # 对齐 PF 的 GateAddNorm（简化）
        self.pos_wise_ff = nn.Sequential(         # 对齐 PF 的 GRN（简化）
            nn.Linear(self.H, self.H),
            nn.ReLU(),
            nn.Linear(self.H, self.H),
        )
        self.pre_output_addnorm = AddNorm(self.H) # 对齐 PF 的 pre_output_gate_norm（简化）

        # 输出头（只在 decoder 段出）
        self.head = nn.Linear(self.H, self.O)

    @staticmethod
    def expand_static_context(context: torch.Tensor, timesteps: int) -> torch.Tensor:
        return context[:, None, :].expand(-1, timesteps, -1)

    def _project_with_vsn(self, x_cont: torch.Tensor, which: str) -> torch.Tensor:
        """
        连续特征：逐列 Linear(1->H) 投影 -> 堆叠 (B,T,H,F) -> 按列 softmax 权重 (B,T,F) -> 加权合成 (B,T,H)
        """
        B, T, F = x_cont.shape
        outs = [self.prescalers[name](x_cont[..., j:j+1]) for j, name in enumerate(self.x_reals)]
        stacked = torch.stack(outs, dim=-1)                          # (B,T,H,F)
        scorer  = self.vsn_score_enc if which == "enc" else self.vsn_score_dec
        weights = torch.softmax(scorer(x_cont), dim=-1)              # (B,T,F)
        h = torch.matmul(stacked, weights.unsqueeze(-1)).squeeze(-1) # (B,T,H)
        return h

    def _build_causal_attn_mask(self, Te: int, Td: int, device) -> torch.Tensor:
        """
        返回 (Td, Te+Td) 布尔掩码；True=禁止关注。
          - 左半 (Td, Te)：全 False（decoder 可看所有 encoder）
          - 右半 (Td, Td)：严格未来为 True（上三角, diagonal=1）
        """
        left  = torch.zeros((Td, Te), dtype=torch.bool, device=device)
        right = torch.triu(torch.ones((Td, Td), dtype=torch.bool, device=device), diagonal=1)
        return torch.cat([left, right], dim=1)  # (Td, Te+Td)

    def _build_key_padding_mask(
        self,
        encoder_lengths: torch.Tensor,  # (B,)
        decoder_lengths: torch.Tensor,  # (B,)
        Te: int, Td: int, device
    ) -> torch.Tensor:
        """
        返回 (B, Te+Td) 布尔掩码；True=该样本该位置是 padding（禁止被关注）。
        规则：每行 [0:len) 有效(False)，[len:T) 为 padding(True)。
        """
        B = encoder_lengths.size(0)
        enc_idx = torch.arange(Te, device=device).unsqueeze(0).expand(B, -1)     # (B,Te)
        dec_idx = torch.arange(Td, device=device).unsqueeze(0).expand(B, -1)     # (B,Td)
        enc_pad = enc_idx >= encoder_lengths.unsqueeze(1)                        # (B,Te) True=pad
        dec_pad = dec_idx >= decoder_lengths.unsqueeze(1)                        # (B,Td) True=pad
        key_padding_mask = torch.cat([enc_pad, dec_pad], dim=1)                  # (B,Te+Td)
        return key_padding_mask

    # -------- ⭐ 核心 1：带长度的 LSTM 前向（请你填） --------
    def _run_lstm_with_lengths(
        self,
        lstm: nn.LSTM,
        x_seq: torch.Tensor,          # (B,T,H)
        lengths: torch.Tensor,        # (B,)
        h0: torch.Tensor, c0: torch.Tensor  # (B,H)
    ):
        """
        目标：仅沿真实长度滚动 LSTM。
        步骤：
          1) pack = pack_padded_sequence(x_seq, lengths=..., batch_first=True, enforce_sorted=False)
          2) out_packed, (h, c) = lstm(pack, (h0[None], c0[None]))
          3) out, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=x_seq.size(1))
        返回：out (B,T,H), (h,c) 其中 h,c 形状为 (1,B,H)
        """
        # ===== 你来写（按上面的 1-2-3 步）=====
        pack = pack_padded_sequence(x_seq, lengths=lengths, batch_first=True, enforce_sorted=False)
        out_packed, (h, c) = lstm(pack, (h0.unsqueeze(0), c0.unsqueeze(0)))
        out, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=x_seq.size(1))
        return out, (h, c)  

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
        h0 = self.init_h(static)           # (B,H)
        c0 = self.init_c(static)           # (B,H)

        # 2) VSN 投影
        enc_h = self._project_with_vsn(enc, which="enc")     # (B,Te,H)
        dec_h = self._project_with_vsn(dec, which="dec")     # (B,Td,H)

        # 3) -------- ⭐ 核心 2：用 pack/pad 的 LSTM 前向（请你填） --------
        # 提示：第二次调用时初态用上一行输出的 (hT, cT)，但要从 (1,B,H) 压回 (B,H)
        # enc_out, (hT, cT) = self._run_lstm_with_lengths(self.lstm_enc, enc_h, enc_len, h0, c0)
        # dec_out, _        = self._run_lstm_with_lengths(self.lstm_dec, dec_h, dec_len, hT.squeeze(0), cT.squeeze(0))
        enc_out, (hT, cT) = self._run_lstm_with_lengths(self.lstm_enc, enc_h, enc_len, h0, c0)
        dec_out, _        = self._run_lstm_with_lengths(self.lstm_dec, dec_h, dec_len, hT.squeeze(0), cT.squeeze(0))

        # 4) post-LSTM（门控 + 残差归一化；skip=dec_h）
        u = self.post_lstm_gate_dec(dec_out)                 # (B,Td,H)
        y = self.post_lstm_addnorm_dec(u, dec_h)             # (B,Td,H)

        # 5) 注意力输入：Q=decoder 段（y），K/V=encoder+decoder 拼接
        attn_input = torch.cat([enc_out, y], dim=1)          # (B,Te+Td,H)
        q = y; k = attn_input; v = attn_input

        # 6) 两类 mask：因果 + padding
        attn_mask = self._build_causal_attn_mask(Te, Td, device=enc.device)               # (Td, Te+Td)
        key_padding_mask = self._build_key_padding_mask(enc_len, dec_len, Te, Td, enc.device)  # (B,Te+Td)

        # 7) 注意力前向（两类 mask 同时生效）
        attn_out, attn_weights = self.mha(q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask)  # (B,Td,H)

        # 8) post-attn 残差归一化 & 9) 前馈 + pre-output AddNorm & 10) 输出头
        y_attn = self.post_attn_addnorm(attn_out, y)         # (B,Td,H)
        ff    = self.pos_wise_ff(y_attn)                     # (B,Td,H)
        fused = self.pre_output_addnorm(ff, y_attn)          # (B,Td,H)
        pred  = self.head(fused)                             # (B,Td,O)
        return {"prediction": pred}


# ========== 自测（你填完两处 TODO 后再跑） ==========
if __name__ == "__main__":
    torch.manual_seed(0)
    B, Te, Td, F, S, H, O = 2, 5, 3, 4, 6, 8, 2

    x = {
        "encoder_cont": torch.randn(B, Te, F),
        "decoder_cont": torch.randn(B, Td, F),
        "encoder_lengths": torch.tensor([Te, Te-1]),
        "decoder_lengths": torch.tensor([Td, Td-1]),
    }

    model = MiniTFT_Step11(hidden_size=H, output_size=O, x_reals=[f"f{i}" for i in range(F)],static_size=S, num_heads=2)
    out = model(x)   # <- 填完 TODO 后取消注释
    print("pred shape:", out["prediction"].shape)  # 期望: (2, 3, 2)
