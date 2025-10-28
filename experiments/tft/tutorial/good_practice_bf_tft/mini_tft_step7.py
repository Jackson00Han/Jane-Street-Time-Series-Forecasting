import torch
from torch import nn

class MiniTFT_Step7(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, x_reals: list[str], static_size: int):
        super().__init__()
        self.H = hidden_size
        self.O = output_size
        self.x_reals = x_reals
        F = len(x_reals)

        # prescalers + 两套 VSN 打分层（延用前一步）
        self.prescalers = nn.ModuleDict({name: nn.Linear(1, self.H) for name in x_reals})
        self.vsn_score_enc = nn.Linear(F, F, bias=False)
        self.vsn_score_dec = nn.Linear(F, F, bias=False)

        # 静态 -> 初态
        self.init_h = nn.Linear(static_size, self.H)
        self.init_c = nn.Linear(static_size, self.H)

        # 关键：双 LSTM（你已经在 __init__ 装上了）
        self.lstm_enc = nn.LSTM(input_size=self.H, hidden_size=self.H, batch_first=True)
        self.lstm_dec = nn.LSTM(input_size=self.H, hidden_size=self.H, batch_first=True)

        # 输出头
        self.head = nn.Linear(self.H, self.O)
        self.static_size = static_size

    @staticmethod
    def expand_static_context(context: torch.Tensor, timesteps: int) -> torch.Tensor:
        return context[:, None, :].expand(-1, timesteps, -1)

    def _project_with_vsn(self, x_cont: torch.Tensor, which: str) -> torch.Tensor:
        B, T, F = x_cont.shape
        outs = [self.prescalers[name](x_cont[..., j:j+1]) for j, name in enumerate(self.x_reals)]
        stacked = torch.stack(outs, dim=-1)                   # (B,T,H,F)
        scorer  = self.vsn_score_enc if which == "enc" else self.vsn_score_dec
        weights = torch.softmax(scorer(x_cont), dim=-1)       # (B,T,F)
        h = torch.matmul(stacked, weights.unsqueeze(-1)).squeeze(-1)  # (B,T,H)
        return h

    def forward(self, x: dict) -> dict:
        enc, dec = x["encoder_cont"], x["decoder_cont"]      # (B,Te,F), (B,Td,F)
        B, Te, _ = enc.shape
        Td = dec.size(1)

        # 1) 静态 -> 初态 (B,H)
        static = torch.zeros(B, self.static_size, device=enc.device)
        h0 = self.init_h(static)           # (B,H)
        c0 = self.init_c(static)           # (B,H)

        # 2) 连续特征投影 + 变量选择得到时序隐表示
        enc_h = self._project_with_vsn(enc, which="enc")     # (B,Te,H)
        dec_h = self._project_with_vsn(dec, which="dec")     # (B,Td,H)

        # 3) 接 LSTM：注意 LSTM 期望初态形状是 (num_layers, B, H)，这里 num_layers=1
        h0_l = h0.unsqueeze(0)             # (1,B,H)
        c0_l = c0.unsqueeze(0)             # (1,B,H)

        # TODO 1: 编码器前向：把 enc_h 喂给 self.lstm_enc，初态用 (h0_l, c0_l)
        # enc_out: (B,Te,H); (hT, cT): (1,B,H)
        enc_out, (hT, cT) = self.lstm_enc(enc_h, (h0_l, c0_l))

        # TODO 2: 解码器前向：把 dec_h 喂给 self.lstm_dec，初态用 (hT, cT)
        # dec_out: (B,Td,H)
        dec_out, _ = self.lstm_dec(dec_h, (hT, cT))

        # 4) 只对解码器输出走 head
        pred = self.head(dec_out)          # (B,Td,O)

        # （可选）演示：把 h0 扩到时间维，便于你打印查看
        h0_time = self.expand_static_context(h0, Te)         # (B,Te,H)
        return {"prediction": pred, "h0": h0, "c0": c0, "h0_time": h0_time}


if __name__ == "__main__":
    B, Te, Td, F, S = 2, 5, 3, 4, 6
    x = {
        "encoder_cont": torch.randn(B, Te, F),
        "decoder_cont": torch.randn(B, Td, F),
    }
    model = MiniTFT_Step7(hidden_size=8, output_size=2, x_reals=[f"f{i}" for i in range(F)], static_size=S)
    out = model(x)
    print("pred:", out["prediction"].shape)  # 期望: (2, 3, 2)
    print("enc h0->Te broadcast:", out["h0_time"].shape)  # 期望: (2, 5, 8)
