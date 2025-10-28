import torch
from torch import nn

class MiniTFT_Step2(nn.Module):
    """
    Step2：只对 decoder 段做线性输出（不做 prescalers/VSN/LSTM/注意力）
    输入：
      x["encoder_cont"]: (B, Te, F)
      x["decoder_cont"]: (B, Td, F)
    输出：
      out["prediction"]: (B, Td, O)
    """
    def __init__(self, output_size: int = 1):
        super().__init__()
        self.output_size = output_size
        self.head = None  # 延迟初始化，等拿到 F 再建 nn.Linear(F, O)

    def _lazy_build_head(self, F: int):
        # TODO 1: 建立线性头，把特征维 F -> output_size
        # 例如：self.head = nn.Linear(F, self.output_size)
        self.head = nn.Linear(F, self.output_size)
        

    def forward(self, x: dict) -> dict:
        enc = x["encoder_cont"]  # (B, Te, F)
        dec = x["decoder_cont"]  # (B, Td, F)
        B, Te, F = enc.shape
        Td = dec.size(1)

        # TODO 2: 如果 head 还没建，用 _lazy_build_head(F) 创建
        if self.head is None:
            self._lazy_build_head(F)
        # TODO 3: 只对“解码段”逐时间步应用线性头
        # 线索：dec 形状 (B, Td, F)，nn.Linear 会自动按最后一维做仿射变换
        pred = self.head(dec)  # (B, Td, self.output_size)

        return {"prediction": pred}


if __name__ == "__main__":
    B, Te, Td, F = 2, 5, 3, 4
    x = {
        "encoder_cont": torch.randn(B, Te, F),
        "decoder_cont": torch.randn(B, Td, F),
    }
    model = MiniTFT_Step2(output_size=3)
    out = model(x)
    print("pred shape:", out["prediction"].shape)  # 期望: (2, 3, 3)
