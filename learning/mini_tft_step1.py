import torch
from torch import nn

class MiniTFT_Step1(nn.Module):
    """
    Step1: 只做形状与键名约定, 不做真实计算。
    约定输入:
    x["encoder_cont"]: (B, Te, F)
    x["decoder_cont"]: (B, Td, F)
    约定输出:
    out["prediction"]: (B, Td, O)
    """
    def __init__(self, output_size: int = 1):
        super().__init__()
        self.output_size = output_size
        # 一个极简“头”，暂时只输出常数零（下一步换成线性层）
        self.register_buffer("_dummy", torch.zeros(1))

    def forward(self, x: dict) -> dict:
        enc = x["encoder_cont"]  # (B, Te, F)
        dec = x["decoder_cont"]  # (B, Td, F)
        B, Te, F = enc.shape
        Td = dec.size(1)

        # TODO 1: 创建一个形状为 (B, Td, self.output_size) 的全零张量，放在 enc 的同设备/同 dtype 上
        # 提示：用 torch.zeros + device=enc.device + dtype=enc.dtype
        pred = torch.zeros(B, Td, self.output_size, device=enc.device, dtype=enc.dtype)

        # TODO 2: 返回字典 {"prediction": pred}
        return {"prediction": pred} 


if __name__ == "__main__":
    B, Te, Td, F = 2, 5, 3, 4
    x = {
        "encoder_cont": torch.randn(B, Te, F),
        "decoder_cont": torch.randn(B, Td, F),
    }
    model = MiniTFT_Step1(output_size=1)
    out = model(x)
    print("pred shape:", out["prediction"].shape)  # 期望: (2, 3, 1)
