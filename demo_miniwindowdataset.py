class MiniWindowDataset():
    def __init__(self, data, enc_len, dec_len):
        self.data = data
        self.enc_len = enc_len
        self.dec_len = dec_len
        
    @classmethod
    def _with_default_enc_dec_len(cls, data):
        return cls(data, enc_len=3, dec_len=2) 
    
    def __len__(self):
        return len(self.data) - self.enc_len - self.dec_len + 1
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of bounds")
        
        encoder = self.data[idx:idx+self.enc_len]
        decoder = self.data[idx+self.enc_len:idx+self.enc_len+self.dec_len]
        target_scale = sum(abs(x) for x in encoder) / len(encoder) if encoder else 1
        
        return {'encoder': encoder, 'decoder': decoder, "target_scale": target_scale}
    
def mini_collate(batch):
    encoders = [item['encoder'] for item in batch]
    decoders = [item['decoder'] for item in batch]
    target_scales = [item['target_scale'] for item in batch]
    return {'batch encoder': encoders, 'batch decoder': decoders, 'scales': target_scales}
    
def main():
    data = [1, 2, 3, 4, 5, 6]
    ds = MiniWindowDataset._with_default_enc_dec_len(data)
    
    b = [ds[0], ds[1]]
    out = mini_collate(b)
    print("batch encoder:", out["batch encoder"])
    print("batch decoder:", out["batch decoder"])
    print("batch target_scale:", out["scales"])


if __name__ == "__main__":
    main()