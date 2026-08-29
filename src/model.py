import numpy as np
import torch
import torch.nn as nn


def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angles = pos * angle_rates
    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return torch.tensor(pe)


class TCMNLite(nn.Module):
    def __init__(self, seq_len=3, num_countries=5, pos_dim=4, hidden_dim=16,
                 country_embed_dim=4, num_classes=3):
        super().__init__()
        self.seq_len = seq_len
        self.pos_dim = pos_dim

        self.register_buffer("pos_encoding", positional_encoding(seq_len, pos_dim))

        self.lstm = nn.LSTM(input_size=1 + pos_dim, hidden_size=hidden_dim, batch_first=True)

        self.country_embedding = nn.Embedding(num_countries, country_embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + country_embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, price_seq, country_ids):
        batch_size = price_seq.size(0)
        pos = self.pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)  
        x = torch.cat([price_seq, pos], dim=-1)                         

        _, (h_n, _) = self.lstm(x)
        seq_repr = h_n[-1]                                            

        country_repr = self.country_embedding(country_ids)              

        combined = torch.cat([seq_repr, country_repr], dim=-1)
        logits = self.classifier(combined)
        return logits


if __name__ == "__main__":
    model = TCMNLite(num_countries=5)
    dummy_price = torch.randn(8, 3, 1)
    dummy_country = torch.randint(0, 5, (8,))
    out = model(dummy_price, dummy_country)
    print("output shape:", out.shape)
