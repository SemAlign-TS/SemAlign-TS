import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TimeSeriesDiffuserV2(nn.Module):

    def __init__(self, seq_len=96, input_dim=1, text_dim=768,
                 model_dim=256, num_layers=4, nhead=8):
        super().__init__()

        self.seq_len = seq_len
        self.model_dim = model_dim

        self.x_proj = nn.Linear(input_dim, model_dim)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

        pos_emb_init = torch.zeros(1, seq_len, model_dim)
        position = torch.arange(seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim)
        )
        pos_emb_init[0, :, 0::2] = torch.sin(position * div_term)
        pos_emb_init[0, :, 1::2] = torch.cos(position * div_term[:model_dim // 2])
        self.pos_emb = nn.Parameter(pos_emb_init)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        self.final_proj = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, input_dim),
        )

    def forward(self, x, t, text_emb, text_mask=None):
        B, L, _ = x.shape

        h = self.x_proj(x)
        h = h + self.pos_emb[:, :L, :]
        time_emb = self.time_mlp(t)
        h = h + time_emb.unsqueeze(1)

        memory = self.text_proj(text_emb)

        memory_key_padding_mask = (~text_mask) if text_mask is not None else None

        output = self.transformer_decoder(
            tgt=h,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        return self.final_proj(output)
