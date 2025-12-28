import torch
import torch.nn as nn


class SimpleViT(nn.Module):
    def __init__(self,
                 img_size=32,
                 patch_size=8,
                 in_chans=3,
                 embed_dim=128,
                 depth=6,
                 num_heads=4,
                 mlp_ratio=2.0,
                 num_classes=200,  # 根据任务自行调整
                 drop_rate=0.1):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"

        self.num_patches = (img_size // patch_size) ** 2  # 32/8=4 => 16个token
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)

        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(drop_rate)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=int(embed_dim * mlp_ratio),
                                                   dropout=drop_rate,
                                                   activation='gelu',
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)

        # 分类头
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, x):
        # x shape: [B, C, H, W]
        x = self.patch_embed(x)  # [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]

        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.transformer(x)
        x = self.norm(x)

        x = x.mean(dim=1)  # 全局平均池化
        return self.head(x)


if __name__ == "__main__":
    model = SimpleViT(img_size=32, num_classes=200)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    print(out.shape)  # torch.Size([2, 200])
