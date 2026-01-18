from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassToken(nn.Module):
    """
    Learnable class token.
    """
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (B, N, D)
        returns: (B, 1, D)
        """
        batch_size = x.size(0)
        return self.cls_token.expand(batch_size, -1, -1)


class MLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float
    ) -> None:
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, mlp_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self attention + skip
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = residual + attn_out

        # MLP + skip
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x

        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer for patchified image inputs.

    Expected input shape:
    (B, num_patches, patch_dim)
    """

    def __init__(self, config: Dict) -> None:
        super().__init__()

        self.num_patches: int = config["num_patches"]
        self.hidden_dim: int = config["hidden_dim"]
        self.num_classes: int = config["num_classes"]

        patch_dim = (
            config["patch_size"]
            * config["patch_size"]
            * config["num_channels"]
        )

        # Patch embedding
        self.patch_embed = nn.Linear(patch_dim, self.hidden_dim)

        # Positional embedding
        self.pos_embed = nn.Embedding(self.num_patches, self.hidden_dim)

        # Class token
        self.class_token = ClassToken(self.hidden_dim)

        # Transformer encoder
        self.encoder = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=config["num_heads"],
                    mlp_dim=config["mlp_dim"],
                    dropout=config["dropout_rate"],
                )
                for _ in range(config["num_layers"])
            ]
        )

        self.norm = nn.LayerNorm(self.hidden_dim)
        self.head = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (B, num_patches, patch_dim)
        returns: (B, num_classes)
        """

        # Patch embedding
        x = self.patch_embed(x)

        # Positional embedding
        positions = torch.arange(
            self.num_patches,
            device=x.device
        )
        x = x + self.pos_embed(positions)

        # Class token
        cls_token = self.class_token(x)
        x = torch.cat([cls_token, x], dim=1)

        # Transformer blocks
        for block in self.encoder:
            x = block(x)

        # Classification head
        x = self.norm(x)
        cls_output = x[:, 0]
        logits = self.head(cls_output)

        return logits