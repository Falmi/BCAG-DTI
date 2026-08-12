from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ablation import A1_ORIGINAL, get_variant_spec, normalize_variant_name
from models import BIN_Interaction_Flat, Embeddings, Encoder_MultipleLayers


class CrossAttentionInteraction(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.drug_to_protein = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.protein_to_drug = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drug_layer_norm = nn.LayerNorm(hidden_size)
        self.protein_layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        drug: torch.Tensor,
        protein: torch.Tensor,
        drug_mask: torch.Tensor,
        protein_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        drug_valid = drug_mask.to(dtype=torch.bool)
        protein_valid = protein_mask.to(dtype=torch.bool)
        drug_update, _ = self.drug_to_protein(
            drug,
            protein,
            protein,
            key_padding_mask=~protein_valid,
            need_weights=False,
        )
        drug = self.drug_layer_norm(drug + drug_update)
        drug = drug.masked_fill(~drug_valid.unsqueeze(-1), 0.0)

        protein_update, _ = self.protein_to_drug(
            protein,
            drug,
            drug,
            key_padding_mask=~drug_valid,
            need_weights=False,
        )
        protein = self.protein_layer_norm(protein + protein_update)
        protein = protein.masked_fill(~protein_valid.unsqueeze(-1), 0.0)
        return drug, protein


def masked_average_max_pool(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.to(dtype=torch.bool).unsqueeze(-1)
    counts = valid.sum(dim=1).clamp_min(1).to(dtype=sequence.dtype)
    average = sequence.masked_fill(~valid, 0.0).sum(dim=1) / counts
    maximum = sequence.masked_fill(~valid, torch.finfo(sequence.dtype).min).max(dim=1).values
    has_valid = valid.any(dim=1)
    maximum = torch.where(has_valid, maximum, torch.zeros_like(maximum))
    return torch.cat((average, maximum), dim=1)


def original_classifier(input_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.ReLU(True),
        nn.BatchNorm1d(512),
        nn.Linear(512, 64),
        nn.ReLU(True),
        nn.BatchNorm1d(64),
        nn.Linear(64, 32),
        nn.ReLU(True),
        nn.Linear(32, 1),
    )


class EnhancedClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims, dropout_rates):
        super().__init__()
        if len(hidden_dims) != 3 or len(dropout_rates) != 3:
            raise ValueError("Enhanced classifier requires three hidden dimensions and dropouts.")
        layers = []
        previous = input_dim
        for hidden, dropout in zip(hidden_dims, dropout_rates):
            layers.append(
                nn.Sequential(
                    nn.Linear(previous, hidden),
                    nn.LayerNorm(hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            previous = hidden
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(previous, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            features = layer(features)
        return self.output(features)


class MolTransAblationModel(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.variant = normalize_variant_name(config["variant"])
        self.spec = get_variant_spec(self.variant)
        self.max_d = int(config["max_drug_seq"])
        self.max_p = int(config["max_protein_seq"])
        self.emb_size = int(config["emb_size"])
        self.dropout_rate = float(config["dropout_rate"])
        self.flatten_dim = int(config["flat_dim"])

        self.drug_embedding = Embeddings(
            config["input_dim_drug"], self.emb_size, self.max_d, self.dropout_rate
        )
        self.protein_embedding = Embeddings(
            config["input_dim_target"], self.emb_size, self.max_p, self.dropout_rate
        )
        encoder_arguments = (
            2,
            self.emb_size,
            config["intermediate_size"],
            config["num_attention_heads"],
            config["attention_probs_dropout_prob"],
            config["hidden_dropout_prob"],
        )
        self.drug_encoder = Encoder_MultipleLayers(*encoder_arguments)
        self.protein_encoder = Encoder_MultipleLayers(*encoder_arguments)

        self.cross_attention = None
        if self.spec.use_cross_attention:
            self.cross_attention = CrossAttentionInteraction(
                self.emb_size,
                num_heads=int(config.get("cross_attention_heads", 8)),
                dropout=self.dropout_rate,
            )

        self.interaction_conv = nn.Conv2d(1, 3, 3, padding=0)
        classifier_input = self.flatten_dim
        if self.spec.use_multiscale_pooling:
            classifier_input += 4 * self.emb_size
        self.classifier_input_dim = classifier_input
        if self.spec.use_enhanced_classifier:
            self.classifier = EnhancedClassifier(
                classifier_input,
                config.get("classifier_hidden_dims", [1024, 512, 256]),
                config.get("classifier_dropout_rates", [0.15, 0.15, 0.075]),
            )
        else:
            self.classifier = original_classifier(classifier_input)

    @staticmethod
    def _extended_mask(mask: torch.Tensor) -> torch.Tensor:
        extended = mask.unsqueeze(1).unsqueeze(2)
        return (1.0 - extended) * -10000.0

    def forward(
        self,
        drug: torch.Tensor,
        protein: torch.Tensor,
        drug_mask: torch.Tensor,
        protein_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = drug.size(0)
        drug_embedding = self.drug_embedding(drug)
        protein_embedding = self.protein_embedding(protein)
        drug_encoded = self.drug_encoder(
            drug_embedding.float(), self._extended_mask(drug_mask).float()
        )
        protein_encoded = self.protein_encoder(
            protein_embedding.float(), self._extended_mask(protein_mask).float()
        )

        if self.cross_attention is not None:
            drug_encoded, protein_encoded = self.cross_attention(
                drug_encoded,
                protein_encoded,
                drug_mask,
                protein_mask,
            )

        interaction = torch.bmm(drug_encoded, protein_encoded.transpose(1, 2))
        interaction = interaction.unsqueeze(1)
        interaction = F.dropout(
            interaction,
            p=self.dropout_rate,
            training=self.training,
        )
        convolution = self.interaction_conv(interaction).reshape(batch_size, -1)
        if convolution.size(1) != self.flatten_dim:
            raise RuntimeError(
                f"Interaction features have dimension {convolution.size(1)}; "
                f"configured flat_dim is {self.flatten_dim}."
            )

        features = convolution
        if self.spec.use_multiscale_pooling:
            global_features = torch.cat(
                (
                    masked_average_max_pool(drug_encoded, drug_mask),
                    masked_average_max_pool(protein_encoded, protein_mask),
                ),
                dim=1,
            )
            features = torch.cat((features, global_features), dim=1)
        return self.classifier(features)


def build_moltrans_model(config: Dict[str, Any], variant: str | None = None) -> nn.Module:
    variant = normalize_variant_name(variant or config.get("variant", A1_ORIGINAL))
    spec = get_variant_spec(variant)
    if not (
        spec.use_cross_attention
        or spec.use_multiscale_pooling
        or spec.use_enhanced_classifier
    ):
        return BIN_Interaction_Flat(**config)
    model_config = dict(config)
    model_config["variant"] = variant
    return MolTransAblationModel(**model_config)
