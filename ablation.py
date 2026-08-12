from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

from experiment import EXPERIMENT_TASKS, normalize_task_name


A1_ORIGINAL = "a1_original"
A2_TRAINING = "a2_training_strategy"
A3_CROSS = "a3_cross_attention"
A4_POOLING = "a4_multiscale_pooling"
A5_CLASSIFIER = "a5_enhanced_classifier"
A6_CROSS_POOLING = "a6_cross_pooling"
A7_CROSS_POOLING_TRAINING = "a7_cross_pooling_training"
A8_FULL = "a8_bcag_dti"

MAIN_TASKS = ("biosnap_full", "bindingdb", "davis")


@dataclass(frozen=True)
class VariantSpec:
    name: str
    display_name: str
    use_training_strategy: bool
    use_cross_attention: bool
    use_multiscale_pooling: bool
    use_enhanced_classifier: bool


@dataclass(frozen=True)
class TrainingSettings:
    optimized: bool
    max_epochs: int
    learning_rate: float
    weight_decay: float
    warmup_epochs: int
    patience: int | None
    dropout_rate: float
    label_smoothing: float
    use_amp: bool
    max_grad_norm: float | None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


VARIANT_SPECS: Dict[str, VariantSpec] = {
    A1_ORIGINAL: VariantSpec(A1_ORIGINAL, "A1 Original MolTrans", False, False, False, False),
    A2_TRAINING: VariantSpec(A2_TRAINING, "A2 Training Strategy", True, False, False, False),
    A3_CROSS: VariantSpec(A3_CROSS, "A3 Cross-Attention", False, True, False, False),
    A4_POOLING: VariantSpec(A4_POOLING, "A4 Multiscale Pooling", False, False, True, False),
    A5_CLASSIFIER: VariantSpec(A5_CLASSIFIER, "A5 Enhanced Classifier", False, False, False, True),
    A6_CROSS_POOLING: VariantSpec(
        A6_CROSS_POOLING, "A6 Cross-Attention + Pooling", False, True, True, False
    ),
    A7_CROSS_POOLING_TRAINING: VariantSpec(
        A7_CROSS_POOLING_TRAINING,
        "A7 Cross-Attention + Pooling + Training",
        True,
        True,
        True,
        False,
    ),
    A8_FULL: VariantSpec(A8_FULL, "A8 BCAG-DTI", True, True, True, True),
}
VARIANT_CHOICES = tuple(VARIANT_SPECS)
NEW_VARIANTS = VARIANT_CHOICES[1:]


def normalize_variant_name(variant: str | None) -> str:
    normalized = (variant or A1_ORIGINAL).strip().lower().replace("-", "_")
    if normalized not in VARIANT_SPECS:
        raise ValueError(f"Unsupported model variant: {variant}")
    return normalized


def get_variant_spec(variant: str | None) -> VariantSpec:
    return VARIANT_SPECS[normalize_variant_name(variant)]


def checkpoint_variant(checkpoint: Dict[str, Any]) -> str:
    return normalize_variant_name(
        checkpoint.get("variant")
        or checkpoint.get("train_args", {}).get("variant")
        or A1_ORIGINAL
    )


def tasks_for_variant(variant: str) -> Tuple[str, ...]:
    variant = normalize_variant_name(variant)
    if variant in {A1_ORIGINAL, A8_FULL}:
        return tuple(EXPERIMENT_TASKS)
    return MAIN_TASKS


def build_new_experiment_matrix() -> List[Tuple[str, str]]:
    return [
        (variant, task)
        for variant in NEW_VARIANTS
        for task in tasks_for_variant(variant)
    ]


def resolve_training_settings(variant: str, task: str) -> TrainingSettings:
    spec = get_variant_spec(variant)
    task = normalize_task_name(task)
    is_davis = task == "davis"
    if not spec.use_training_strategy:
        return TrainingSettings(
            optimized=False,
            max_epochs=50,
            learning_rate=1e-4,
            weight_decay=0.0,
            warmup_epochs=0,
            patience=None,
            dropout_rate=0.1,
            label_smoothing=0.0,
            use_amp=False,
            max_grad_norm=None,
        )
    return TrainingSettings(
        optimized=True,
        max_epochs=60 if is_davis else 50,
        learning_rate=1.2e-4 if is_davis else 1e-4,
        weight_decay=5e-6 if is_davis else 1e-5,
        warmup_epochs=5 if is_davis else 3,
        patience=25 if is_davis else 15,
        dropout_rate=0.1 if is_davis else 0.15,
        label_smoothing=0.1 if is_davis else 0.0,
        use_amp=True,
        max_grad_norm=1.0,
    )


def build_model_config(
    base_config: Dict[str, Any],
    variant: str,
    task: str,
    settings: TrainingSettings | None = None,
) -> Dict[str, Any]:
    spec = get_variant_spec(variant)
    task = normalize_task_name(task)
    settings = settings or resolve_training_settings(spec.name, task)
    config = dict(base_config)
    config.update(
        {
            "variant": spec.name,
            "task": task,
            "dropout_rate": settings.dropout_rate,
            "use_cross_attention": spec.use_cross_attention,
            "use_multiscale_pooling": spec.use_multiscale_pooling,
            "use_enhanced_classifier": spec.use_enhanced_classifier,
            "cross_attention_heads": 8,
        }
    )
    if spec.use_enhanced_classifier:
        head_dropout = settings.dropout_rate if spec.use_training_strategy else 0.15
        config["classifier_hidden_dims"] = [1024, 512, 256]
        config["classifier_dropout_rates"] = [
            head_dropout,
            head_dropout,
            head_dropout * 0.5,
        ]
    return config


def variant_manifest() -> List[Dict[str, Any]]:
    return [asdict(VARIANT_SPECS[name]) for name in VARIANT_CHOICES]
