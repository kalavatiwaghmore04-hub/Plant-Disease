import json
from pathlib import Path
from threading import Lock
from urllib.request import Request, urlopen

import numpy as np

from .config import Settings
from .schemas import PredictionItem, PredictionResponse


# Public Hugging Face model repository
HF_MODEL_BASE_URL = (
    "https://huggingface.co/Kalavati/plantguard-ai-model/"
    "resolve/main"
)


class ModelUnavailableError(RuntimeError):
    """Model artifacts cannot serve inference."""


class ModelService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._labels: dict[int, str] = {}
        self._error: str | None = None
        self._lock = Lock()

    @property
    def ready(self) -> bool:
        self._ensure_loaded()
        return self._model is not None and bool(self._labels)

    @property
    def error(self) -> str | None:
        self._ensure_loaded()
        return self._error

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._error is not None:
            return

        with self._lock:
            if self._model is not None or self._error is not None:
                return

            try:
                self._load_artifacts()
            except Exception as exc:
                self._error = str(exc)

    def _load_artifacts(self) -> None:
        model_path = self.settings.model_file
        mapping_path = self.settings.class_mapping_file

        # Download missing deployment artifacts from Hugging Face.
        _download_if_missing(
            model_path,
            f"{HF_MODEL_BASE_URL}/plant_disease_model.keras?download=true",
        )

        _download_if_missing(
            mapping_path,
            f"{HF_MODEL_BASE_URL}/class_mapping.json?download=true",
        )

        if not model_path.is_file():
            raise ModelUnavailableError(
                f"Model file not found: {model_path}"
            )

        if not mapping_path.is_file():
            raise ModelUnavailableError(
                f"Class mapping file not found: {mapping_path}"
            )

        import tensorflow as tf

        model = tf.keras.models.load_model(
            model_path,
            compile=False,
        )

        labels = _read_labels(mapping_path)

        output_classes = int(model.output_shape[-1])

        if output_classes != len(labels):
            raise ModelUnavailableError(
                f"Model outputs {output_classes} classes "
                f"but mapping has {len(labels)} entries."
            )

        self._model = model
        self._labels = labels

    def predict(self, batch: np.ndarray) -> PredictionResponse:
        if not self.ready:
            raise ModelUnavailableError(
                self.error or "Model is unavailable."
            )

        probabilities = np.asarray(
            self._model.predict(batch, verbose=0)
        )[0]

        if (
            probabilities.ndim != 1
            or len(probabilities) != len(self._labels)
            or not np.isfinite(probabilities).all()
        ):
            raise ModelUnavailableError(
                "Model returned invalid prediction values."
            )

        indices = np.argsort(probabilities)[::-1][
            : min(3, len(probabilities))
        ]

        predictions = [
            PredictionItem(
                label=self._labels[int(index)],
                confidence=float(probabilities[index]),
            )
            for index in indices
        ]

        return PredictionResponse(
            prediction=predictions[0],
            top_predictions=predictions,
            low_confidence=(
                predictions[0].confidence
                < self.settings.low_confidence_threshold
            ),
            threshold=self.settings.low_confidence_threshold,
        )


def _download_if_missing(path: Path, url: str) -> None:
    """Download an artifact only when it does not already exist."""

    if path.is_file() and path.stat().st_size > 0:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model artifact: {path.name}")

    request = Request(
        url,
        headers={
            "User-Agent": "PlantGuard-AI/1.0",
        },
    )

    temporary_path = path.with_suffix(path.suffix + ".download")

    try:
        with urlopen(request, timeout=300) as response:
            with temporary_path.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)

        if not temporary_path.exists() or temporary_path.stat().st_size == 0:
            raise ModelUnavailableError(
                f"Downloaded artifact is empty: {path.name}"
            )

        temporary_path.replace(path)

        print(f"Downloaded successfully: {path.name}")

    except Exception as exc:
        if temporary_path.exists():
            temporary_path.unlink()

        raise ModelUnavailableError(
            f"Failed to download {path.name} from Hugging Face: {exc}"
        ) from exc


def _read_labels(path: Path) -> dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(raw, dict):
        raise ModelUnavailableError(
            "Class mapping must be a JSON object of index to label."
        )

    try:
        labels = {
            int(index): str(label)
            for index, label in raw.items()
        }
    except (TypeError, ValueError) as exc:
        raise ModelUnavailableError(
            "Class mapping keys must be integer indices."
        ) from exc

    if (
        not labels
        or sorted(labels) != list(range(len(labels)))
        or any(not value.strip() for value in labels.values())
    ):
        raise ModelUnavailableError(
            "Class mapping needs non-empty sequential indices "
            "starting at 0."
        )

    return labels
