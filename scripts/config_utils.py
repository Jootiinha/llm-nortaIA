from pathlib import Path
from typing import Any

import yaml

DEFAULT_MODEL_NAME = "joao-gpt-mini-v1"
CONFIGS_DIR = Path("configs/models")


def resolve_config_path(config_value: str | None) -> Path:
    value = config_value or DEFAULT_MODEL_NAME
    candidate = Path(value)

    if candidate.suffix in {".yaml", ".yml"} or "/" in value:
        path = candidate
    else:
        path = CONFIGS_DIR / f"{value}.yaml"

    if not path.exists():
        raise FileNotFoundError(f"Arquivo de config não encontrado: {path}")

    return path


def load_config(config_value: str | None) -> dict[str, Any]:
    path = resolve_config_path(config_value)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config inválida em {path}: esperado objeto YAML.")

    data.setdefault("name", path.stem)
    data["_config_path"] = str(path)
    data["paths"] = build_paths(data)
    return data


def build_paths(config: dict[str, Any]) -> dict[str, str]:
    name = config.get("name", DEFAULT_MODEL_NAME)
    configured_paths = config.get("paths", {}) or {}

    paths = {
        "tokenizer_corpus_dir": "data/tokenizer_corpus",
        "tokenizer_dir": f"tokenizer/{name}-tokenizer-bpe",
        "train_file": f"data/{name}/train.txt",
        "valid_file": f"data/{name}/valid.txt",
        "output_dir": f"checkpoints/{name}",
        "eval_file": "data/tokenizer_corpus/pt_corpus_instruct_sample.txt",
    }
    paths.update(configured_paths)
    return paths


def get_section(config: dict[str, Any], section_name: str) -> dict[str, Any]:
    section = config.get(section_name, {}) or {}

    if not isinstance(section, dict):
        raise ValueError(f"Seção '{section_name}' da config deve ser um objeto.")

    return section


def get_value(config: dict[str, Any], section_name: str, key: str, default: Any) -> Any:
    section = get_section(config, section_name)
    return section.get(key, default)
