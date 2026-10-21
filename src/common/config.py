from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_environment(raw: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        value = os.getenv(name)
        if value is not None:
            return value
        if default is not None:
            return default
        raise KeyError(f"Environment variable {name!r} is required")

    return _ENV_PATTERN.sub(replace, raw)


def read_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    loaded = yaml.safe_load(_expand_environment(path.read_text(encoding="utf-8")))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Top-level YAML value must be a mapping: {path}")
    return loaded


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _coerce_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _apply_runtime_environment(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    direct = {
        "AGENTREC_DATA_ROOT": ("data_root", str),
        "AGENTREC_ARTIFACT_ROOT": ("artifact_root", str),
        "AGENTREC_DEVICE": ("device", str),
    }
    nested = {
        "AGENTREC_LLM_ENABLED": ("llm", "enabled", _coerce_bool),
        "AGENTREC_LLM_BASE_URL": ("llm", "base_url", str),
        "AGENTREC_LLM_API_KEY": ("llm", "api_key", str),
        "AGENTREC_LLM_MODEL": ("llm", "model", str),
        "AGENTREC_EMBEDDING_PROVIDER": ("embedding", "provider", str),
        "AGENTREC_EMBEDDING_MODEL": ("embedding", "model_name_or_path", str),
        "AGENTREC_EMBEDDING_BASE_URL": ("embedding", "base_url", str),
        "AGENTREC_EMBEDDING_API_KEY": ("embedding", "api_key", str),
    }
    for env_name, (key, caster) in direct.items():
        if env_name in os.environ:
            result[key] = caster(os.environ[env_name])
    for env_name, (section, key, caster) in nested.items():
        if env_name in os.environ:
            result.setdefault(section, {})[key] = caster(os.environ[env_name])
    return result


def load_config(
    environment: str | None = None,
    override_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load environment and component YAML files into one configuration mapping."""
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass
    environment = environment or os.getenv("AGENTREC_ENV", "local")
    if environment not in {"local", "server"}:
        raise ValueError("environment must be 'local' or 'server'")
    config = read_yaml(PROJECT_ROOT / "configs" / f"{environment}.yaml")
    for filename in ("retriever.yaml", "ranker.yaml", "agent.yaml"):
        config = deep_merge(config, read_yaml(PROJECT_ROOT / "configs" / filename))
    if override_path:
        config = deep_merge(config, read_yaml(override_path))
    config = _apply_runtime_environment(config)
    for root_key in ("data_root", "artifact_root"):
        root = Path(str(config[root_key])).expanduser()
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        config[root_key] = str(root.resolve())
    config["environment"] = environment
    return config


def config_value(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def data_path(config: Mapping[str, Any], *parts: str) -> Path:
    return Path(str(config["data_root"])).expanduser().joinpath(*parts)


def artifact_path(config: Mapping[str, Any], *parts: str) -> Path:
    return Path(str(config["artifact_root"])).expanduser().joinpath(*parts)
