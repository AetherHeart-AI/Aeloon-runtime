"""Base class for agent tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal


class Tool(ABC):
    """Abstract base class for agent tools."""

    _TYPE_MAP = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name used in function calls."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does."""

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for tool parameters."""

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """Execute the tool with given parameters."""

    @property
    def concurrency_mode(self) -> Literal["read_only", "mutating", "exclusive"]:
        """Execution mode for task-graph conflict analysis."""

        return "exclusive"

    def cast_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Apply safe schema-driven casts before validation."""

        schema = self.parameters or {}
        if schema.get("type", "object") != "object":
            return params
        return self._cast_object(params, schema)

    def _cast_object(self, obj: Any, schema: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(obj, dict):
            return obj
        props = schema.get("properties", {})
        result: dict[str, Any] = {}
        for key, value in obj.items():
            result[key] = self._cast_value(value, props[key]) if key in props else value
        return result

    def _cast_value(self, val: Any, schema: dict[str, Any]) -> Any:
        target_type = schema.get("type")
        if target_type == "boolean" and isinstance(val, bool):
            return val
        if target_type == "integer" and isinstance(val, int) and not isinstance(val, bool):
            return val
        if target_type in self._TYPE_MAP and target_type not in (
            "boolean",
            "integer",
            "array",
            "object",
        ):
            expected = self._TYPE_MAP[target_type]
            if isinstance(val, expected):
                return val
        if target_type == "integer" and isinstance(val, str):
            try:
                return int(val)
            except ValueError:
                return val
        if target_type == "number" and isinstance(val, str):
            try:
                return float(val)
            except ValueError:
                return val
        if target_type == "string":
            return val if val is None else str(val)
        if target_type == "boolean" and isinstance(val, str):
            val_lower = val.lower()
            if val_lower in ("true", "1", "yes"):
                return True
            if val_lower in ("false", "0", "no"):
                return False
        if target_type == "array" and isinstance(val, list):
            item_schema = schema.get("items")
            return [self._cast_value(item, item_schema) for item in val] if item_schema else val
        if target_type == "object" and isinstance(val, dict):
            return self._cast_object(val, schema)
        return val

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        """Validate tool parameters against JSON schema."""

        if not isinstance(params, dict):
            return [f"parameters must be an object, got {type(params).__name__}"]
        schema = self.parameters or {}
        if schema.get("type", "object") != "object":
            raise ValueError(f"Schema must be object type, got {schema.get('type')!r}")
        return self._validate(params, {**schema, "type": "object"}, "")

    def _validate(self, val: Any, schema: dict[str, Any], path: str) -> list[str]:
        t, label = schema.get("type"), path or "parameter"
        if t == "integer" and (not isinstance(val, int) or isinstance(val, bool)):
            return [f"{label} should be integer"]
        if t == "number" and (not isinstance(val, self._TYPE_MAP[t]) or isinstance(val, bool)):
            return [f"{label} should be number"]
        if (
            t in self._TYPE_MAP
            and t not in ("integer", "number")
            and not isinstance(val, self._TYPE_MAP[t])
        ):
            return [f"{label} should be {t}"]

        errors = []
        if "enum" in schema and val not in schema["enum"]:
            errors.append(f"{label} must be one of {schema['enum']}")
        if t in ("integer", "number"):
            if "minimum" in schema and val < schema["minimum"]:
                errors.append(f"{label} must be >= {schema['minimum']}")
            if "maximum" in schema and val > schema["maximum"]:
                errors.append(f"{label} must be <= {schema['maximum']}")
        if t == "string":
            if "minLength" in schema and len(val) < schema["minLength"]:
                errors.append(f"{label} must be at least {schema['minLength']} chars")
            if "maxLength" in schema and len(val) > schema["maxLength"]:
                errors.append(f"{label} must be at most {schema['maxLength']} chars")
        if t == "object":
            props = schema.get("properties", {})
            for key in schema.get("required", []):
                if key not in val:
                    errors.append(f"missing required {path + '.' + key if path else key}")
            for key, value in val.items():
                if key in props:
                    next_path = f"{path}.{key}" if path else key
                    errors.extend(self._validate(value, props[key], next_path))
        if t == "array" and "items" in schema:
            for index, item in enumerate(val):
                next_path = f"{path}[{index}]" if path else f"[{index}]"
                errors.extend(self._validate(item, schema["items"], next_path))
        return errors

    def to_schema(self) -> dict[str, Any]:
        """Convert tool to OpenAI function schema format."""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
