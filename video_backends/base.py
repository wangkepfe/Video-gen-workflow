"""Backend interface shared by every video model.

A backend is a Python module that exposes the attributes documented below.
This file documents the contract and provides a couple of helpers.
"""
from dataclasses import dataclass, asdict
from typing import Any, Optional


@dataclass
class Field:
    """Describes a single UI control / settings entry.

    Used both for rendering the form in the web app and for validating the
    request payload server-side.
    """
    id: str
    label: str
    type: str                       # "int" | "float" | "text" | "select" | "ratio_buttons"
    default: Any
    step: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    placeholder: Optional[str] = None
    options: Optional[list] = None  # for "select"
    presets: Optional[list] = None  # for "ratio_buttons": [{w, h, label}]
    help: Optional[str] = None
    advanced: bool = False

    def to_dict(self):
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


class MissingFiles(Exception):
    """Raised by load_models() when required model files are absent.

    The message should tell the user exactly which files are missing and
    point them at download_models.bat.
    """


# Backend module contract:
#
#   NAME            : str        unique key used by API and CLI
#   DISPLAY_NAME    : str        user-facing label
#   DESCRIPTION     : str        one-line tagline shown in the picker
#   DEFAULT_NEGATIVE: str        default negative prompt
#   FIELDS          : list[Field]
#
#   def required_files() -> list[tuple[str, str]]:
#       Each tuple is (human label, absolute path). Used by load_models() to
#       raise MissingFiles before any heavy work, and by the web UI to show
#       a clear setup hint.
#
#   def load_models() -> Any
#       Returns whatever opaque value will be passed back to generate().
#       Typically a dict so generate() can do `models["clip"]`, etc.
#
#   def generate(models, image_path, prompt, negative, settings, output_path) -> None
#       Runs the I2V pipeline and writes the video to `output_path`.
#       `settings` is a dict matching the FIELDS schema, already validated.
