from __future__ import annotations

import os
from pathlib import Path

import forge_ui

from modules import scripts
from modules.processing import StableDiffusionProcessing

from anima3b.files import adapters
from anima3b.runtime import Anima3BRuntime

forge_ui.require(1, "sd-forge-anima-3-8b")

DEFAULT_ADAPTER = "Anima-3.8B-expanded_adapter.safetensors"


def _adapter_choices() -> list[str]:
    return list(adapters()) or [DEFAULT_ADAPTER]


def _info(params: dict, name: str):
    # Older versions wrote "Anima 3.8B ..." keys, which Forge's infotext parser reads as "8B ..." (no dots in keys)
    return params.get(f"Anima 3-8B {name}", params.get(f"8B {name}"))


def _restore_adapter(params: dict):
    # The runtime writes the file name only; map it back to the dropdown entry (which may include a subfolder)
    name = _info(params, "adapter")
    return next((choice for choice in _adapter_choices() if Path(choice).name == name), None)


def _restore_number(name: str):
    def restore(params: dict):
        value = _info(params, name)
        try:
            return None if value is None else float(value)
        except ValueError:
            return None

    return restore


# UI and infotext restore (forge_ui, sd-forge-ui-kit). The infotext itself is written by Anima3BRuntime.install(),
# which knows whether a bundled v2 connector replaced the selected adapter.
PANEL = forge_ui.Panel(
    "anima3b",
    "Anima 3.8B (Qwen3.5 / v2)",
    [
        forge_ui.Dropdown(
            "adapter", "Adapter", _adapter_choices, lambda: _adapter_choices()[0],
            restore=_restore_adapter, axis=False,
            info="Legacy v1 adapter, or a v2 connector sidecar (fixed strength 1.0). Bundled anima.3-8B-v2 "
            "checkpoints activate automatically and ignore this selector.",
            tooltip="The Qwen3.5-to-Anima adapter that turns the larger text encoder's output into conditioning the "
            "Anima diffusion model understands. A v1 adapter is blended in by the strength below; a v2 connector "
            "sidecar (e.g. Anima-3.8B-expanded_adapter) always runs at its trained strength. Checkpoints with a "
            "bundled v2 connector use it even when this extension is off.",
        ),
        forge_ui.Slider(
            "strength", "Adapter strength", 1.0, 0.0, 2.0, 0.05, restore=_restore_number("strength"),
            info="Legacy v1 only; bundled v2 is fixed at trained strength 1.0.",
            tooltip="How much of the adapter's conditioning is used (v1 adapters). 1 = as trained; lower drifts back "
            "towards the stock Anima encoder's reading of the prompt, higher exaggerates it (stronger style, more "
            "artefacts).",
        ),
        forge_ui.Checkbox(
            "negative", "Use adapter on negative prompt", restore=lambda d: _info(d, "negative strength") is not None,
            info="Off keeps negatives on the native Anima encoder.",
            tooltip="On: the negative prompt is encoded through the adapter too, so negatives are understood the same "
            "way as the positive prompt (often stronger negatives). Off: negatives use the stock encoder.",
        ),
        forge_ui.Slider(
            "negative_strength", "Negative adapter strength", 1.0, 0.0, 2.0, 0.05,
            restore=_restore_number("negative strength"),
            visible_if=lambda v: v["negative"], depends=("negative",),
            info="Adapter strength applied to the negative prompt.",
            tooltip="Like Adapter strength, for the negative prompt (v1 adapters; v2 is fixed at 1.0).",
        ),
    ],
    description="Anima 3.8B: encodes prompts with Qwen3.5 through an adapter / v2 connector for Anima models. Anima "
    "models only.",
    enabled_from=lambda d: _info(d, "adapter") is not None,
    preset_file=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "presets.json"),
    xyz_prefix="[Anima 3.8B]",
)


class Anima3BScript(scripts.Script):
    sorting_priority = 260209301

    def __init__(self):
        super().__init__()
        self.runtime = Anima3BRuntime()

    def title(self):
        return "Anima 3.8B"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return PANEL.ui(self, is_img2img)

    def process_batch(self, p: StableDiffusionProcessing, *args, **kwargs):
        v = PANEL.values(p, args)
        if not v["enabled"] and not self.runtime.is_v2_bundle(p.sd_model):
            return
        self.runtime.install(
            p,
            v["adapter"],
            v["strength"],
            v["negative_strength"] if v["negative"] else None,
        )

    def postprocess(self, p, processed, *args):
        self.runtime.restore(p)
