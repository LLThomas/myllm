from .qwen2_week1 import Qwen2ModelWeek1
from .qwen2_week2 import Qwen2ModelWeek2

def shortcut_name_to_full_name(shortcut_name: str):
    lower_shortcut_name = shortcut_name.lower()
    if lower_shortcut_name == "qwen2-0.5b":
        return "Qwen/Qwen2-0.5B-Instruct"
    else:
        return shortcut_name


def dispatch_model(model_name: str, hf_model, week: int, **kwargs):
    model_name = shortcut_name_to_full_name(model_name)
    if week == 1 and model_name.startswith("Qwen/Qwen2"):
        return Qwen2ModelWeek1(hf_model, **kwargs)
    elif week == 2 and model_name.startswith("Qwen/Qwen2"):
        return Qwen2ModelWeek2(hf_model, **kwargs)
    else:
        raise ValueError(f"{model_name} for week {week} not supported")