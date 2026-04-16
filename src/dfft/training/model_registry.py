

from src.dfft.models.moment import MOMENTForecaster
from src.dfft.models.momentLora import MOMENTLoRAForecaster
from src.dfft.models.moment_time import MOMENTTimeForecaster
from src.dfft.models.timesfm import TimesFMZeroShot, TimesFMLinearHead, TimesFMLoRA

def get_moment_model(size, model_type, config_kwargs):
    size_map = {
        "small": "AutonLab/MOMENT-1-small",
        "base": "AutonLab/MOMENT-1-base",
        "large": "AutonLab/MOMENT-1-large"
    }
    model_name = size_map.get(size)
    if model_name is None:
        raise ValueError(f"Unknown MOMENT size: {size}")

    # Defaults
    params = {
        "model_name": model_name,
        "input_len": 512,
        "output_len": 24,
        "random_seed": 42,
    }

    if model_type == "lora":
        params.update({
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "trainable": True
        })
        params.update({k: v for k, v in config_kwargs.items() if k in params})
        model = MOMENTLoRAForecaster(
            input_len=params["input_len"],
            output_len=params["output_len"],
            model_name=params["model_name"],
            r=params["r"],
            lora_alpha=params["lora_alpha"],
            lora_dropout=params["lora_dropout"],
            random_seed=params["random_seed"]
        )
        return model, params["trainable"]
    elif model_type == "finetuned":
        params.update({
            "freeze_linear": False,
            "trainable": True
        })
        params.update({k: v for k, v in config_kwargs.items() if k in params})
        model = MOMENTForecaster(
            input_len=params["input_len"],
            output_len=params["output_len"],
            model_name=params["model_name"],
            freeze_linear=params["freeze_linear"],
            random_seed=params["random_seed"]
        )
        return model, params["trainable"]
    elif model_type == "time":
        params.update({
            "freeze_linear": True,
            "trainable": True
        })
        params.update({k: v for k, v in config_kwargs.items() if k in params})
        model = MOMENTTimeForecaster(
            input_len=params["input_len"],
            output_len=params["output_len"],
            model_name=params["model_name"],
            freeze_linear=params["freeze_linear"],
            freeze_encoder=True,
            random_seed=params["random_seed"]
        )
        return model, params["trainable"]
    elif model_type == "time-finetuned":
        params.update({
            "freeze_linear": False,
            "trainable": True
        })
        params.update({k: v for k, v in config_kwargs.items() if k in params})
        model = MOMENTTimeForecaster(
            input_len=params["input_len"],
            output_len=params["output_len"],
            model_name=params["model_name"],
            freeze_linear=params["freeze_linear"],
            freeze_encoder=True,
            random_seed=params["random_seed"]
        )
        return model, params["trainable"]
    elif model_type == "zeroshot":
        params.update({
            "freeze_linear": True,
            "trainable": False
        })
        params.update({k: v for k, v in config_kwargs.items() if k in params})
        model = MOMENTForecaster(
            input_len=params["input_len"],
            output_len=params["output_len"],
            model_name=params["model_name"],
            freeze_linear=params["freeze_linear"],
            random_seed=params["random_seed"]
        )
        return model, params["trainable"]
    else:
        raise ValueError(f"Unknown MOMENT model_type: {model_type}")

def get_timesfm_model(size, model_type, config_kwargs):
    """
    TimesFM model factory
    """
    # Use SAME model for all sizes as requested
    model_name = "google/timesfm-2.5-200m-transformers"

    # Base parameters that all TimesFM models require
    params = {
        "model_name": model_name,
        "input_len": config_kwargs.get("input_len", 512),
        "output_len": config_kwargs.get("output_len", 24),
        "random_seed": config_kwargs.get("random_seed", 42),
    }

    # -------------------------
    # ZERO-SHOT
    # -------------------------
    if model_type == "zeroshot":
        model = TimesFMZeroShot(**params)
        return model, False

    # -------------------------
    # NATIVE HEAD / FINETUNED
    # -------------------------
    elif model_type in ["linear", "finetuned"]:
        model = TimesFMLinearHead(**params)
        return model, True

    # -------------------------
    # LORA
    # -------------------------
    elif model_type == "lora":
        params.update({
            "r": config_kwargs.get("r", 8),
            "lora_alpha": config_kwargs.get("lora_alpha", 16),
            "lora_dropout": config_kwargs.get("lora_dropout", 0.05),
        })
        print(params)
        model = TimesFMLoRA(**params)
        return model, True

    else:
        raise ValueError(f"Unknown TimesFM model_type: {model_type}")


def get_model(model_name: str, config_kwargs: dict):
    # Modular parsing for MOMENT models
    if model_name.startswith("moment"):
        # e.g. moment-small-lora, moment-base-finetuned, moment-large, moment-small-time, moment-small-time-finetuned
        parts = model_name.split("-")
        if len(parts) == 3:
            _, size, model_type = parts
        elif len(parts) == 2:
            _, size = parts
            model_type = "zeroshot"  # default type
        elif len(parts) == 4 and parts[2] == "time" and parts[3] == "finetuned":
            # moment-small-time-finetuned
            _, size, t1, t2 = parts
            model_type = f"{t1}-{t2}"
        else:
            raise ValueError(f"Unknown MOMENT model name format: {model_name}")
        return get_moment_model(size, model_type, config_kwargs)

    elif model_name.startswith("timesfm"):
    # Examples:
    # timesfm-small
    # timesfm-base-lora
    # timesfm-large-finetuned

        parts = model_name.split("-")

        if len(parts) == 3:
            _, size, model_type = parts

        elif len(parts) == 2:
            _, size = parts
            model_type = "zeroshot"

        else:
            raise ValueError(f"Unknown TimesFM model name format: {model_name}")

        return get_timesfm_model(size, model_type, config_kwargs)
    else:
        raise ValueError(f"Unknown model variation selected: {model_name}.")


