



def get_moment_model(size, model_type, config_kwargs):
    
    from dfft.models.moment import MOMENTForecaster
    from dfft.models.momentLora import MOMENTLoRAForecaster
    from dfft.models.moment_time import MOMENTTimeForecaster
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
    from dfft.models.timesfm import TimesFMZeroShot, TimesFMLinearHead, TimesFMLoRA
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


def get_classical_model(model_family, model_variant, config_kwargs):
    """
    Classical neural forecasting model factory.
    Supported families: linear, mlp, lstm, gru, tcn
    Example model names consumed by parser:
      - mlp
      - mlp-small
      - lstm-base
      - tcn-large
    """
    size_map = {
        "small": {
            "hidden_size": 64,
            "hidden_channels": 32,
            "num_layers": 1,
            "dropout": 0.05,
            "mlp_hidden_dims": [128, 64],
        },
        "base": {
            "hidden_size": 128,
            "hidden_channels": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "mlp_hidden_dims": [256, 128],
        },
        "large": {
            "hidden_size": 256,
            "hidden_channels": 128,
            "num_layers": 3,
            "dropout": 0.15,
            "mlp_hidden_dims": [512, 256],
        },
    }

    variant = (model_variant or "base").lower()
    if variant not in size_map:
        raise ValueError(f"Unknown classical model variant: {variant}")

    preset = size_map[variant]
    input_len = config_kwargs.get("input_len", 512)
    output_len = config_kwargs.get("output_len", 24)
    in_channels = config_kwargs.get("in_channels", 1)

    if model_family == "linear":
        from dfft.models.linear import LinearForecaster

        model = LinearForecaster(
            input_len=input_len,
            output_len=output_len,
            in_channels=in_channels,
        )
        return model, True

    if model_family == "mlp":
        from dfft.models.mlp import MLPForecaster

        model = MLPForecaster(
            input_len=input_len,
            output_len=output_len,
            in_channels=in_channels,
            hidden_dims=config_kwargs.get("hidden_dims", preset["mlp_hidden_dims"]),
            dropout=config_kwargs.get("dropout", preset["dropout"]),
        )
        return model, True

    if model_family == "lstm":
        from dfft.models.lstm import LSTMForecaster

        model = LSTMForecaster(
            input_len=input_len,
            output_len=output_len,
            input_dim=in_channels,
            hidden_size=config_kwargs.get("hidden_size", preset["hidden_size"]),
            num_layers=config_kwargs.get("num_layers", preset["num_layers"]),
            dropout=config_kwargs.get("dropout", preset["dropout"]),
            bidirectional=config_kwargs.get("bidirectional", False),
        )
        return model, True

    if model_family == "gru":
        from dfft.models.gru import GRUForecaster

        model = GRUForecaster(
            input_len=input_len,
            output_len=output_len,
            input_dim=in_channels,
            hidden_size=config_kwargs.get("hidden_size", preset["hidden_size"]),
            num_layers=config_kwargs.get("num_layers", preset["num_layers"]),
            dropout=config_kwargs.get("dropout", preset["dropout"]),
            bidirectional=config_kwargs.get("bidirectional", False),
        )
        return model, True

    if model_family == "tcn":
        from dfft.models.tcn import TCNForecaster

        model = TCNForecaster(
            input_len=input_len,
            output_len=output_len,
            in_channels=in_channels,
            hidden_channels=config_kwargs.get("hidden_channels", preset["hidden_channels"]),
            num_layers=config_kwargs.get("num_layers", preset["num_layers"] + 1),
            kernel_size=config_kwargs.get("kernel_size", 3),
            dropout=config_kwargs.get("dropout", preset["dropout"]),
        )
        return model, True

    raise ValueError(f"Unknown classical model family: {model_family}")


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

    elif model_name.split("-")[0] in {"linear", "mlp", "lstm", "gru", "tcn"}:
        # Supported formats:
        # linear
        # linear-small
        # mlp-base
        # lstm-large
        parts = model_name.split("-")
        family = parts[0]

        if len(parts) == 1:
            variant = "base"
        elif len(parts) == 2:
            variant = parts[1]
        else:
            raise ValueError(f"Unknown classical model name format: {model_name}")

        return get_classical_model(family, variant, config_kwargs)
    else:
        raise ValueError(f"Unknown model variation selected: {model_name}.")


