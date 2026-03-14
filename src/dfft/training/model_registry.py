from src.dfft.models.moment import MOMENTForecaster

def get_model(model_name: str, config_kwargs: dict):

    if model_name == "moment-small":
        params = {
            "model_name": "AutonLab/MOMENT-1-small",
            "input_len": 512,
            "output_len": 24,
            "freeze_linear": True,
            "random_seed": 42,
            "trainable": False
        }
        params.update({k: v for k, v in config_kwargs.items() if k in params})
        model = MOMENTForecaster(
            input_len=params["input_len"],
            output_len=params["output_len"],
            model_name=params["model_name"],
            freeze_linear=params["freeze_linear"],
            random_seed=params["random_seed"]
        )
        return model, params["trainable"]
    elif model_name == "moment-base":
        params = {
            "model_name": "AutonLab/MOMENT-1-base",
            "input_len": 512,
            "output_len": 24,
            "freeze_linear": True,
            "random_seed": 42,
            "trainable": False
        }
        params.update({k: v for k, v in config_kwargs.items() if k in params})
        model = MOMENTForecaster(
            input_len=params["input_len"],
            output_len=params["output_len"],
            model_name=params["model_name"],
            freeze_linear=params["freeze_linear"],
            random_seed=params["random_seed"]
        )
        return model, params["trainable"]
    elif model_name == "moment-large":
        params = {
            "model_name": "AutonLab/MOMENT-1-large",
            "input_len": 512,
            "output_len": 24,
            "freeze_linear": True,
            "random_seed": 42,
            "trainable": False
        }
        params.update({k: v for k, v in config_kwargs.items() if k in params})
        model = MOMENTForecaster(
            input_len=params["input_len"],
            output_len=params["output_len"],
            model_name=params["model_name"],
            freeze_linear=params["freeze_linear"],
            random_seed=params["random_seed"]
        )
        return model, params["trainable"]
    elif model_name == "moment-small-finetuned":
        params = {
            "model_name": "AutonLab/MOMENT-1-small",
            "input_len": 512,
            "output_len": 24,
            "freeze_linear": False,
            "random_seed": 42,
            "trainable": True
        }
        params.update({k: v for k, v in config_kwargs.items() if k in params})
        model = MOMENTForecaster(
            input_len=params["input_len"],
            output_len=params["output_len"],
            model_name=params["model_name"],
            freeze_linear=params["freeze_linear"],
            random_seed=params["random_seed"]
        )
        return model, params["trainable"]
    elif model_name == "moment-base-finetuned":
        params = {
            "model_name": "AutonLab/MOMENT-1-base",
            "input_len": 512,
            "output_len": 24,
            "freeze_linear": False,
            "random_seed": 42,
            "trainable": True
        }
        params.update({k: v for k, v in config_kwargs.items() if k in params})
        model = MOMENTForecaster(
            input_len=params["input_len"],
            output_len=params["output_len"],
            model_name=params["model_name"],
            freeze_linear=params["freeze_linear"],
            random_seed=params["random_seed"]
        )
        return model, params["trainable"]
    elif model_name == "moment-large-finetuned":
        params = {
            "model_name": "AutonLab/MOMENT-1-large",
            "input_len": 512,
            "output_len": 24,
            "freeze_linear": False,
            "random_seed": 42,
            "trainable": True
        }
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
        raise ValueError(f"Unknown model variation selected: {model_name}.")


