import os
import sys
import time
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.run import main


def build_model_names():
    families = ["linear", "mlp", "lstm", "gru", "tcn"]
    variants = ["small", "base", "large"]

    model_names = []
    for family in families:
        for variant in variants:
            model_names.append(f"{family}-{variant}")
    return model_names


def run_single_experiment(config_path, model_name, dataset_name, version, config_name, lr):
    start_time = time.time()
    status = "success"
    error = ""

    try:
        main(
            config_path=config_path,
            model_name=model_name,
            dataset_name=dataset_name,
            version=version,
        )
    except Exception as e:
        status = "failed"
        error = str(e)

    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    elapsed_str = f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"

    return {
        "Config": config_name,
        "Model": model_name,
        "Learning Rate": lr,
        "Time Taken": elapsed_str,
        "Status": status,
        "Error": error,
    }


def run_experiments(max_workers=None):
    base_configs = ["configs/baseline.yaml", "configs/decision.yaml"]
    dataset = "NEMA"
    learning_rates = [0.01, 0.001, 0.0001]
    model_names = build_model_names()
    base_version = "2"

    if max_workers is None:
        max_workers = int(os.environ.get("RUNEXP2_WORKERS", "2"))
    max_workers = max(1, max_workers)
    print(f"Running with max_workers={max_workers}")

    results = []

    for config_path in base_configs:
        config_name = os.path.splitext(os.path.basename(config_path))[0]

        with open(config_path, "r") as f:
            base_config = yaml.safe_load(f)

        for lr in learning_rates:
            temp_config = base_config.copy()
            temp_config["training"]["learning_rate"] = lr

            temp_config_path = f"configs/temp/{config_name}_lr_{lr}.yaml"
            os.makedirs(os.path.dirname(temp_config_path), exist_ok=True)
            with open(temp_config_path, "w") as f:
                yaml.dump(temp_config, f)

            version = f"{base_version}_{config_name}_lr_{lr}"
            print("version: ", version)
            print(f"\n{'=' * 70}")
            print(f"Dispatching parallel jobs: Config={config_name}, LR={lr}, Models={len(model_names)}")
            print(f"{'=' * 70}")

            futures = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for model_name in model_names:
                    futures.append(
                        executor.submit(
                            run_single_experiment,
                            temp_config_path,
                            model_name,
                            dataset,
                            version,
                            config_name,
                            lr,
                        )
                    )

                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    if result["Status"] == "failed":
                        print(
                            f"Experiment failed: Config={result['Config']}, Model={result['Model']}, "
                            f"LR={result['Learning Rate']} | Error: {result['Error']}"
                        )
                    print(
                        f"Finished experiment: Config={result['Config']}, Model={result['Model']}, "
                        f"LR={result['Learning Rate']}. Time taken: {result['Time Taken']}"
                    )

            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)

    print("\n" + "=" * 90)
    print(f"{'NON-TSFM EXPERIMENT RUN TIMES SUMMARY':^90}")
    print("=" * 90)
    print(f"{'Config':<15} | {'Model':<25} | {'Learning Rate':<15} | {'Time Taken':<15} | {'Status':<10}")
    print("-" * 90)
    for res in results:
        print(
            f"{res['Config']:<15} | {res['Model']:<25} | "
            f"{res['Learning Rate']:<15} | {res['Time Taken']:<15} | {res['Status']:<10}"
        )
    print("=" * 90)


if __name__ == "__main__":
    try:
        run_experiments(12)
    except KeyboardInterrupt:
        print("Experiment run interrupted by user.")
    except Exception as e:
        print(f"Error with 12 workers: {e}. Retrying with 6 workers...")
        try:
            run_experiments(6)
        except KeyboardInterrupt:
            print("Experiment run interrupted by user.")
        except Exception as e:
            print(f"Error with 6 workers: {e}. Retrying with 4 workers...")
            try:
                run_experiments(4)
            except KeyboardInterrupt:
                print("Experiment run interrupted by user.")
            except Exception as e:
                print(f"Error with 4 workers: {e}. Retrying with 2 workers...")
                try:
                    run_experiments(2)
                except KeyboardInterrupt:
                    print("Experiment run interrupted by user.")
                except Exception as e:
                    print(f"Error with 2 workers: {e}. Giving up.")
                    raise