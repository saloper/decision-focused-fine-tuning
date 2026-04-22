import yaml
import time
import os
import sys

# Ensure the parent directory is in the path to import from scripts.run
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.run import main

def run_experiments():
    base_config_path = "configs/baseline.yaml"
    base_config_path = "configs/decision.yaml"
    dataset = "NEMA"
    model_sizes = ["moment-small", "moment-base", "moment-large"]
    model_features = ["-finetuned", "-lora"]
    # Ablation values for learning rate
    learning_rates = [0.01, 0.003, 0.0003, 0.0001]
    
    model_names = []

    for size in model_sizes:
        for feature in model_features:
            print(f"Adding model: {size + feature}")
            model_names.append(size + feature)
    

    
    base_version = "2" # Base version as used before
    
    # Load the base config
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)
        
    results = []
    
    for lr in learning_rates:
        # Create temp config for this learning rate
        temp_config = base_config.copy()
        temp_config["training"]["learning_rate"] = lr
        
        # Save temp config
        temp_config_path = f"configs/temp/decision_lr_{lr}.yaml"
        os.makedirs(os.path.dirname(temp_config_path), exist_ok=True)
        with open(temp_config_path, "w") as f:
            yaml.dump(temp_config, f)
            
        for model_name in model_names:
            version = f"{base_version}_lr_{lr}"
            
            print(f"\n{'='*50}")
            print(f"Starting experiment: Model={model_name}, LR={lr}...")
            print(f"{'='*50}")
            
            start_time = time.time()
            
            try:
                1
                main(config_path=temp_config_path, model_name=model_name, dataset_name=dataset, version=version)
            except Exception as e:
                print(f"Experiment failed with error: {e}")
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Format time nicely
            hours, rem = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(rem, 60)
            elapsed_str = f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"
            
            print(f"Finished experiment: Model={model_name}, LR={lr}. Time taken: {elapsed_str}")
            
            results.append({
                "Model": model_name,
                "Learning Rate": lr,
                "Time Taken": elapsed_str
            })
                
        # Cleanup temp config after running all models for this LR
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

    # Print summary table at the end
    print("\n" + "="*60)
    print(f"{'EXPERIMENT RUN TIMES SUMMARY':^60}")
    print("="*60)
    print(f"{'Model':<25} | {'Learning Rate':<15} | {'Time Taken':<15}")
    print("-" * 60)
    for res in results:
        print(f"{res['Model']:<25} | {res['Learning Rate']:<15} | {res['Time Taken']:<15}")
    print("="*60)

if __name__ == "__main__":
    run_experiments()


#                 EXPERIMENT RUN TIMES SUMMARY
# ============================================================
# Model                     | Learning Rate   | Time Taken
# ------------------------------------------------------------
# moment-small-finetuned    | 0.01            | 00:19:18.26
# moment-small-lora         | 0.01            | 00:17:09.36
# moment-base-finetuned     | 0.01            | 00:21:01.22
# moment-base-lora          | 0.01            | 00:17:30.26
# moment-large-finetuned    | 0.01            | 00:20:04.47
# moment-large-lora         | 0.01            | 00:28:21.40
# moment-small-finetuned    | 0.003           | 00:19:07.04
# moment-small-lora         | 0.003           | 00:20:37.16
# moment-base-finetuned     | 0.003           | 00:15:28.46
# moment-base-lora          | 0.003           | 00:17:44.84
# moment-large-finetuned    | 0.003           | 00:32:27.11
# moment-large-lora         | 0.003           | 00:20:22.36
# moment-small-finetuned    | 0.0003          | 00:30:12.75
# moment-small-lora         | 0.0003          | 00:28:22.07
# moment-base-finetuned     | 0.0003          | 00:25:01.08
# moment-base-lora          | 0.0003          | 00:27:07.99
# moment-large-finetuned    | 0.0003          | 00:30:45.60
# moment-large-lora         | 0.0003          | 00:37:06.05
# moment-small-finetuned    | 0.0001          | 00:49:03.30
# moment-small-lora         | 0.0001          | 00:50:48.67
# moment-base-finetuned     | 0.0001          | 00:51:22.41
# moment-base-lora          | 0.0001          | 00:52:26.27
# moment-large-finetuned    | 0.0001          | 01:02:49.97
# moment-large-lora         | 0.0001          | 01:03:01.74
# ============================================================