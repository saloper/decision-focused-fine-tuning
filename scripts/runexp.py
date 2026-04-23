import yaml
import time
import os
import sys

# Ensure the parent directory is in the path to import from scripts.run
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.run import main

def run_experiments():
    # Store both configs in a list to iterate over
    base_configs = ["configs/baseline.yaml", "configs/decision.yaml"]
    dataset = "NEMA"
    model_sizes = ["moment-small", "moment-base", "moment-large"]
    model_features = ["-finetuned", "-lora"]
    # Ablation values for learning rate
    learning_rates = [0.01, 0.003, 0.0003, 0.0001]
    
    model_names = []

    for size in model_sizes:
        for feature in model_features:
            model_names.append(size + feature)
            
    base_version = "2" # Base version as used before
    results = []
    
    # Iterate through each configuration file
    for config_path in base_configs:
        config_name = os.path.splitext(os.path.basename(config_path))[0] # Extracts 'baseline' or 'decision'
        
        # Load the current base config
        with open(config_path, "r") as f:
            base_config = yaml.safe_load(f)
            
        for lr in learning_rates:
            # Create temp config for this learning rate
            temp_config = base_config.copy()
            temp_config["training"]["learning_rate"] = lr
            
            # Save temp config with the config name to avoid overlap
            temp_config_path = f"configs/temp/{config_name}_lr_{lr}.yaml"
            os.makedirs(os.path.dirname(temp_config_path), exist_ok=True)
            with open(temp_config_path, "w") as f:
                yaml.dump(temp_config, f)
                
            for model_name in model_names:
                # Include the config name in the version string
                version = f"{base_version}_{config_name}_lr_{lr}"
                
                print(f"\n{'='*60}")
                print(f"Starting experiment: Config={config_name}, Model={model_name}, LR={lr}...")
                print(f"{'='*60}")
                
                start_time = time.time()
                
                try:
                    main(config_path=temp_config_path, model_name=model_name, dataset_name=dataset, version=version)
                except Exception as e:
                    print(f"Experiment failed with error: {e}")
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                # Format time nicely
                hours, rem = divmod(elapsed_time, 3600)
                minutes, seconds = divmod(rem, 60)
                elapsed_str = f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"
                
                print(f"Finished experiment: Config={config_name}, Model={model_name}, LR={lr}. Time taken: {elapsed_str}")
                
                results.append({
                    "Config": config_name,
                    "Model": model_name,
                    "Learning Rate": lr,
                    "Time Taken": elapsed_str
                })
                    
            # Cleanup temp config after running all models for this LR and config
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)

    # Print summary table at the end
    print("\n" + "="*80)
    print(f"{'EXPERIMENT RUN TIMES SUMMARY':^80}")
    print("="*80)
    print(f"{'Config':<15} | {'Model':<25} | {'Learning Rate':<15} | {'Time Taken':<15}")
    print("-" * 80)
    for res in results:
        print(f"{res['Config']:<15} | {res['Model']:<25} | {res['Learning Rate']:<15} | {res['Time Taken']:<15}")
    print("="*80)

if __name__ == "__main__":
    run_experiments()

# ================================================================================
#                           EXPERIMENT RUN TIMES SUMMARY                          
# ================================================================================
# Config          | Model                     | Learning Rate   | Time Taken     
# --------------------------------------------------------------------------------
# baseline        | moment-small-finetuned    | 0.001           | 00:02:26.81    
# baseline        | moment-small-lora         | 0.001           | 00:01:59.55    
# baseline        | moment-base-finetuned     | 0.001           | 00:02:32.55    
# baseline        | moment-base-lora          | 0.001           | 00:02:29.17    
# baseline        | moment-large-finetuned    | 0.001           | 00:02:59.55    
# baseline        | moment-large-lora         | 0.001           | 00:03:14.89    
# baseline        | moment-small-finetuned    | 0.0003          | 00:04:17.86    
# baseline        | moment-small-lora         | 0.0003          | 00:04:43.49    
# baseline        | moment-base-finetuned     | 0.0003          | 00:05:12.58    
# baseline        | moment-base-lora          | 0.0003          | 00:04:44.60    
# baseline        | moment-large-finetuned    | 0.0003          | 00:06:19.52    
# baseline        | moment-large-lora         | 0.0003          | 00:07:20.26    
# baseline        | moment-small-finetuned    | 0.0001          | 00:09:47.81    
# baseline        | moment-small-lora         | 0.0001          | 00:10:02.64    
# baseline        | moment-base-finetuned     | 0.0001          | 00:11:04.53    
# baseline        | moment-base-lora          | 0.0001          | 00:11:34.34    
# baseline        | moment-large-finetuned    | 0.0001          | 00:12:52.40    
# baseline        | moment-large-lora         | 0.0001          | 00:14:44.97    
# decision        | moment-small-finetuned    | 0.001           | 00:01:44.31    
# decision        | moment-small-lora         | 0.001           | 00:01:46.37    
# decision        | moment-base-finetuned     | 0.001           | 00:02:07.54    
# decision        | moment-base-lora          | 0.001           | 00:01:56.81    
# decision        | moment-large-finetuned    | 0.001           | 00:03:11.91    
# decision        | moment-large-lora         | 0.001           | 00:02:56.08    
# decision        | moment-small-finetuned    | 0.0003          | 00:02:51.49    
# decision        | moment-small-lora         | 0.0003          | 00:02:42.70    
# decision        | moment-base-finetuned     | 0.0003          | 00:03:11.91    
# decision        | moment-base-lora          | 0.0003          | 00:02:42.06    
# decision        | moment-large-finetuned    | 0.0003          | 00:04:39.56    
# decision        | moment-large-lora         | 0.0003          | 00:04:15.11    
# decision        | moment-small-finetuned    | 0.0001          | 00:06:43.48    
# decision        | moment-small-lora         | 0.0001          | 00:05:03.84    
# decision        | moment-base-finetuned     | 0.0001          | 00:06:17.96    
# decision        | moment-base-lora          | 0.0001          | 00:05:57.82    
# decision        | moment-large-finetuned    | 0.0001          | 00:06:47.14    
# decision        | moment-large-lora         | 0.0001          | 00:08:35.97    
# ================================================================================