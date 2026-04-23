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
# baseline        | moment-small-finetuned    | 0.0001          | 00:16:56.77    
# baseline        | moment-small-lora         | 0.0001          | 00:17:05.61    
# baseline        | moment-base-finetuned     | 0.0001          | 00:17:26.16    
# baseline        | moment-base-lora          | 0.0001          | 00:17:49.28    
# baseline        | moment-large-finetuned    | 0.0001          | 00:19:20.81    
# baseline        | moment-large-lora         | 0.0001          | 00:19:54.53    
# baseline        | moment-small-finetuned    | 0.0003          | 00:15:10.50    
# baseline        | moment-small-lora         | 0.0003          | 00:15:41.56    
# baseline        | moment-base-finetuned     | 0.0003          | 00:16:52.40    
# baseline        | moment-base-lora          | 0.0003          | 00:15:21.11    
# baseline        | moment-large-finetuned    | 0.0003          | 00:16:18.23    
# baseline        | moment-large-lora         | 0.0003          | 00:19:34.68    
# baseline        | moment-small-finetuned    | 0.001           | 00:10:01.48    
# baseline        | moment-small-lora         | 0.001           | 00:08:29.27    
# baseline        | moment-base-finetuned     | 0.001           | 00:08:28.09    
# baseline        | moment-base-lora          | 0.001           | 00:10:06.31    
# baseline        | moment-large-finetuned    | 0.001           | 00:09:38.34    
# baseline        | moment-large-lora         | 0.001           | 00:10:16.41    
# decision        | moment-small-finetuned    | 0.0001          | 00:13:12.70    
# decision        | moment-small-lora         | 0.0001          | 00:19:46.42    
# decision        | moment-base-finetuned     | 0.0001          | 00:11:25.44    
# decision        | moment-base-lora          | 0.0001          | 00:11:07.90    
# decision        | moment-large-finetuned    | 0.0001          | 00:07:30.51    
# decision        | moment-large-lora         | 0.0001          | 00:16:22.58    
# decision        | moment-small-finetuned    | 0.0003          | 00:14:38.34    
# decision        | moment-small-lora         | 0.0003          | 00:12:10.56    
# decision        | moment-base-finetuned     | 0.0003          | 00:08:28.99    
# decision        | moment-base-lora          | 0.0003          | 00:09:26.59    
# decision        | moment-large-finetuned    | 0.0003          | 00:12:12.13    
# decision        | moment-large-lora         | 0.0003          | 00:10:10.21    
# decision        | moment-small-finetuned    | 0.001           | 00:12:30.50    
# decision        | moment-small-lora         | 0.001           | 00:08:12.11    
# decision        | moment-base-finetuned     | 0.001           | 00:11:28.23    
# decision        | moment-base-lora          | 0.001           | 00:16:40.29    
# decision        | moment-large-finetuned    | 0.001           | 00:17:29.21    
# decision        | moment-large-lora         | 0.001           | 00:10:22.33    
# ================================================================================