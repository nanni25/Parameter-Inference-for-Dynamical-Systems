import argparse
import subprocess
import sys
import os
import json

def run_pipeline(args):
    print("========================================")
    print("  STARTING PIPELINE ")
    print("========================================\n")

    # Setup directories and dynamic file paths
    os.makedirs(args.cured_dir, exist_ok=True)
    os.makedirs(args.targets_dir, exist_ok=True) # <-- Creates the Targets folder
    
    base_filename = os.path.basename(args.input_model)
    cured_model_path = os.path.join(args.cured_dir, f"Cured_{base_filename}")
    
    # Extract the name without extension to create a unique JSON file name
    model_name_no_ext = os.path.splitext(base_filename)[0]
    targets_path = os.path.join(args.targets_dir, f"{model_name_no_ext}_targets.json")

    try:
        # Phase 1: The Target Generator (LLM)
        print(">>> Generating Targets via LLM...")
        subprocess.run([sys.executable, "targets.py", 
                        "--input_sbml", args.input_model, 
                        "--output_json", targets_path], check=True)

        # Phase 2: The SBML Modifier 
        print(">>> Modifying SBML file...")
        subprocess.run([sys.executable, "modifier.py", 
                        "--input_sbml", args.input_model, 
                        "--output_sbml", cured_model_path], check=True)

        # Phase 3: The Optimizer
        print(">>> Running the Optimizer...")
        optimizer_cmd = [
            sys.executable, "optimizer.py",
            "--model_path", cured_model_path,
            "--targets_path", targets_path,
            "--sim_time", str(args.sim_time),
            "--sim_steps", str(args.sim_steps),
            "--pop_size", str(args.pop_size),
            "--generations", str(args.generations),
            "--learning_rate", str(args.learning_rate),
            "--sigma", str(args.sigma),
            "--patience", str(args.patience),
            "--min_delta", str(args.min_delta)
        ]
        subprocess.run(optimizer_cmd, check=True)

        print("========================================")
        print("  FINISHED SUCCESSFULLY  ")
        print("========================================")

    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Program crashed during execution. {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SBML Optimization Pipeline.")
    
    # Take a single config file instead of all hyperparameters
    parser.add_argument(
        "--config", 
        type=str, 
        default=os.path.join("Configurations", "config.json"), 
        help="Path to the JSON configuration file."
    )
    
    initial_args = parser.parse_args()
    
    # Validate that the config file exists
    if not os.path.exists(initial_args.config):
        print(f"[ERROR] Configuration file not found at: {initial_args.config}")
        sys.exit(1)
        
    # Load the JSON data
    with open(initial_args.config, "r") as f:
        config_data = json.load(f)
        
    # Convert the dictionary back into an object to maintain compatibility with run_pipeline
    args = argparse.Namespace(**config_data)
    
    run_pipeline(args)