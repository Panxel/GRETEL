import subprocess
import json
from pathlib import Path
import jsoncomment
import optuna
import os

iteration = 0

def objective(trial):
    # Define the hyperparameters to optimize

    ######## Ajust others here #######
    epochs = trial.suggest_categorical('epochs', [25, 50, 75, 100, 150])
    
    ##################################



    # Update the configuration file with the new hyperparameters
    ######## Change paramters accordingly ########
    new_config_file_path = update_config_file(config_file_path, epochs)
    ##############################################


    # Run the experiment and get the result
    oracle_acc = run_experiment(new_config_file_path)

    return oracle_acc

def run_experiment(config_file):
      
    command = f"conda run -n GRTL python main.py {config_file}"
    subprocess.run(command, text=True, shell=True)  #RUN EXPERIMENT
    
    
    base_folder = current_directory + "/output/results/explainer_opti"
    #print(base_folder)
    # Find the most recent results path
    output_path = find_most_recent_results_path(base_folder)

    with open(output_path, 'r') as file:
        json_data = json.load(file)

    # Extract the 'Correctness' values
    explainer_correctness_values = json_data.get('Correctness', [])

    # Extract the 'Graph_Edit_Distance' values
    explainer_GED_values = json_data.get('Graph_Edit_Distance', [])

    explainer_correctness = explainer_correctness_values.count(1)/len(explainer_correctness_values)
    explainer_GED = sum(explainer_GED_values)/len(explainer_GED_values)
    return explainer_GED, explainer_correctness

def find_most_recent_results_path(base_folder):
    # Create a Path object for the base folder
    base_path = Path(base_folder)

    # Use rglob to recursively search for the specified file pattern
    results_paths = list(base_path.rglob('results_run_0_-1.json'))

    # Sort the results paths by modification time
    sorted_results_paths = sorted(results_paths, key=lambda p: p.stat().st_mtime, reverse=True)

    # Check if any results paths were found
    if sorted_results_paths:
        most_recent_path = sorted_results_paths[0]
        return most_recent_path

    return None

# This creates the new config file with the random parameters chosen in def objective
def update_config_file(config_file_path, epochs):
    config_directory = os.path.dirname(config_file_path)
    
    with open(config_file_path, 'r') as file:
        config = json.load(file)



    ######## Ajust others here #######
    # gccountergan example:
    # "explainers": [{"class": "src.explainer.generative.gcountergan.GCounteRGAN","parameters":{"epochs": 150}}],
    # Update Explainer params
    config['explainers'][0]['class'] = "src.explainer.generative.gcountergan.GCounteRGAN"
    config['explainers'][0]['parameters']['epochs'] = epochs
    #we didnt find more params here

    ##################################



    # Save the updated configuration
    global iteration 
    output_dir = os.path.join(config_directory, 'explainer_opt_gan')    
    os.makedirs(output_dir, exist_ok=True)
    updated_config_path = os.path.join(output_dir, os.path.basename(config_file_path).replace('.json', f'_gan_epochs{epochs}_{iteration}.json'))
    with open(updated_config_path, 'w') as file:
        json.dump(config, file, indent=2)
    iteration +=1
    return updated_config_path

            
if __name__ == "__main__":
    
    
    config_file_path = "config/submission/explainers_template_optimization/TWITTER-explainer_template.json" #add optimized oracle file here
    current_directory = os.getcwd()
    study = optuna.create_study(directions=['minimize', 'maximize'])
    study.optimize(objective, n_trials=10)  # adjust the number of trials



    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    best_trials = study.best_trials

    for idx, trial in enumerate(best_trials):
        values = trial.values  
        print(f"Trial {idx + 1} - Index: {trial.number}, Graph Edit Distance, Correctness : {values} ")
