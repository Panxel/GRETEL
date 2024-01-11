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
    epochs = trial.suggest_categorical('epochs', [1, 10, 50, 100])
    lr = trial.suggest_categorical('lr', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    lambda_cfe = trial.suggest_categorical('lambda_cfe', [0.01, 0.1, 0.5, 1])
    alpha = trial.suggest_categorical('alpha', [0.01,0.1, 0.5, 1])
    batch_size_ratio = trial.suggest_categorical('batch_size_ratio', [0.01, 0.1, 0.5, 1])
    ##################################



    # Update the configuration file with the new hyperparameters
    ######## Change paramters accordingly ########
    new_config_file_path = update_config_file(config_file_path, epochs, lr, lambda_cfe, alpha, batch_size_ratio)
    ##############################################


    # Run the experiment and get the result
    oracle_acc = run_experiment(new_config_file_path)

    return oracle_acc

def run_experiment(config_file):
      
    command = f"conda run -n GRTL python main.py {config_file}"
    subprocess.run(command, text=True, shell=True)  #RUN EXPERIMENT
    
    
    base_folder = current_directory + "/output/results/optimization"
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
    results_paths = list(base_path.rglob('results_run--1.json'))

    # Sort the results paths by modification time
    sorted_results_paths = sorted(results_paths, key=lambda p: p.stat().st_mtime, reverse=True)

    # Check if any results paths were found
    if sorted_results_paths:
        most_recent_path = sorted_results_paths[0]
        return most_recent_path

    return None

# This creates the new config file with the random parameters chosen in def objective
def update_config_file(config_file_path, epochs, lr, lambda_cfe, alpha, batch_size_ratio):
    config_directory = os.path.dirname(config_file_path)
    
    with open(config_file_path, 'r') as file:
        config = json.load(file)



    ######## Ajust others here #######
    # clear example:
    # "explainers": [{"class": "src.explainer.generative.clear.CLEARExplainer","parameters":{ "epochs": 100, "lr": 0.01, "lambda_cfe": 0.1, "alpha": 0.4, "batch_size_ratio": 0.15 }}],
    # Update Explainer params
    config['explainers'][0]['class'] = "src.explainer.generative.clear.CLEARExplainer"
    explainer_params = config['explainers'][0]['parameters']
    explainer_params['epochs'] = epochs
    explainer_params['lr'] = lr
    explainer_params['lambda_cfe'] = lambda_cfe
    explainer_params['alpha'] = alpha
    explainer_params['batch_size_ratio'] = batch_size_ratio
    #we just use this subset of params
    ##################################



    # Save the updated configuration
    global iteration 
    output_dir = os.path.join(config_directory, 'explainer_opt_irand')    
    os.makedirs(output_dir, exist_ok=True)
    updated_config_path = os.path.join(output_dir, os.path.basename(config_file_path).replace('.json', f'_clear_epochs{epochs}_lr{lr}_{iteration}.json'))
    with open(updated_config_path, 'w') as file:
        json.dump(config, file, indent=2)
    iteration +=1
    return updated_config_path

            
if __name__ == "__main__":
    
    
    config_file_path = "config/submission/explainers_template_optimization/TWITTER_explainer_template.json" #add optimized oracle file here
    current_directory = os.getcwd()
    study = optuna.create_study(directions=['minimize', 'maximize'])
    study.optimize(objective, n_trials=20)  # adjust the number of trials



    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    best_trials = study.best_trials

    for idx, trial in enumerate(best_trials):
        values = trial.values  
        print(f"Trial {idx + 1} - Index: {trial.number}, Graph Edit Distance, Correctness : {values} ")
