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
    n_nodes = trial.suggest_categorical('n_nodes', [1, 10, 50, 100])
    lr = trial.suggest_categorical('lr', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    batch_size_ratio = trial.suggest_categorical('lr', [0.1, 0.3, 0.5, 0.8])
    ##################################



    # Update the configuration file with the new hyperparameters
    ######## Change paramters accordingly ########
    new_config_file_path = update_config_file(config_file_path, epochs=epochs, n_nodes=n_nodes, lr=lr, batch_size_ratio=batch_size_ratio)
    ##############################################


    # Run the experiment and get the result
    oracle_acc = run_experiment(new_config_file_path)

    return oracle_acc

def run_experiment(config_file):
      
    command = f"conda run -n GRTL python main.py {config_file}"
    subprocess.run(command, text=True, shell=True)  #RUN EXPERIMENT
    
    # NEED TO CHANGE FOR OPT
    base_folder = current_directory + "/output/results/optimization/TwitterGCN-b482cdc7f20a861ad62d177a2e8f0323"
    print(base_folder)
    # Find the most recent results path
    output_path = find_most_recent_results_path(base_folder)

    with open(output_path, 'r') as file:
        json_data = json.load(file)

    # Extract the 'Correctness' values
    explainer_correctness_values = json_data.get('Correctness', [])

    # Extract the 'Graph_Edit_Distance' values
    explainer_GED_values = json_data.get('Graph_Edit_Distance', [])

    # Maybe get oracle_calls?

    explainer_correctness = explainer_correctness_values.count(1)/len(explainer_correctness_values)
    explainer_GED = sum(explainer_GED_values)/len(explainer_GED_values)
    return explainer_GED, explainer_correctness

    

def find_most_recent_results_path(base_folder):
    # Get all subdirectories in the base folder
    subdirectories = [d for d in Path(base_folder).iterdir() if d.is_dir()]

    # Filter subdirectories that contain 'results_run--1.json'
    filtered_subdirectories = [
        d for d in subdirectories if (d / 'results_run--1.json').exists()
    ]

    # Sort the filtered subdirectories by modification time
    sorted_subdirectories = sorted(
        filtered_subdirectories, key=lambda d: d.stat().st_mtime, reverse=True
    )

    # Check if any subdirectories were found
    if sorted_subdirectories:
        most_recent_folder = sorted_subdirectories[0]
        results_json_path = most_recent_folder / 'results_run--1.json'
        return results_json_path

    return None


def update_config_file(config_file_path, epochs, batch_size_ratio, n_nodes, lr):
    config_directory = os.path.dirname(config_file_path)
    
    with open(config_file_path, 'r') as file:
        config = json.load(file)



    ######## Ajust others here #######

    # Update Oracle params
    explainer_params = config['explainers'][0]['parameters']
    explainer_params['epochs'] = epochs
    explainer_params['batch_size_ratio'] = batch_size_ratio
    explainer_params['lr'] = lr
    explainer_params['n_nodes'] = n_nodes

    ##################################



    # Save the updated configuration
    global iteration 
    output_dir = os.path.join(config_directory, 'oracle_opt')    
    os.makedirs(output_dir, exist_ok=True)
    updated_config_path = os.path.join(output_dir, os.path.basename(config_file_path).replace('.json', f'_epochs{epochs}_batch{batch_size}_{iteration}.json'))
    with open(updated_config_path, 'w') as file:
        json.dump(config, file, indent=2)
    iteration +=1
    return updated_config_path

            
if __name__ == "__main__":
    config_file_path = "config/submission/oracle_template_optimization/TWITTER-test.json"
    current_directory = os.getcwd()
    study = optuna.create_study(directions=['minimize', 'maximize'])
    study.optimize(objective, n_trials=2)  # You can adjust the number of trials



    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print(f"Value: {trial.value}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")