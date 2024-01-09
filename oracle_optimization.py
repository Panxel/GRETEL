import subprocess
import json
from pathlib import Path
import jsoncomment
import optuna
import os

iteration = 0

def objective(trial):
    # Define the hyperparameters to optimize
    epochs = trial.suggest_categorical('epochs', [1, 10, 50, 100])
    batch_size = trial.suggest_categorical('batch_size', [1, 32, 64])
    optimizer = trial.suggest_categorical('optimizer', ["torch.optim.Adam", "torch.optim.RMSprop"])
    lr = trial.suggest_categorical('lr', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    loss_f = trial.suggest_categorical('loss_f', ["torch.nn.CrossEntropyLoss", "torch.nn.MSELoss"])
    reduction = trial.suggest_categorical('reduction', ["mean", "sum"])
    num_conv_layers = trial.suggest_categorical('num_conv_layers', [1, 2, 3, 4, 5])
    num_dense_layers = trial.suggest_categorical('num_dense_layers', [1, 2, 3])
    conv_booster = trial.suggest_categorical('conv_booster', [1.0, 2.0, 3.0])
    linear_decay = trial.suggest_categorical('linear_decay', [1.0, 1.5, 2.0])



    # Update the configuration file with the new hyperparameters
    new_config_file_path = update_config_file(config_file_path, epochs, batch_size, optimizer, lr, loss_f, reduction, num_conv_layers, num_dense_layers, conv_booster, linear_decay)

    # Run the experiment and get the result
    oracle_acc = run_experiment(new_config_file_path)

    return oracle_acc

def run_experiment(config_file):
      
    command = f"conda run -n GRTL python main.py {config_file}"
    subprocess.run(command, text=True, shell=True)  #RUN EXPERIMENT
    
    # NEED TO CHANGE FOR OPT
    base_folder = current_directory + "/output/results/optimization/TwitterGCN-b482cdc7f20a861ad62d177a2e8f0323"
    #print(base_folder)
    # Find the most recent results path
    output_path = find_most_recent_results_path(base_folder)

    with open(output_path, 'r') as file:
        json_data = json.load(file)

    # Extract the 'Oracle_Accuracy' values
    oracle_accuracy_values = json_data.get('Oracle_Accuracy', [])
    # Maybe get oracle_calls?
    oracle_acc = oracle_accuracy_values.count(1)/len(oracle_accuracy_values)
    return oracle_acc

    

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



def update_config_file(config_file_path, epochs, batch_size, optimizer, lr, loss_f, reduction, num_conv_layers, num_dense_layers, conv_booster, linear_decay):
    config_directory = os.path.dirname(config_file_path)
    
    with open(config_file_path, 'r') as file:
        config = json.load(file)

    # Update Oracle params
    oracle_params = config['do-pairs'][0]['oracle']['parameters']
    oracle_params['epochs'] = epochs
    oracle_params['batch_size'] = batch_size

    # Update optimizer params
    oracle_params['optimizer']['class'] = optimizer
    oracle_params['optimizer']['parameters']['lr'] = lr

    # Update loss function params
    oracle_params['loss_fn']['parameters']['reduction'] = reduction
    oracle_params['loss_fn']['class'] = loss_f 

    # Update model params
    model_params = oracle_params['model']['parameters']
    model_params['num_conv_layers'] = num_conv_layers
    model_params['num_dense_layers'] = num_dense_layers
    model_params['conv_booster'] = conv_booster
    model_params['linear_decay'] = linear_decay

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
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)  # You can adjust the number of trials



    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print(f"Value: {trial.value}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")