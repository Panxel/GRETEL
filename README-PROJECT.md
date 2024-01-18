## Finding the outputs
The output logs for the explainers are in `/output/results/<explainer_name>_optimized`
The folder `optimization` and `explainer_opti` in `/output/results` were part of the hyperparameter optimization and are only interesting for reconstructing how we optimized the parameters.


## Finding the inputs that generated the outputs
The final and optimized json files we used for generating the outputs can be found in `/config/submission/<explainer_name>_optimized`
The template folders are again primarily irrelevant and were part of the hyperparameter optimization


## Finding the relevant scripts for hyperparameter optimization
The scripts that we used for optimizing the hyperparameters can be found in `/optimization`. The filenames should be self explanatory what each of the files was used for


## Finding the generator file
The generator file can be found at the same place as the other generators i.e. `/src/dataset/generators/twitter.py`