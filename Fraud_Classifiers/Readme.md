# Code for Detecting Fraudulent Users in Temporal Transaction Network
Method is discussed in Method.pdf

## 1. Folder Structure

- Data: Folder to keep the data.
- Experiment: Frauddetector class along with the runner file to run the main function (Python).
- Motif_counting: Temporal Motif counting code (C++).
- Output: Output of the motif counting algorithm will be saved here (.txt, .pkl).
- config: An yaml file which is fed to the runner function defined in Experiment (yaml)


## 2. Create a conda environment and work on the conda environment

```
$ conda env create --file motifcounter.yml
$ conda activate motifcounter
```

## 3. Set the parameters in config.yaml file

- "root_path" (str) := Path to data,
- "filename" (str) := Name of the data file,
- "feature_file" (str) := Name of the file to save the features. Should be a .pkl file,

- "motif_count_params" := a dictionary containing parameters for the motif counting algorithm.
	- "needed" (bool) := Whether to run the motif counting algorithm or not.,
	- "num_node" (int) := Maximum number of nodes of the featured motif. For us it is 3,
	- "num_edge" (int) := Maximum number of nodes of the featured motif. For us it is 2,
	- "dc" (int) := Time difference between two consecutive events in a motif,
	- "dw" (int) := Time difference between first and last events in a motif,

- "feature_creation_params" := a dictionary containing parameters for the feature creation.
	- "needed" (bool) := Whether to create the features or not.,
	- "input_path" (str) := path to the input files. No need to change this.,
	- "measure": (int) := What measure to use. Possibilities are 0,1 and 2.

- "run_classifier" := a dictionary containing parameters for the classification algorithm.
	- "needed" (bool) := Whether classification is required or not,
	- "num_rep" (int) := Number of repeations of the classification algorithm.



## 4. Command to run the code

```
$ cd Experiment
$ python runner.py --conf ../config/config.yaml --preprocess
$ python runner.py --conf ../config/config.yaml
```

- "--preprocess" is necessary for the first run. For the subsequent run it is not needed. 

## 5. Hyperparameter Tuning

-  Set parameters in the config file following 3 and run:

```
$ python runner.py --conf ../config/config.yaml
```
