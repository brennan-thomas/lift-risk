# Lift Detection
Detecting lift risk levels using deep residual LSTM neural networks

## Requirements
python 3.6<br>
numpy<br>
pandas<br>
scikit-learn<br>
tensorflow<br>
matplotlib<br>

The easiest way to install all requirements is to use [conda](https://docs.conda.io/en/latest/) and set up an environment using `environment.yml` as detailed [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

## Usage
The main code runners are `classify.py` and `detect.py`. They train a modified DeepConvLSTM models using NIOSH lifting data and report the models' performance. Run `python classify.py -h` or `python detect.py -h` for usage information. You can also just run them with default settings.

Output results from `classify.py` are saved to `trials/{name}` where name specified in the run arguments (default "test"). Results from `detect.py` are saved to `find_start/{name}` using the same conventions.