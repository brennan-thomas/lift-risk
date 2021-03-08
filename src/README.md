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
The main code runner is classify.py. It trains a modified DeepConvLSTM model using NIOSH lifting data and reports the model's performance. Run python classify.py -h for usage information.