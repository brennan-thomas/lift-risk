# Using this Repository

## Lift Detection

Lift detection is still a work in progress. When it is usable this section will be updated with documentation on running it.

## Lift Classification

Lift classification is implemented in the file [classify.py](../src/classify.py). The program will train lift classification models and report the performance and results of those models. Training options are controlled through flags passed when the program is run. The available flags are as follows:

`-h --  shows a help message listing all available flags`<br>
`-m --  the type of model to train. available models are defined in `[models.py](../src/models.py)<br>
`-e --  the number of epochs to train the models for. an early stopping module is implemented so the model will end training early if validation loss stops decreasing`<br>
`-n -- the name of this particular run. used to name directors for saving output results and models`<br>
`-s -- if this flag is used, training will be performed using a simple train/test split rather than cross-validation. this results in faster training but may not be as representative of real model performance`<br>
`-t -- if a simple split is used, this flag sets the percentage of data to reserve as test data`<br>
`-c -- the number of models to train if using a simple split. this emulates the cross-validation behavior of training multiple models`<br>
`-ks    -- the size of the convolution kernel to use in the model`

The simplest training setup would be something like this:<br>
`python classify.py -m residual_4class_dense -e 500 -n test -s`<br>
to train a single model for 500 epochs using the `residual_4class_dense` architecture.

After training, the model is tested and then results, in the form of heatmaps, swarmplots, and confusion matrices, are saved in the directory `trials/[name]/`, where [name] is defined by the `-n` flag. Trained models are saved in the directory `models/[name]/` with the name `model[idx].hdf5`, where `idx` is the model number based on which fold it was trained on.