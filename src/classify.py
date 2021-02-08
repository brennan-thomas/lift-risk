import pandas as pd
import random
import os
import pickle
import numpy as np
from numpy import genfromtxt, dstack, moveaxis, array
import processing as pr
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import sys
#import keras_models as models
import heatmap as hm
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain, combinations
#from keras.callbacks import EarlyStopping
import datetime
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import models


def powerset(iterable):
    """Return a powerset of the given iterable, treating the iterable as a set"""

    s = list(iterable)
    tuples = chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))
    lists = map(lambda x: list(x), tuples)
    return list(lists)

def sensors_to_channels(sensor_list):
    """Append the channel postfix to the given sensors"""

    channels = []
    for sensor in sensor_list:
        channels.append(sensor + "_A_x")
        channels.append(sensor + "_A_y")
        channels.append(sensor + "_A_z")
        channels.append(sensor + "_G_x")
        channels.append(sensor + "_G_y")
        channels.append(sensor + "_G_z")
    return channels


def classToRiskMapArrReduced(probs):
    """Generate a real-valued prediction via a probability-weighted sum of classes

    Arguments:
    probs   -- the probabilities predicted by the model
    """

    result = []
    for el in probs:
        tot = 0
        for idx, cell in enumerate(el):
            tot += (idx + 1) * cell
        result.append(tot)
    return result

def leave_one_out_split(data_set, person):
    """Split the dataset so that one person's trials are held out for testing
    
    Arguments:
    data_set    -- the DataFrame with every person's trials
    person      -- the person to leave out of training, 1 to 10

    Returns:
    (train, test, val)  -- tuple of DataFrames for model consumption. trian/val contain all subjects
                           other than the one in test
    """

    for_training = data_set.loc[data_set['person'] != person]
    train, val = train_test_split(for_training, test_size=0.3, stratify=for_training.loc[:, 'class_label'])
    return (train, data_set.loc[data_set['person'] == person], val)

def k_fold(data_set, lo, hi):
    """Partition the dataset into folds, where each fold contains
       contains (train, test, val) where a given subject is left out

    Arguments:
    data_set    -- DataFrame containing trials for each subject
    lo          -- subject left out of training in the first fold, generally 1
    hi          -- subject left out of training in the last fold, generally the number of subjects in the data
    """

    folds = []
    # Remove person 4 due to misaligned sensors
    data_set = data_set.loc[data_set['person'] != 4]
    for i in range(lo, hi + 1):
        if i != 4:
            folds.append(leave_one_out_split(data_set, i))
    return folds
    
def simple_split(data_set, percentage_test=0.3, num_trials=1):
    """Split the dataset into train, test, validate sets

    Arguments:
    data_set    -- DataFrame with each trial
    percentage_test -- percentage of the dataset to hold back for testing and validation
    num_trials      -- number of folds to create, for symmetry with k-folds splitting
    """
    folds = []
    for i in range(num_trials):
        train, test = train_test_split(data_set, test_size=percentage_test, stratify=data_set.loc[:, 'class_label'])
        test, val = train_test_split(test, test_size=0.5, stratify=test.loc[:, 'class_label'])
        folds.append((train, test, val))
    return folds

def getclasses():
    """Get class labels in order for each trial"""

    csv = pd.read_csv('./metadata/lift_times_untrimmed.csv')
    # TODO remove bad data samples
    return pd.DataFrame(csv)['class_label']

def getpeople():
    """Get subject labels in order for each trial"""

    csv = pd.read_csv('./metadata/lift_times_untrimmed.csv')
    return pd.DataFrame(csv)['person']

def createFeatures(sheets, classes):
    """Create a DataFrame with a 'data' and 'class_label' columns, where
       each row is a single trial

    Arguments:
    sheets  -- array of trials, where each element is the IMU data for that trial
    classes -- array of class labels for each trial
    """

    agged = pd.DataFrame(columns=['data'])

    for i in range(len(sheets)):
        new_row = sheets[i]
        agged.loc[i] = [new_row]

    agged = agged.assign(class_label = classes)
    return agged

            
def keras_class_conversion(vec, n_classes):
    """Perform one-hot embeddings for the given vector of real values

    Arguments:
    vec         -- vector of class labels from 1 to n_classes
    n_classes   -- number of classes, sets the dimension of the embedding
    """

    return np.array(list(map(lambda x: [1 if i == x - 1 else 0 for i in range(n_classes)], vec)))

def get_data(simple=False, test_split = 0.25, count=1):
    """Load data and format it into folds of (train, test, val) for use in model

    Arguments:
    simple          -- whether or not to use a simple split rather than cross-validation (default False)
    test_split      -- test set percentage if using simple split (default 0.25)
    count           -- number of folds if using simple split (default 1)
    """


    filtered, filenames, max_size, _ = pr.preprocess('./source', './metadata/lift_times_untrimmed.csv')
    classes = pr.getclasses()
    filenames = np.array(filenames)

    filtered = np.stack(filtered)

    class_labels = getclasses()
    features = createFeatures(filtered, class_labels)
    features = features.assign(person=getpeople())
    
    # Remove Person 4 because of incorrect sensors
    #features = features[features['person'] != 4]
    
    # Split the data, either for cross validation or a simple train/test split
    if not simple:
        folds = k_fold(features, 1, 10)
    else:
        folds = simple_split(features, test_split, count)

    return folds

def train_model_fold(train_data, val_data, model_name, epochs, model_num=0, batch_size=32, kernel=5, test_name='test', lr=0.001, reg=0.01, dropout=0.5):
    """Train a model on one fold of data

    Arguments:
    train_data  -- DataFrame with columns 'data' and 'class_label'
    val_data    -- DataFrame with same format as train_data, to use for validation
    model_name  -- type of model to train. must be name of a model defined in models.py
    epochs      -- maximum number of epochs to train for. may terminate before this due to early stopping
    model_num   -- identifies which model this is, since we train a different model for each fold
    batch_size  -- batch size for model training (default 32)
    kernel      -- kernel size for Conv1D layers
    test_name   -- name of the current test run. used for labeling, files, etc
    lr      -- learning rate for Adam optimizer (default 0.0001)
    reg     -- regularization parameter (default 0.001)
    dropout -- dropout percentage (default 0.5)
    """
    
    X_train = train_data.loc[:, 'data']
    X_train = np.stack([x for x in X_train]) # extract from dataframe and stack into samples
    y_train = keras_class_conversion(train_data.loc[:, 'class_label'], 4) # one-hot embedding

    X_val = val_data.loc[:, 'data']
    X_val = np.stack([x for x in X_val])
    y_val = keras_class_conversion(val_data.loc[:, 'class_label'], 4)

    # Create the named model
    model_func = getattr(models, model_name)
    model = model_func((X_train.shape[1], X_train.shape[2]), kernel, lr=lr, reg=reg, dropout=dropout)

    # TensorBoard logging
    log_dir = 'logs/fit/{}-{}'.format(test_name, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
   
    # Fit the model to the fold of data, using early stopping to halt training if val loss stops decreasing
    print('Fitting model {}...'.format(model_num + 1))
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_data=(X_val, y_val), callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=100, restore_best_weights=True), callback])

    return model

# Test a trained model on one fold of data
def test_model_fold(test_data, model, batch_size=32):
    """Test a trained model on one fold of data

    Arguments:
    test_data   -- DataFrame with columns 'data' and 'class_label'
    model       -- the trained model to use for testing
    batch_size  -- the batch size (default 32)
    
    Returns a dict with values:
    pred_mapped         -- class predictions, 1-4
    pred_prob_mapped    -- real-valued predictions based on probability weighting
    actual              -- actual class labels
    mat                 -- confusion matrix with rows as actual and columns as predictions
    """

    X_test = test_data.loc[:, 'data']
    X_test = np.stack([x for x in X_test])
    y_test = keras_class_conversion(test_data.loc[:, 'class_label'],4) # one-hot embedding

    probs = model.predict(X_test, batch_size=batch_size)
    preds = probs.argmax(axis=-1)
    pred_mapped = preds + 1
    #pred_mapped = wholeClassToRiskMapArrReduced(preds)
    pred_prob_mapped = classToRiskMapArrReduced(probs)
    actual = test_data.loc[:, 'class_label']
    
    mat = confusion_matrix(actual, pred_mapped, labels=list(range(1,5)))

    return {'pred_mapped': pred_mapped,
            'pred_prob_mapped': pred_prob_mapped,
            'actual': actual,
            'mat': mat}

# Trains a model for every fold of data and compiles results
def train_all(folds, model_name, epochs, kernel=5, test_name='test', lr=1e-5, reg=0.01, dropout=0.25):
    """Train a model for every fold of data and compile results

    Arguments:
    folds   -- list of tuples, where each tuple is data of the form (train, test, val)
    model_name  -- type of model to train. must be name of a model defined in models.py
    epochs      -- maximum number of epochs to train for. may terminate before this due to early stopping
    kernel      -- kernel size for Conv1D layers
    test_name   -- name of the current test run. used for labeling, files, etc
    lr      -- learning rate for Adam optimizer (default 0.0001)
    reg     -- regularization parameter (default 0.001)
    dropout -- dropout percentage (default 0.5)
    """
    results = []
    models = []
    mats = []
    for idx, fold in enumerate(folds):
        train, test, val = fold

        model = train_model_fold(train, val, model_name, epochs, idx, kernel=kernel, test_name=test_name, lr=lr, reg=reg, dropout=dropout)
        out = test_model_fold(test, model)
        results.append(out)
        #models.append(model) #removing this bc memory seems to be an issue
        mats.append(out['mat'])

    total_mat = np.sum(x['mat'] for x in results)
    print('Total Confusion Matrix:')
    print(total_mat)

    all_probs = np.concatenate([x['pred_prob_mapped'] for x in results])
    all_pred_ints = np.concatenate([x['pred_mapped'] for x in results])
    all_actual = np.concatenate([x['actual'] for x in results])
    all_people = np.concatenate([np.full(shape=len(results[i]['actual']), 
                                         fill_value=i+1, 
                                         dtype=np.int) for i in range(len(results))])

    all_results = pd.DataFrame(
        index=range(len(all_probs)),
        data = {'Prediction': all_probs, 'Actual': all_actual,
        'Person': all_people}
    )

    bac = balanced_accuracy_score(all_actual, all_pred_ints, adjusted=True)
    print('Balanced Accuracy:', bac)

    return all_results, bac, total_mat, models, mats


# Creates a heatmap confusion matrix and swarmplot for prediction results
def display_results(all_results, bac, total_mat, mats, name):
    """Creates a heatmap confusion matrix and swarmplot for prediction results

    Arguments:
    all_results -- results dictionary from train_all
    bac         -- balanced accuracy score
    total_mat   -- confusion matrix with values from all folds
    mats        -- list of individual confusion matrices from each fold
    name        -- name of the trial for file naming
    """

    hm.heatmap(total_mat, name, './trials/{}/total.png'.format(name))
    with open('./trials/{}/matrix.txt'.format(name), 'w') as f:
        f.write('Balanced Accuracy: {}\n'.format(bac))
        f.write(str(total_mat))
    for i, mat in enumerate(mats):
        hm.heatmap(mat, name + ' {}'.format(i + 1), './trials/{}/{}.png'.format(name, i + 1))

    f_2, ax_2 = plt.subplots(1, 1, figsize=(6, 10), dpi=300)
    sns.swarmplot(x='Actual', y='Prediction', hue='Person',
    data = all_results, ax=ax_2)
    f_2.tight_layout()
    plt.savefig('./trials/{}/swarmplot.png'.format(name))

def run_test(args, lr=0.001, reg=0.01, dropout=0.5):
    """Train and test models using given parameters"""    
    
    folds = get_data(args.simple, args.test_split, args.count)

    all_results, bac, total_mat, models, mats = train_all(folds, args.model, args.epochs, kernel=args.kernel_size, test_name=args.name, lr=lr, reg=reg, dropout=dropout)
    display_results(all_results, bac, total_mat, mats, args.name)
    
    os.makedirs('models/' + args.name, exist_ok=True)
    for i, model in enumerate(models):
        model.save('models/{}/model{}.hdf5'.format(args.name, i))
    print('Results: ')
    print(all_results)
    print("BAC: ")
    print(bac)
    print('Total Conf Mat: ')
    print(total_mat)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='The type of model to use', default='residual_conv_oneskip_preact')
    parser.add_argument('-e', '--epochs', help='The maximum numer of epochs to train', default=300, type=int)
    parser.add_argument('-n', '--name', help='The name of this trial', default='test', type=str)
    parser.add_argument('-s', '--simple', help='Simple split instead of k-fold', action='store_true')
    parser.add_argument('-t', '--test-split', help='Percentage of data to test on if simple split', default=0.3)
    parser.add_argument('-c', '--count', help='The number of trials to run in a simple split', default=1, type=int)
    parser.add_argument('-ks', '--kernel-size', help='Kernel size for conv', default=5, type=int)

    args = parser.parse_args()
    run_test(args, lr=1e-3, reg=1e-4, dropout=0.5)



