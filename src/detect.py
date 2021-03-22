import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
import pandas as pd
from datetime import datetime
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, f1_score
import models
import random
from heatmap import binary
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import tensorflow as tf
import argparse
import pickle

import processing as pr

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='The type of model to use, defined in models.py', default='find_start_window')
parser.add_argument('-e', '--epochs', help='The maximum numer of epochs to train, default 300. The model may not train for this many epochs if the validation loss stops decreasing.', default=50, type=int)
parser.add_argument('-n', '--name', help='The name of this trial, used to name output files', default='test', type=str)
parser.add_argument('-s', '--simple', help='Simple split instead of k-fold. This will speed up training considerably (by 10x) but increases the variation between runs. Useful if you are just testing something out.', action='store_true')
parser.add_argument('-t', '--test-split', help='Percentage of data to test on if simple split, used for train/test data split', default=0.25)
parser.add_argument('-c', '--count', help='The number of trials to run in a simple split. More trials reduces variation but takes longer', default=1, type=int)
parser.add_argument('-b', '--batch', help='The batch size', default=32, type=int)
parser.add_argument('-w', '--window', help='The size of the sliding window to use to segment the trials into samples', default=50, type=int)
args = parser.parse_args()

def leave_one_out_split(data_set, person):
    """
    Assumes that each person in the data (a DataFrame) has the same amount of trials.
    Chooses a range to use for testing and returns the rest as training.
    Returns a tuple of two DataFrame: (train, test)
    """
    for_training = data_set.loc[data_set['person'] != person]
    train, val = train_test_split(for_training, test_size=0.3)
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
    for i in range(lo, hi + 1):
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
        train, test = train_test_split(data_set, test_size=percentage_test)
        test, val = train_test_split(test, test_size=0.5)
        folds.append((train, test, val))
    return folds

def getclasses():
    """Get class labels in order for each trial"""

    csv = pd.read_csv('./metadata/lift_times_complete.csv')
    
    df = pd.DataFrame(csv)[['filename', 'class_label']]
    i = list(np.where(df['filename'] == 'Subject_02_P2_Zone12_T1')[0]) + list(np.where(df['filename'] == 'Subject_02_P2_Sit_T1')[0])
    #pd.set_option("display.max_rows", None)
    df = df.drop(i).reset_index(drop=True)
    return df['class_label']
    #return pd.DataFrame(csv)['class_label']

def getpeople():
    """Get subject labels in order for each trial"""

    csv = pd.read_csv('./metadata/lift_times_complete.csv')
    
    df = pd.DataFrame(csv)[['filename', 'person']]
    i = list(np.where(df['filename'] == 'Subject_02_P2_Zone12_T1')[0]) + list(np.where(df['filename'] == 'Subject_02_P2_Sit_T1')[0])
    df = df.drop(i).reset_index(drop=True)
    return df['person']
    #return pd.DataFrame(csv)['person']

def getfilenames():
    csv = pd.read_csv('./metadata/lift_times_complete.csv')
    
    df = pd.DataFrame(csv)[['filename']]
    i = list(np.where(df['filename'] == 'Subject_02_P2_Zone12_T1')[0]) + list(np.where(df['filename'] == 'Subject_02_P2_Sit_T1')[0])
    df = df.drop(i).reset_index(drop=True)
    return df['filename']
    #return pd.DataFrame(csv)['person']

def createFeatures(sheets, target):
    """Create a DataFrame with a 'data' and 'target' columns, where
       each row is a single trial

    Arguments:
    sheets  -- array of trials, where each element is the IMU data for that trial
    target -- array of start time stamp for each trial
    """

    agged = pd.DataFrame(columns=['data'])

    for i in range(len(sheets)):
        new_row = sheets[i]
        agged.loc[i] = [new_row]

    agged = agged.assign(target = target)
    return agged

def get_target(lift_start_frame, trial_length):
    """Returns the label (1 or 0) for each frame in the trial based on the start of the lift"""

    probs = np.zeros(trial_length)
    
    probs[max(0, lift_start_frame-5):min(trial_length-1, lift_start_frame+5)] = 1

    return probs

def get_data(simple=False, test_split = 0.3, count=1):
    """Load data and format it into folds of (train, test, val) for use in model

    Arguments:
    simple          -- whether or not to use a simple split rather than cross-validation (default False)
    test_split      -- test set percentage if using simple split (default 0.25)
    count           -- number of folds if using simple split (default 1)
    """

    filtered, filenames, max_size, starts, ends = pr.preprocess('./source', './metadata/lift_times_complete.csv', pad=False)
    
    filenames = np.array(filenames)

    class_labels = getclasses()
    features = createFeatures(filtered, starts)
    features = features.assign(person=getpeople())
    features = features.assign(filename=getfilenames())

    # Remove Person 4 P2 because of incorrect sensors
    features = features[~features['filename'].str.contains('04_P2')]
    if not simple:
        folds = k_fold(features, 1, 10)
    else:
        folds = simple_split(features, test_split, count)

    return folds

# Counts instances of each class in the dataset
def count(counts, batch):
    features, labels = batch
    class_1 = labels == 1
    class_1 = tf.cast(class_1, tf.int32)
    class_0 = labels == 0
    class_0 = tf.cast(class_0, tf.int32)

    counts['class_0'] += tf.reduce_sum(class_0)
    counts['class_1'] += tf.reduce_sum(class_1)

    return counts


def make_dataset(X_train, start_times, window_size, balance=True, batch=True, repeat=False):
    #print(X_train.shape)
    #print(start_times.shape)
    #exit()
    def gen():
        i = 0
        for sample in X_train:
            start = start_times[i]
            for idx in range(len(sample) - window_size):
                window = sample[idx:(idx+window_size)]
                if start is not None and start >= idx and start < idx+(window_size):
                   target = 1
                else:
                    target = 0

                yield (window, target)
            i += 1

    dataset = tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.int32), output_shapes=([window_size, X_train[0].shape[1]], []))

    def class_func(features, label):
        return label

    dataset = dataset.shuffle(1000)
    if repeat:
        dataset = dataset.repeat()
    if balance: 
        counts = dataset.take(1000).reduce(initial_state={'class_0': 0, 'class_1': 0}, reduce_func=count)
        counts = np.array([counts['class_0'].numpy(), counts['class_1'].numpy()]).astype(np.float32)
        fractions = counts / counts.sum()
        print('Original: {}'.format(fractions))
        resampler = tf.data.experimental.rejection_resample(class_func, target_dist=[0.5, 0.5], initial_dist=fractions)
        dataset = dataset.apply(resampler).map(lambda x, sample: sample)
        
        counts = dataset.take(1000).reduce(initial_state={'class_0': 0, 'class_1': 0}, reduce_func=count)
        counts = np.array([counts['class_0'].numpy(), counts['class_1'].numpy()]).astype(np.float32)
        fractions = counts / counts.sum()
        print('Balanced: {}'.format(fractions))
    if batch:
        dataset = dataset.batch(64)
    return dataset

      
def train_model_fold(train_data, val_data, model_name, epochs, window_size, model_num=0, batch_size=32):
    """Train a model on one fold of data

    Arguments:
    train_data  -- DataFrame with columns 'data' and 'class_label'
    val_data    -- DataFrame with same format as train_data, to use for validation
    model_name  -- type of model to train. must be name of a model defined in models.py
    epochs      -- maximum number of epochs to train for. may terminate before this due to early stopping
    window_size -- size of the sliding window to segment data
    model_num   -- identifies which model this is, since we train a different model for each fold
    batch_size  -- batch size for model training (default 32)
    """

    X_train = train_data.loc[:, 'data']
    X_train = list(X_train)
    y_train = train_data.loc[:, 'target']
    y_train = list(y_train)

    X_val = val_data.loc[:, 'data']
    X_val = list(X_val)
    y_val = val_data.loc[:, 'target']
    y_val = list(y_val)

    model_func = getattr(models, model_name)
    model = model_func((window_size, X_train[0].shape[1]), lr=1e-3, reg=1e-4, dropout=0.5) 

    data = make_dataset(X_train, y_train, window_size=window_size, balance=True, repeat=True)
    val = make_dataset(X_val, y_val, window_size=window_size, balance=True, repeat=True)
    print('Fitting model {}...'.format(model_num + 1))
    model.fit(data, epochs=epochs, callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=25, restore_best_weights=True)], steps_per_epoch=500, validation_steps=50, validation_data=val)

    return model

def test_model_fold(test_data, model, window_size, batch_size=1):
    """Test a trained model on one fold of data

    Arguments:
    test_data   -- DataFrame with columns 'data' and 'class_label'
    model       -- the trained model to use for testing
    window_size -- the size of the sliding window to segment data
    batch_size  -- the batch size (default 32)
    
    Returns:
    confusion matrix
    accuracy
    ROC curve
    precision-recall curve
    f1 score
    segmented test values
    """
    
    X_test = list(test_data.loc[:, 'data'])
    y_test = list(test_data.loc[:, 'target'])
    

    data = make_dataset(X_test, y_test, window_size=window_size, batch=False, balance=False, repeat=False)
    
    X_test = []
    y_test = []
    for x, y in data.as_numpy_iterator():
        X_test.append(x)
        y_test.append(y)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    probs = np.squeeze(model.predict(X_test))
    print(probs)
    preds = np.zeros(len(probs))
    preds[probs > 0.5] = 1
    print(preds)    
    print(y_test)
    correct = preds[preds == y_test]
    
    roc = roc_curve(y_test, probs)
    prec_rec = precision_recall_curve(y_test, probs)
    f1 = f1_score(y_test, preds)

    acc = len(correct) / len(preds)
    print('Accuracy: {}'.format(acc))
    print('F1 Score: {}'.format(f1))
    
    conf = confusion_matrix(y_test, preds)
    return conf, acc, roc, prec_rec, f1, y_test

def plots(name, conf, acc, roc, pr, f1, y_test, idx=0):
    """Creates all metric plots"""

    #with open('./lift_start/{}/matrix.txt'.format(name), 'w') as f:
    #    f.write(str(conf))
    binary(conf, 'Lift Detect', './lift_start/{}/{}.png'.format(name, idx))
    data = {'roc': roc, 'prec_rec': pr, 'f1': f1}
    pickle.dump(data, open('./lift_start/{}/metrics.p'.format(name), 'wb'))

    # precision-recall curve
    prec, rec, pr_thres = pr
    no_skill = len(y_test[y_test==1]) / len(y_test)
    plt.plot([0, 1], [1, 1], linestyle='--')
    plt.plot(rec, prec)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig('./lift_start/{}/pr_curve.png'.format(name))
    plt.clf()

    # roc curve
    fp, tp, roc_thres = roc
    ns_fp = np.linspace(0, 1, len(fp))
    ns_tp = ns_fp
    plt.plot(ns_fp, ns_tp, linestyle='--')
    plt.plot(fp, tp)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('./lift_start/{}/roc_curve.png'.format(name))
    plt.clf()

def run():
    model_name = args.model
    epochs = args.epochs
    name = args.name
    window_size = args.window

    folds = get_data(args.simple, args.test_split, args.count)

    mats = []
    rocs = []
    prs = []

    def avg_3(thrup_list):
        l1, l2, l3 = zip(*thrup_list)
        
        return (np.mean(l1), np.mean(l2), np.mean(l3))

    for idx, fold in enumerate(folds):
        train, test, val = fold

        model = train_model_fold(train, val, model_name, epochs, window_size=window_size, model_num=idx, batch_size=args.batch)
        conf, acc, roc, pr, f1, yt = test_model_fold(test, model, window_size=window_size, batch_size=args.batch)
        os.makedirs('./lift_start/{}/'.format(name), exist_ok=True)    
        plots(name, conf, acc, roc, pr, f1, yt, idx)
        mats.append(conf)
        rocs.append(roc)
        prs.append(pr)

    mats = np.array(mats)
    print(mats.shape)
    total_mat = np.sum(mats, axis=0)
    print(total_mat.shape)
    with open('./lift_start/{}/matrix.txt'.format(name), 'w') as f:
        f.write(str(total_mat))
    binary(total_mat, 'Lift Detect', './lift_start/{}/total.png'.format(name))

    return

    avg_roc = avg_3(rocs)
    avg_pr = avg_3(prs)
    # precision-recall curve
    prec, rec, pr_thres = avg_pr
    #no_skill = len(y_test[y_test==1]) / len(y_test)
    plt.plot([0, 1], [1, 1], linestyle='--')
    plt.plot(rec, prec)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig('./lift_start/{}/pr_curve.png'.format(name))
    plt.clf()

    # roc curve
    fp, tp, roc_thres = avg_roc
    ns_fp = np.linspace(0, 1, len(fp))
    ns_tp = ns_fp
    plt.plot(ns_fp, ns_tp, linestyle='--')
    plt.plot(fp, tp)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('./lift_start/{}/roc_curve.png'.format(name))
    plt.clf()

if __name__ == '__main__':
    run()