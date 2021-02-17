import datetime
import os

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.preprocessing import LabelBinarizer

import models

def train_model(model, train_data, val_data=None, test_name='test', epochs=500, batch_size=32):
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

    # TensorBoard logging
    log_dir = 'logs/fit/{}-{}'.format(test_name, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    callbacks = [TensorBoard(log_dir=log_dir, histogram_freq=1)]
    
    X_train, y_train = train_data
    if val_data is not None:
        X_val, y_val = val_data
        callbacks.append(EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True))

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_data=(X_val, y_val), callbacks=callbacks)

    return model


def test_model(model, test_data, batch_size=32):
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

    X_test, y_test = test_data

    probs = model.predict(X_test, batch_size=batch_size)

    return probs


# Trains a model for every fold of data and compiles results
def train_all(train_folds, model_name, epochs, kernel=5, test_name='test', lr=1e-5, reg=0.01, dropout=0.25, batch_size=32):
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
    trained_models = []
    for idx, fold in enumerate(train_folds):
        train, val = fold

        model_func = getattr(models, model_name)
        model = model_func((train[0].shape[1], train[0].shape[2]), kernel, lr=lr, reg=reg, dropout=dropout)

        print('Fitting model {}...'.format(idx))
        model = train_model(model, train, val, test_name=test_name, epochs=epochs, batch_size=batch_size)
        trained_models.append(model)

    return trained_models

def test_all(test_folds, models, test_name='test'):
    """Train a model for every fold of data and return trained models

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