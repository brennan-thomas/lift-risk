# Test Plan

## I. Overall Test Plan

Each significant module of the system (data processing, lift detection, and lift classification) are tested individually for both functionality and performance. Models are tested with data that is known to be classified correctly so that their functionality within the system can be tested. Performance tests measure the general accuracy of the models on large inputs of data. Targets values for these tests are preliminary and the tests may fall short of or exceed these values. End-to-end tests ensure that the full system works together regardless of the individual models' performances. Finally, tested with synthetic data are included to determine the individual models' behavior on unknown or abnormal data.

## II. Test Cases

PRE1.1: **Preprocessing Test 1** <br>
PRE1.2: This test will ensure that data is correctly preprocessed and formatted for lift detection <br>
PRE1.3: This test will load IMU data for each sample from file and slice them using a sliding window to create a dataset suitable for lift detection. <br>
PRE1.4: Inputs: Stored data files containing information for trials, length of sliding window to use<br>
PRE1.5: Expected Outputs: Time-series IMU data of the specified window size with 36 channels<br>
PRE1.6: Normal<br>
PRE1.7: Whitebox<br>
PRE1.8: Functional<br>
PRE1.9: Unit<br>

PRE2.1: **Preprocessing Test 2** <br>
PRE2.2: This test will ensure that data is correctly preprocessed and formatted for lift classification <br>
PRE2.3: This test will load IMU data for each sample from file and format them for consumption by the classification model<br>
PRE2.4: Inputs: Stored data files containing information for trials<br>
PRE2.5: Expected Outputs: Time-series IMU data with 36 channels<br>
PRE2.6: Normal<br>
PRE2.7: Whitebox<br>
PRE2.8: Functional<br>
PRE2.9: Unit<br>

TRA1.1: **Training Test 1**<br>
TRA1.2: This test will ensure that cross-validation process is followed for each model<br>
TRA1.3: This test will begin the training process on training data for each fold of the cross validation and generate a trained model for each fold.<br>
TRA1.4: Inputs: Model architecture to follow, list of folds of training data<br>
TRA1.5: Expected Outputs: A trained model for each fold of data<br>
TRA1.6: Normal<br>
TRA1.7: Whitebox<br>
TRA1.8: Functional<br>
TRA1.9: Unit<br>

TRA2.1: **Training Test 2**<br>
TRA2.2: This test will ensure that each model of the cross-validation training process is succesfully trained<br>
TRA2.3: This test will begin the training process on training data for each fold of the cross validation and report the training and validation accuracy for each<br>
TRA2.4: Inputs: Model architecture to follow, list of folds of training data<br>
TRA2.5: Expected Outputs: Training accuracy over 90%, validation accuracy over 80% for each model<br>
TRA2.6: Normal<br>
TRA2.7: Blackbox<br>
TRA2.8: Performance<br>
TRA2.9: Unit<br>

DET1.1: **Detection Test 1** <br>
DET1.2: This test will ensure that the lift detection model can detect a lift is occurring with a high degree of confidence<br>
DET1.3: This test will give the trained detection models a list of *positive* lift samples and report the recall achieved. <br>
DET1.4: Inputs: List of trained model for each fold of cross-validation, positive test data <br>
DET1.5: Expected Outputs: Test recall over 80% for the *positive* class<br>
DET1.6: Normal<br>
DET1.7: Blackbox<br>
DET1.8: Performance<br>
DET1.9: Unit<br>

DET2.1: **Detection Test 2** <br>
DET2.2: This test will ensure that the lift detection model can detect a lift is *not* occurring with a high degree of confidence<br>
DET2.3: This test will give the trained detection models a list of *negative* lift samples and report the recall achieved. <br>
DET2.4: Inputs: List of trained model for each fold of cross-validation, negative test data <br>
DET2.5: Expected Outputs: Test recall over 80% for the *negative* class<br>
DET2.6: Normal<br>
DET2.7: Blackbox<br>
DET2.8: Performance<br>
DET2.9: Unit<br>

DET3.1: **Detection Test 3**<br>
DET3.2: This test will ensure that if the lift detection model detects a lift, the lift classification process starts<br>
DET3.3: This test will give the detection model a sample that is known to be detected as positive, and make sure that the classification process is set to start<br>
DET3.4: Inputs: Known positive sample, trained lift detection model<br>
DET3.5: Expected Outputs: The lift classification process starts<br>
DET3.6: Normal<br>
DET3.7: Whitebox<br>
DET3.8: Functional<br>
DET3.9: Integration<br>

DET4.1: **Detection Test 4**<br>
DET4.2: This test will ensure that if the lift detection model does not detect a lift, the lift classification progress does not start<br>
DET4.3: This test will give the detection model a sample that is known to be detected as negative, and make sure that the classification process does not start<br>
DET4.4: Inputs: Known negative sample, trained lift detection model<br>
DET4.5: Expected Outputs: Lift classification process does not begin<br>
DET4.6: Normal<br>
DET4.7: Whitebox<br>
DET4.8: Functional<br>
DET4.9: Integration<br>

CLA1.1: **Classification Test 1** <br>
CLA1.2: This test will ensure that the lift classification model can classify a low-risk lift with a high degree of confidence<br>
CLA1.3: This test will give the trained classification models a list of *low-risk* lift samples and report the recall achieved. <br>
CLA1.4: Inputs: List of trained model for each fold of cross-validation, low-risk test data <br>
CLA1.5: Expected Outputs: Test recall over 80% for the *low-risk* class<br>
CLA1.6: Normal<br>
CLA1.7: Blackbox<br>
CLA1.8: Performance<br>
CLA1.9: Unit<br>

CLA2.1: **Classification Test 2** <br>
CLA2.2: This test will ensure that the lift classification model can classify a medium-risk lift with a high degree of confidence<br>
CLA2.3: This test will give the trained classification models a list of *medium-risk* lift samples and report the recall achieved. <br>
CLA2.4: Inputs: List of trained model for each fold of cross-validation, medium-risk test data <br>
CLA2.5: Expected Outputs: Test recall over 80% for the *medium-risk* class<br>
CLA2.6: Normal<br>
CLA2.7: Blackbox<br>
CLA2.8: Performance<br>
CLA2.9: Unit<br>

CLA3.1: **Classification Test 3** <br>
CLA3.2: This test will ensure that the lift classification model can classify a high-risk lift with a high degree of confidence<br>
CLA3.3: This test will give the trained classification models a list of *high-risk* lift samples and report the recall achieved. <br>
CLA3.4: Inputs: List of trained model for each fold of cross-validation, high-risk test data <br>
CLA3.5: Expected Outputs: Test recall over 80% for the *high-risk* class<br>
CLA3.6: Normal<br>
CLA3.7: Blackbox<br>
CLA3.8: Performance<br>
CLA3.9: Unit<br>

CLA4.1: **Classification Test 4** <br>
CLA4.2: This test will ensure that the lift classification model can classify a sample that is not a lift with a high degree of confidence<br>
CLA4.3: This test will give the trained classification models a list of non-lift samples and report the recall achieved. <br>
CLA4.4: Inputs: List of trained model for each fold of cross-validation, non-lift test data <br>
CLA4.5: Expected Outputs: Test recall over 80% for the *none* class<br>
CLA4.6: Normal<br>
CLA4.7: Blackbox<br>
CLA4.8: Performance<br>
CLA4.9: Unit<br>

END1.1: **End-to-end Test 1**<br>
END1.2: This test will ensure that the entire process returns negative if a lift is not present<br>
END1.3: This test will give the pipeline a sample known to be marked as no lift by the lift detection model and make sure the process returns with it marked as such<br>
END1.4: Inputs: Trained detection model, trained classification model, non-lift sample<br>
END1.5: Expected Outputs: Labeled as no lift<br>
END1.6: Normal<br>
END1.7: Whitebox<br>
END1.8: Functional<br>
ENd1.9: Integration<br>

END2.1: **End-to-end Test 2**<br>
END2.2: This test will ensure that the entire process returns low-risk if a low-risk lift is detected<br>
END2.3: This test will give the pipeline a sample known to be marked as a lift by the lift detection model and marked as low-risk by the classification model, and make sure the process returns with it marked as such<br>
END2.4: Inputs: Trained detection model, trained classification model, low-risk sample<br>
END2.5: Expected Outputs: Labeled as low-risk lift<br>
END2.6: Normal<br>
END2.7: Whitebox<br>
END2.8: Functional<br>
ENd2.9: Integration<br>

END3.1: **End-to-end Test 3**<br>
END3.2: This test will ensure that the entire process returns medium-risk if a medium-risk lift is detected<br>
END3.3: This test will give the pipeline a sample known to be marked as a lift by the lift detection model and marked as medium-risk by the classification model, and make sure the process returns with it marked as such<br>
END3.4: Inputs: Trained detection model, trained classification model, medium-risk sample<br>
END3.5: Expected Outputs: Labeled as medium-risk lift<br>
END3.6: Normal<br>
END3.7: Whitebox<br>
END3.8: Functional<br>
ENd3.9: Integration<br>

END4.1: **End-to-end Test 4**<br>
END4.2: This test will ensure that the entire process returns high-risk if a high-risk lift is detected<br>
END4.3: This test will give the pipeline a sample known to be marked as a lift by the lift detection model and marked as high-risk by the classification model, and make sure the process returns with it marked as such<br>
END4.4: Inputs: Trained detection model, trained classification model, high-risk sample<br>
END4.5: Expected Outputs: Labeled as high-risk lift<br>
END4.6: Normal<br>
END4.7: Whitebox<br>
END4.8: Functional<br>
ENd4.9: Integration<br>

END5.1: **End-to-end Test 5**<br>
END5.2: This test will ensure that the entire process returns low-risk if a low-risk lift is detected<br>
END5.3: This test will give the pipeline a sample known to be marked as a lift by the lift detection model and marked as low-risk by the classification model, and make sure the process returns with it marked as such<br>
END5.4: Inputs: Trained detection model, trained classification model, low-risk sample<br>
END5.5: Expected Outputs: Labeled as low-risk lift<br>
END5.6: Normal<br>
END5.7: Whitebox<br>
END5.8: Functional<br>
ENd5.9: Integration<br>

END6.1: **End-to-end Test 6**<br>
END6.2: This test will ensure that the entire process returns negative if a lift is not present but is not flagged by the detection model<br>
END6.3: This test will give the pipeline a sample known to be marked as a lift by the lift detection model but marked as None by the classification model and make sure the process returns with it marked as such<br>
END6.4: Inputs: Trained detection model, trained classification model, non-lift sample<br>
END6.5: Expected Outputs: Labeled as no lift<br>
END6.6: Normal<br>
END6.7: Whitebox<br>
END6.8: Functional<br>
ENd6.9: Integration<br>

END7.1: **End-to-end Test 7**<br>
END7.2: This test will ensure that the entire process correctly classifies lift models<br>
END7.3: This test will give the pipeline a list of samples from the test data and record the classification performance<br>
END7.4: Inputs: Trained detection model, trained classification model, test data<br>
END7.5: Expected Outputs: Test accuracy over 80%<br>
END7.6: Normal<br>
END7.7: Blackbox<br>
END7.8: Performance<br>
END7.9: Integration<br>

ABN1.1: **Abnormal Test 1**<br>
ABN1.2: This test will determine the detection model's performance on unknown/abnormal data sequences<br>
ABN1.3: This test generates synthetic IMU data to give to the detection model and records the result<br>
ABN1.4: Inputs: Trained detection model, list of synthetic abnormal data<br>
ABN1.5: Expected Outputs: General output of model on unknown/abnormal data<br>
ABN1.6: Abnormal<br>
ABN1.7: Blackbox<br>
ABN1.8: Performance<br>
ABN1.9: Unit<br>

ABN2.1: **Abnormal Test 2**<br>
ABN2.2: This test will determine the classification model's performance on unknown/abnormal data sequences<br>
ABN2.3: This test generates synthetic IMU data to give to the detection model and records the result<br>
ABN2.4: Inputs: Trained classification model, list of synthetic abnormal data<br>
ABN2.5: Expected Outputs: General output of model on unknown/abnormal data<br>
ABN2.6: Abnormal<br>
ABN2.7: Blackbox<br>
ABN2.8: Performance<br>
ABN2.9: Unit<br>


## III. Test Case Matrix

| Test | Normal/Abnormal | Whitebox/Blackbox | Functional/Performance | Unit/Integration |
|---|---|---|---|---|
| PRE1 | Normal | Whitebox | Functional | Unit |
| PRE2 | Normal | Whitebox | Functional | Unit |
| TRA1 | Normal | Whitebox | Functional | Unit |
| TRA2 | Normal | Blackbox | Performance | Unit |
| DET1 | Normal | Blackbox | Performance | Unit |
| DET2 | Normal | Blackbox | Performance | Unit |
| DET3 | Normal | Whitebox | Functional | Integration |
| DET4 | Normal | Whitebox | Functional | Integration |
| CLA1 | Normal | Blackbox| Performance | Unit |
| CLA2 | Normal | Blackbox| Performance | Unit |
| CLA3 | Normal | Blackbox| Performance | Unit |
| CLA4 | Normal | Blackbox| Performance | Unit |
| END1 | Normal | Whitebox | Functional | Integration |
| END2 | Normal | Whitebox | Functional | Integration |
| END3 | Normal | Whitebox | Functional | Integration |
| END4 | Normal | Whitebox | Functional | Integration |
| END5 | Normal | Whitebox | Functional | Integration |
| END6 | Normal | Whitebox | Functional | Integration |
| END7 | Normal | Blackbox | Performance | Integration |
| ABN1 | Abnormal| Blackbox | Performance | Unit |
| ABN2 | Abnormal| Blackbox | Performance | Unit |