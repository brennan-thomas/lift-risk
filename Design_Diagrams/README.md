# Design Diagrams

These design diagrams show the design of the lift detection/classification system at different levels of detail. D0 shows least detail, while D2 shows most detail.

## Legend

![Curved block is stored data. Rectangle is a process. Diamond is a conditional. Parallelogram is input/output. Oval is termination. Arrow is data flow.](Legend.png "Design Diagram Legend")

## D0

![D0](D0.png "D0")

Some actor (usually a worker) will be supplied with wearable IMU sensors to wear during activities. Some action will generate time-series data from the sensors, which will be sent to the machine learning process. This process will decide on a classification for the action, whether that is a risk level for a lift or nothing if it is not a lift.

## D1

![D1](D1.png "D1")

Similar to above, but the data is stored so it can be used in the model. Prior to the machine learning process, the time-series data must be formatted and preprocessed for model consumption. Depending on if the model is being trained or not, the data may need to be split into train/test sets first. The machine learning process will output probabilities for each class, and these probabilities will be used to decide on a final classification.

## D2

![D2](D2.png "D2")

In this diagram, the machine learning process is split into two networks. The first is a lift detection network, that determines if the action being performed is a lift. If not, no classification is necessary so the process terminates. Otherwise, the data is then used in the classification network to determine class probabilities and the final classification. The detection and classification networks may have different data preprocessing or formatting requirements.