# lift-risk
Lift detection and risk classification using machine learning on IMU sensor data

See Project-Description.md for full details of the project


### Team Members
Brennan Thomas<br>
Advisor: Dr. Rashmi Jha

### Abstract
Repeated lifting of heavy objects creates increased risk for incidences of lower back pain. Determining when a person is lifting in a risky way can help a worker learn to avoid dangerous lifts. The goal of this project is to develop a machine learning approach for detecting the risk level of an individual lift based on IMU data from various sensors placed on a worker’s body.

<br>

### [Project Description](Project-Description.md)

<br>

### User Stories and Design Diagrams

[User Stories](User_Stories.md)<br>
[Design Diagrams](Design_Diagrams)<br>
<br><br>

### [Test Plan](Test_Plan.md)

### [User Manual](docs/Setup.md)
### [FAQ](docs/FAQ.md)
### [Spring Final Presentation](presentations/final.pptx)
### [Final Expo Poster](presentations/poster.pdf)

### [Initial Self-Assessment](essays/initial_assessment.docx)
### [Final Self-Assessment](essays/final_assessment.docx)


### Summary of Hours

Total: 150 hours<br>
Hours involved research into machine learning methods for time-series human activity recognition, development of models architecture and framework, and testing and refinement of model performance.

<br><br>

### Appendix


Source code for lift classification system can be found in the [src](src) directory. Documentation on using the code can be found in the [docs](docs) directory, starting with the [requirements setup](docs/Setup.md).

Research:

[1]  1981. Work practices guide for manual lifting. Technical Report.  https://doi.org/10.26616/nioshpub81122<br>
[2]  1994. Applications manual for the revised NIOSH lifting equation. Technical Report.  https://doi.org/10.26616/nioshpub94110<br>
[3] Menekse Barim, Ming-Lun Lu, Shuo Feng, Grant Hughes, Marie Hayden, and Dwight Werren. 2019. Accuracy of An Algorithm Using Motion Data Of Five Wearable IMU Sensors For Estimating Lifting Duration And Lifting Risk Factors. Proceedings of the Human Factors and Ergonomics Society Annual Meeting63 (11 2019), 1105–1111.  https://doi.org/10.1177/1071181319631367<br>
[4] Francisco Ordóñez and Daniel Roggen. 2016.  Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition. Sensors 16, 1 (Jan. 2016), 115.  https://doi.org/10.3390/s16010115