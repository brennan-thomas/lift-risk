# Road Map

## Milestones

A list of major milestones along with their deliverables and expected completion date. 

**Milestone:** Preprocessing
* Deliverable: A python implementation of a preprocessing pipeline, able to consume saved training/testing data for use in the machine learning models
* Expected Completion: Oct 26th 2020

**Milestone:** Lift Detection
* Deliverable: A python implementation of a machine learning model capable of detecting if a lift occurred in a segment of accelerometer data received from the preprocessing pipeline
* Expected Completion: Dec 1st 2020

**Milestone:** Lift Classification
* Deliverable: A python implementation of a machine learning model capable of classifying the risk level of a segment of accelerometer data received from the preprocessing pipeline
* Expected Completion: Jan 5th 2021

**Milestone:** Refinement
* Deliverable: Modifications to previous pipelines, models, and processes based on knowledge of how well they perform together and what can be improved.
* Expected Completion: Feb 28th 2021

<br></br>
## Timeline

A timeline of tasks' expected start and completion date

| Task/Milestone                                                                                        | Start Date | Completion Date |
|-------------------------------------------------------------------------------------------------------|------------|-----------------|
| **Preprocessing**                                                                                         | **10/12/20**   | **10/26/20**        |
| Research best practices for preprocessing data for machine learning                                   | 10/12/20   | 10/19/20        |
| Implement preprocessing in pipeline for training and testing and/or real use                          | 10/19/20   | 10/26/20        |
| **Lift Detection**                                                                                        | **10/26/20**   | **12/01/20**        |
| Research strategies for event detection using time-series data                                        | 10/26/20   | 11/03/20        |
| Develop machine learning program for lift detection                                                   | 11/03/20   | 11/10/20        |
| Determine metrics for evaluating detection model performance                                          | 11/10/20   | 11/17/20        |
| Document performance of detection solution                                                            | 11/17/20   | 11/24/20        |
| Implement detection model into pipeline after preprocessing                                           | 11/24/20   | 12/01/20        |
| **Lift Classification**                                                                                   | **12/01/20**   | **01/05/21**        |
| Research strategies for time-series classification                                                    | 12/01/20   | 12/08/20        |
| Develop machine learning program for lift classification                                              | 12/08/20   | 12/15/20        |
| Determine metrics for evaluating classification model performance                                     | 12/15/20   | 12/22/20        |
| Document performance of classification solution                                                       | 12/22/20   | 12/29/20        |
| Implement classification model into pipeline after lift detection                                     | 12/29/20   | 01/05/21        |
| **Refinement**                                                                                            | **01/05/21**   | **02/28/21**        |
| Determine which preprocessing strategies work best for the selected detection/classification networks | 01/05/21   | 01/12/21        |
| Refine detection solution for better performance                                                      | 01/12/21   | 02/28/21        |
| Refine classification solution for better performance                                                 | 01/12/21   | 02/28/21        |
| **Create interface for pipeline so that it can be easily used for lift classification**                   | **03/01/21**   | **03/08/21**        |
| **Document overall solution performance**                                                                 | **03/08/21**   | **03/15/21**        |

<br></br>
## Effort Matrix

Division of effort for tasks

| Task | Assignee | Expected Effort (1-5) | 
| ---  | ---      | ---             |
| Research best practices for preprocessing data for machine learning | Brennan | 2
| Determine which preprocessing strategies work best for the selected detection/classification networks | Brennan | 4
| Implement preprocessing in pipeline for training and testing and/or real use | Brennan | 3
| Research strategies for event detection using time-series data | Brennan | 3
| Develop machine learning program for lift detection | Brennan | 5
| Determine metrics for evaluating detection model performance | Brennan | 3
| Document performance of detection solution | Brennan | 2
| Refine detection solution for better performance | Brennan | 3
| Implement detection model into pipeline after preprocessing | Brennan | 3
| Research strategies for time-series classification | Brennan | 3
| Develop machine learning program for lift classification | Brennan | 5
| Determine metrics for evaluating classification model performance | Brennan | 3
| Document performance of classification solution | Brennan | 2
| Refine classification solution for better performance | Brennan | 3
| Implement classification model into pipeline after lift detection | Brennan | 3
| Create interface for pipeline so that it can be easily used for lift classification | Brennan | 2
| Document overall solution performance | Brennan | 3