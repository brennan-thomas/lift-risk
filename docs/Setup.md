# Setup

## Requirements

Python 3.6.12<br>
Tensorflow 2.1.0<br>
Pandas 1.1.5<br>
Scikit-learn 0.23.2<br>
Matplotlib 3.3.2<br>
Seaborn 0.11.1<br>
Numpy 1.19.2<br>

## Conda Installation
The easiest way to install all required dependencies is via the [conda](https://docs.conda.io/en/latest/) package manager. After installing conda, navigate to the [src](../src) directory and run:

`conda env create -f environment.yml`

This will create the conda environment LIFT and install all necessary packages. You can then run:

`conda activate LIFT`

to activate the environment. After activation, you can [run the detection and classification systems](Models.md).