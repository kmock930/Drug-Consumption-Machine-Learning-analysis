# Problem Statement
## Aims
1. Convert the multi-class problems into binary classification tasks.
2. Predict whether a person is a consumer of chocolate and magic mushroom.
# Dataset: Drug Consumption Analysis Dataset
The dataset can be found at this link: https://archive.ics.uci.edu/dataset/373/drug+consumption+quantified.
## Description of the Dataset
- Contains a row identifier, 12 features describing the user data, and 18 classification problems related to using 18 different drugs.
- For each drug, it indicates whether a person has 'never used', 'used over a decade ago', 'used in the last decade', 'used in the last year', 'used in the last month', 'used in the last week', or 'used in the last day'.
# Implementation Details
- Split the Dataset and Perform Feature Engineering.
- Perform Supervised Learning using a pipeline of 6 classifiers.
- Identify potential issues in the dataset / the classifier itself.
- Provide results from Evaluation with some useful plots and metrics.
- Summarize the analysis in a report.
# Project Structure
- You should expect a report in `.pdf` format at the root level.
- Please expand the folder at the root level to view codes.
- This project branches out the analysis into 2 notebooks:
  1. Modelling - please check the file `CSI5155 Assignment 1 Modelling Part- Kelvin Mock 300453668.ipynb`
  2. Evaluation - please check the file `CSI5155 Assignment 1 Evaluation Part - Kelvin Mock 300453668.ipynb`
- Models are data dumped into several `.pkl` files from time-to-time in different phases to maintain the code's maintainability.
- The training sets and test sets are also data dumped into several `.pkl` files.
- `choc` directory shows data dumped files related to the Chocolate dataset (which is split from the original dataset).
- `mushrooms` directory shows data dumped files related to the Mushrooms dataset (which is also split from the original dataset).