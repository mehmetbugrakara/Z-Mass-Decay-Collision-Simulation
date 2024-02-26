Particle Data Processor

This is a Python script for processing particle data from ROOT files, training and evaluating multiple classification models, and visualizing the data and model performance.

Overview
The Particle Data Processor script performs the following tasks:

Loading Data: Loads signal and background data from ROOT files containing particle data.
Processing Data: Processes particle data for signal and background events, including photons, muons, electrons, missing transverse energy (MET), and jets.
Feature Engineering: Calculates the sum of values in each event, extracts certain values for each group, and merges different data frames containing background and signal data.
Data Cleaning: Removes outliers from the data.
Model Training and Evaluation: Trains and evaluates multiple classification models including Logistic Regression, Decision Tree, Random Forest, XGBoost, and CatBoost.
Visualization: Visualizes histograms of features for background and signal classes, feature importance, correlation matrix, and ROC curve.

Requirements
Python 3.8.18
Libraries: uproot3, pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, catboost

Usage

Clone the repository:
git clone https://github.com/your_username/particle-data-processor.git

Install the required libraries:
pip install -r requirements.txt

Run the script:
python particle_data_processor.py
