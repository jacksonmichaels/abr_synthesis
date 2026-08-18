#!/usr/bin/env python
# coding: utf-8
"""
Runs multitask model with conv-conv-pool architecture:
- training on entire train set
- accuracy evaluation on held-out test set
This is the architecture used for the final MD-CNN model

Authors:
	Michael Chen (original version)
	Anna G. Green
	Chang Ho Yoon
"""

import sys
import glob
import os
import yaml
import sparse

import tensorflow as tf
import keras.backend as K
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
from tb_cnn_codebase import *

drugs = ['RIFAMPICIN', 'ISONIAZID', 'PYRAZINAMIDE',
             'ETHAMBUTOL', 'STREPTOMYCIN', 'LEVOFLOXACIN',
             'CAPREOMYCIN', 'AMIKACIN', 'MOXIFLOXACIN',
             'OFLOXACIN', 'KANAMYCIN', 'ETHIONAMIDE',
             'CIPROFLOXACIN']
num_drugs = len(drugs)


## Compute the performance for the training set
def compute_drug_auc_table(y, y_pred, drug_to_threshold):
    """
    Computes the AUC, sensitivity, specificity, for given threshold

    Parameters
    ----------
    y_train: np.array
        actual values for y
    y_train_pred: np.array
        predicted values for y
    drug_to_threshold: dict of str->float
        The prediction threshold for each drug
    Returns
    -------
    pd.DataFrame with columns: 'Algorithm', 'Drug', "num_sensitive", "num_resistant",'AUC', "threshold", "spec", "sens"
    """
    column_names = ['Algorithm', 'Drug', "num_sensitive", "num_resistant",'AUC', "threshold", "spec", "sens"]
    results = pd.DataFrame(columns=column_names)

    for idx, drug in enumerate(drugs):
        print(f"calculating test metrics for drug: {drug}")

        # Calculate the threshold from the TRAINING data, not the test data
        threshold = float(drug_to_threshold[drug])
        non_missing_val = np.where(y[:, idx] != -1)[0]
        
        # Check if non_missing_val is empty (no valid data for this drug) -> no phenotype
        if len(non_missing_val)==0:
            # If no valid data, insert NaN values for metrics
            print(f"No valid data for drug: {drug} as all the rows are missing")
            results.loc[idx] = ['MD-CNN', drug, 0, 0, np.nan, threshold, np.nan, np.nan]
            continue  # Skip the rest of the loop and move to the next drug


        auc_y = np.reshape(y[non_missing_val, idx], (len(non_missing_val), 1)).astype(int)
        auc_preds = np.reshape(y_pred[non_missing_val, idx], (len(non_missing_val), 1))

        num_sensitive = np.sum(auc_y==1)
        num_resistant = np.sum(auc_y==0)

        # If we don't have at least 1 R and 1 S isolate we can't assess model
        if num_sensitive==0 or num_resistant==0:
            results.loc[idx] = ['MD-CNN', drug, num_sensitive, num_resistant, np.nan, threshold, np.nan, np.nan]
            continue  

        # Compute the AUC
        auc = roc_auc_score(auc_y, auc_preds)

        # Binarize the predicted values
        binary_prediction = np.array(y_pred[non_missing_val] > threshold).astype(int)

        # Be careful - RS encoding to numeric, resistant==0
        # Specificity = #TN / #Condition Negative,  # Sensitivity = #TP / #Condition Positive, Here defining "positive" as resistant
        spec = np.sum(np.logical_and(binary_prediction == 1, y[non_missing_val] == 1)) / num_sensitive
        sens = np.sum(np.logical_and(binary_prediction == 0, y[non_missing_val] == 0)) / num_resistant

        results.loc[idx] = ['MD-CNN', drug, num_sensitive, num_resistant, auc, threshold, spec, sens]

    return results


# Threshold selection for each drug based on training data
def calculate_drug_thresholds(y_train, y_train_pred, thresholds_path):
    """
    Calculate the thresholds for each drug based on the training data
    Parameters
    ----------
    y_train: np.array
        actual values for y
    y_train_pred: np.array
        predicted values for y
    thresholds_path: str
        Path to save the thresholds

    Returns
    -------
    pd.DataFrame with thresholds for each drug
    Drug to threshold mapping dict
    """
    
    print("Calculating thresholds for each drug...")
    threshold_data = []

    for idx, drug in enumerate(drugs):
        print(f"Calculating threshold for {drug}...")
        train_metrics = get_threshold_val(y_train[:, idx], y_train_pred[:, idx])
        train_metrics["drug"] = drug
        threshold_data.append(train_metrics)

    threshold_df = pd.DataFrame(threshold_data)
    threshold_df.to_csv(thresholds_path, index=False)
    print(f"Thresholds saved to {thresholds_path}")

    drug_to_threshold = {x:y for x,y in zip(threshold_df.drug, threshold_df.threshold)}

    return threshold_df, drug_to_threshold

def load_model(saved_model_path):
    """
    Load the model from the specified path.
    Parameters
    ----------
    saved_model_path: str
        Path to the saved model directory
        
    Returns
    -------
    model: keras.models.Model
        Loaded model
    """

    print("Loading model...")
    if os.path.isdir(saved_model_path):
        return models.load_model(saved_model_path, custom_objects={
            'masked_weighted_accuracy': masked_weighted_accuracy,
            'masked_multi_weighted_bce': masked_multi_weighted_bce
        })
    else:
        raise FileNotFoundError(f"Model directory not found at {saved_model_path}")
    
def create_input_data(df_geno_pheno, pkl_file_sparse_train, pkl_file_sparse_test, train_indices, test_indices):
    """
    Create the input data for the model
    Parameters
    ----------
    df_geno_pheno: pd.DataFrame
        Dataframe containing the genotype-phenotype data
    pkl_file_sparse_train: str
        Path to save the training data
    pkl_file_sparse_test: str
        Path to save the testing data
    train_indices: np.array
        Indices for the training data
    test_indices: np.array
        Indices for the testing data

    Returns
    -------
    X_sparse_train: sparse.COO
        Sparse training data
    X_sparse_test: sparse.COO
        Sparse testing data
    """
    if os.path.isfile(pkl_file_sparse_train) and os.path.isfile(pkl_file_sparse_test):
        print("X input already exists, loading X...")
        X_sparse_train = sparse.load_npz(pkl_file_sparse_train)
        X_sparse_test = sparse.load_npz(pkl_file_sparse_test)
        print("done!\n")
    else:
        print("creating X from geno_pheno df...")
        X_all = create_X(df_geno_pheno)
        print("done!")

        X_sparse = sparse.COO(X_all)

        X_all = X_sparse.todense()
        assert (X_all.shape[0] == len(df_geno_pheno))
        
        print("\nsplitting the X data into training and testing sets...")
        X_sparse_train = X_sparse[train_indices, :]
        X_sparse_test = X_sparse[test_indices, :]
        del X_sparse
        print("done!\n")

        print(f"saving X_train to {pkl_file_sparse_train} as compressed sparse matrix...")
        sparse.save_npz(pkl_file_sparse_train, X_sparse_train, compressed=False)
        print("done!\n")

        print(f"saving X_test to {pkl_file_sparse_train} as compressed sparse matrix...")
        sparse.save_npz(pkl_file_sparse_test, X_sparse_test, compressed=False)
        print("done!\n")

    return X_sparse_train, X_sparse_test

def create_train_test_data(train_df, test_df, X_sparse_train, X_sparse_test):
    """
    Create the output data for the model
    Parameters
    ----------
    train_df: pd.DataFrame
        Dataframe containing the training data
    test_df: pd.DataFrame
        Dataframe containing the testing data
    X_sparse_train: sparse.COO
        Sparse training data
    X_sparse_test: sparse.COO
        Sparse testing data

    Returns
    -------
    X_train: sparse.COO
        Sparse training data
    y_train: np.array
        Training labels
    X_test: sparse.COO
        Sparse testing data
    y_test: np.array
        Testing labels
    """
    
    y_all_train, y_array = rs_encoding_to_numeric(train_df, drugs)
    y_all_test, y_array_test = rs_encoding_to_numeric(test_df, drugs)

    del train_df
    print("done!\n")

    # obtain phenotype data for CNN
    print("obtaining train and test phenotype data for the drugs for CNN...")
    y_all_train = y_all_train[drugs].values.astype(int)
    y_all_test = y_all_test[drugs].values.astype(int)
    print("done!\n")

    print("considering isolates with at least 1 resistance status across all drugs...")
    # obtain isolates with at least 1 resistance status to length of drugs
    train_indices_with_R_phenotype = np.where(y_all_train.sum(axis=1) != -num_drugs)
    test_indices_with_R_phenotype = np.where(y_all_test.sum(axis=1) != -num_drugs)

    X_train = X_sparse_train[train_indices_with_R_phenotype]
    X_test = X_sparse_test[test_indices_with_R_phenotype]

    y_train = y_all_train[train_indices_with_R_phenotype]
    y_test = y_all_test[test_indices_with_R_phenotype]

    return X_train, y_train, X_test, y_test


#---------------------------

def main(kwargs):
    output_path = kwargs["output_path"]
    pkl_file_sparse_train = kwargs['pkl_file_sparse_train']
    pkl_file_sparse_test = kwargs['pkl_file_sparse_test']
    parquet_file = kwargs["metadata_path"]
    h5_file = kwargs["h5_path"]
    test_size = kwargs['test_size']
    saved_model_path = kwargs['saved_model_path']
    thresholds_path = kwargs['threshold_file']
    
    # Load the geno pheno data
    if os.path.isfile(parquet_file) and os.path.isfile(h5_file):
        print("genotype-phenotype df files already exist, proceeding with modeling")
    else:
        print("creating genotype-phenotype df dataset")
        make_geno_pheno_dataset(**kwargs)
        print("done!\n")

    print("loading combined genotype-phenotype data")
    df_geno_pheno = load_combined_geno_pheno(**kwargs)


    # Perform a 80/20 train-test split
    df_geno_pheno = df_geno_pheno.reset_index(drop=True)
    all_indices = df_geno_pheno.index

    train_indices, test_indices = train_test_split(all_indices, test_size=test_size, random_state=42)
    train_df = df_geno_pheno.loc[train_indices]
    test_df = df_geno_pheno.loc[test_indices]
    print(f"Number of training samples: {len(train_indices)}")
    print(f"Number of testing samples: {len(test_indices)}\n")


    print("Getting X input data...")
    X_sparse_train, X_sparse_test = create_input_data(df_geno_pheno, pkl_file_sparse_train, pkl_file_sparse_test, train_indices, test_indices)
    print("done!\n")


    print("creating y from geno_pheno df...")
    X_train, y_train, X_test, y_test = create_train_test_data(train_df, test_df, X_sparse_train, X_sparse_test)
    print("done!\n")


    ### Load the model
    print("Loading model...")
    model = load_model(saved_model_path)
    print("done!\n")

    ## Get the thresholds for evaluation
    print("Predicting for training data, necessary to get thresholds for test AUC calculation...")
    y_train_pred = model.predict(X_train.todense())
    # Select the prediction threshold for each drug based on TRAINING SET DATA
    auc_thresholds, drug_to_threshold = calculate_drug_thresholds(y_train, y_train_pred, thresholds_path)

    # Compute AUC for test set data
    print('Predicting for test data...')
    y_test_pred = model.predict(X_test.todense())
    results = compute_drug_auc_table(y_test, y_test_pred, drug_to_threshold)
    results.to_csv(f"{output_path}/test_set_auc.csv")


if __name__ == "__main__":
    _, input_file = sys.argv

    # load kwargs from config file (input_file)
    kwargs = yaml.safe_load(open(input_file, "r"))
    print(kwargs)

    main(kwargs)
 
