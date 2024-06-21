import os
import sys
import ast
import argparse
import torch
import glob
from torch.utils.data import DataLoader
from tqdm import tqdm
from network import ClfNet, WeibullModel
import numpy as np
import pandas as pd
from util import replace_linear, FairnessMetrics, FairnessMetricsMultiClass, Find_Optimal_Cutoff, SurvivalMetrics
import numpy as np
import loralib as lora
from pathlib import Path
from dataset import generateDataSet, get_datasets
from fairmetric import *
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from bootstrap_significant_test.bootstrap_TCGA_improvement_test import CV_bootstrap_improvement_test
from bootstrap_significant_test.bootstrap_TCGA_bias_test import CV_bootstrap_bias_test

def parse_args(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cancer",
        nargs='+',
        default=None,
        required=True,
        help="Cancers are the targets for this task.",
    )
    parser.add_argument(
        "--fair_attr",
        default=None,
        required=True,
        help="Protected attribute we want to improve for this task.",
    )
    parser.add_argument(
        "--task",
        type=int,
        default="1",
        help="Downstream task: '1:cancer classification, 2:tumor detection, 3:survival prediction, 4:genetic classification",
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="./models/", 
        help="Folder path where to save model weights"
    )
    parser.add_argument(
        "--weight_path", 
        type=str, 
        default="", 
        help="Path to specific model weight file"
    )
    parser.add_argument(
        "--curr_fold", 
        type=int, 
        default=0, 
        help="For k-fold experiments, current fold."
    )
    parser.add_argument(
        "--partition", 
        type=int, 
        default=1, 
        help="Data partition method:'1:train/valid/test(6:2:2), 2:k-folds'."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=0, 
        help="Random seed for data partition."
    )
    parser.add_argument(
        "--reweight", 
        action='store_true', 
        help="For FAIR-Tuning."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cpu or cuda",
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args):

    cancer_folder = str(args.task) + "_" + "_".join(args.cancer)
    model_names = [name.split('/')[-1] for name in glob.glob(args.model_path + f"{cancer_folder}_{args.partition}/*")]
    max_index = max([int(name.split('-')[0]) for name in model_names])

    baseline_csvs = [glob.glob(f'{args.model_path}{cancer_folder}_{args.partition}/{max_index}-*_{i}/inference_results_fold{i}.csv')[0] for i in range(4)]

    reweight_names = [name.split('/')[-1] for name in glob.glob(args.model_path + f"{cancer_folder}_{args.partition}_reweight/*")]
    max_reweight_index = max([int(name.split('-')[0]) for name in reweight_names])
    reweight_csvs = [glob.glob(f'{args.model_path}{cancer_folder}_{args.partition}_reweight/{max_reweight_index}-*_{i}_reweight/inference_results_fold{i}.csv')[0] for i in range(4)]

    dfs_baseline = [pd.read_csv(csv) for csv in baseline_csvs]
    dfs_corrected = [pd.read_csv(csv) for csv in reweight_csvs]

    columns = {
        'labels': 'label',
        'probs': 'prob',
        'predictions': 'pred',
        'senAttrs': 'sens_attr',
    }
    dfs_baseline = [df.rename(columns=columns) for df in dfs_baseline]
    dfs_corrected = [df.rename(columns=columns) for df in dfs_corrected]

    dfs_baseline = [df.loc[~df['sens_attr'].isna()] for df in dfs_baseline]
    dfs_corrected = [df.loc[~df['sens_attr'].isna()] for df in dfs_corrected]

    print(dfs_baseline[0].head())
    print(dfs_corrected[0].head())

    aggregate_method = 'fisher'
    n_bootstrap = 1000
    privileged_group = None
    df_p_worse_baseline, df_p_better_baseline, fairResult_baseline, df_CI_baseline = (
        CV_bootstrap_bias_test(
            dfs_baseline,
            privileged_group=privileged_group,
            n_bootstrap=n_bootstrap,
            aggregate_method=aggregate_method,
        )
    )
    df_p_worse_corrected, df_p_better_corrected, fairResult_corrected, df_CI_corrected = (
        CV_bootstrap_bias_test(
            dfs_corrected,
            privileged_group=privileged_group,
            n_bootstrap=n_bootstrap,
            aggregate_method=aggregate_method,
        )
    )
    print(df_p_worse_baseline)

if __name__ == "__main__":
    args = parse_args()
    main(args)
