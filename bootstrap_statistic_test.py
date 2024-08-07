import os
import argparse
import glob
import numpy as np
import pandas as pd
import numpy as np
from fairmetric import *
import matplotlib.pyplot as plt
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
        "--reweight_path", 
        type=str, 
        default="", 
        help="Path to specific model reweight file"
    )
    parser.add_argument(
        "--partition", 
        type=int, 
        default=1, 
        help="Data partition method:'1:train/valid/test(6:2:2), 2:k-folds'."
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

def get_csvs(args, cancer_folder):
    model_names = [name.split('/')[-1] for name in glob.glob(args.model_path + f"{cancer_folder}_{args.partition}/*")]
    try:
        max_index = max([int(name.split('-')[0]) for name in model_names])
        if args.weight_path != '':
            max_index = args.weight_path
        baseline_csvs = [glob.glob(f'{args.model_path}{cancer_folder}_{args.partition}/{max_index}-*_{i}/inference_results_fold{i}.csv')[0] for i in range(4)]
        
        reweight_names = [name.split('/')[-1] for name in glob.glob(args.model_path + f"{cancer_folder}_{args.partition}_reweight/*")]
        max_reweight_index = max([int(name.split('-')[0]) for name in reweight_names])
        if args.reweight_path != '':
            max_reweight_index = args.reweight_path
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
        return dfs_baseline, dfs_corrected

    except:
        print(f'{cancer_folder} not found!')
        return None, None

def run_statistics(args, dfs_baseline, dfs_corrected, cancer_folder, n_bootstrap, aggregate_method, privileged_group):
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

    df_improv, df_p_better, df_p_worse = CV_bootstrap_improvement_test(
        dfs_baseline,
        dfs_corrected,
        privileged_group=privileged_group,
        n_bootstrap=n_bootstrap,
        aggregate_method=aggregate_method,
        ID_col='ID_col'
    )
    
    if eval(args.fair_attr).keys() == "Sex" or eval(args.fair_attr).keys() == "gender":
        fair_folder = "gender"
    else:
        fair_folder = "race"

    if not os.path.exists(f'significance_test_results/{fair_folder}/{cancer_folder}_{args.partition}'):
        os.mkdir(f'significance_test_results/{fair_folder}/{cancer_folder}_{args.partition}')
    df_p_worse_baseline.to_csv(f'significance_test_results/{fair_folder}/{cancer_folder}_{args.partition}/bias_baseline.csv')
    df_p_worse_corrected.to_csv(f'significance_test_results/{fair_folder}/{cancer_folder}_{args.partition}/bias_corrected.csv')
    df_p_better.to_csv(f'significance_test_results/{fair_folder}/{cancer_folder}_{args.partition}/improvement.csv')
    print(f'{cancer_folder}_{args.partition} saved!')

def main(args):
    aggregate_method = 'fisher'
    n_bootstrap = 1000
    privileged_group = None

    if args.task == 4:
        for models in os.listdir(args.model_path):
            if models.split("_")[1] == args.cancer[0] and models.split("_")[-1] == str(args.partition):
                geneType = models.split("_")[2]
                geneName = models.split("_")[3:-1]
                geneName = "_".join(geneName)
                cancer_folder = str(args.task) + "_" + "_".join(args.cancer) + "_" + geneType + "_" + geneName
                print(cancer_folder)
                dfs_baseline, dfs_corrected = get_csvs(args, cancer_folder)
                if dfs_baseline == None or dfs_corrected == None:
                    continue
                run_statistics(args, dfs_baseline, dfs_corrected, cancer_folder, n_bootstrap, aggregate_method, privileged_group)

    else:
        cancer_folder = str(args.task) + "_" + "_".join(args.cancer)
        print(cancer_folder)
        
        dfs_baseline, dfs_corrected = get_csvs(args, cancer_folder)
        
        # print(dfs_baseline[0].head())
        # print(dfs_corrected[0].head())

        run_statistics(args, dfs_baseline, dfs_corrected, cancer_folder, n_bootstrap, aggregate_method, privileged_group)


if __name__ == "__main__":
    args = parse_args()
    main(args)
