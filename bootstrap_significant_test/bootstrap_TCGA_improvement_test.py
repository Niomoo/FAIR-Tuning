
'''
Script to calculate the improvement of the corrected model over the baseline model
'''


import pandas as pd
import json
import os
from os.path import join, basename, dirname
from scipy.stats import combine_pvalues
import glob
import numpy as np
from argparse import ArgumentParser
from typing import List, Tuple, Literal
from utils import FairnessMetrics, get_metric_names
from tqdm import tqdm

SUBJ_ID_COL = 'bcr_patient_barcode'
EXCLUDE_RACES = ['AMERICAN INDIAN OR ALASKA NATIVE', '[not available]']
MINORITY_RACES = ['BLACK OR AFRICAN AMERICAN', 'ASIAN']



def get_TCGA_demo_info(
        demo_col,
        demo_folder='TCGA_all_clinical'):
    '''
    get the demographic information from the TCGA clinical files
    '''
    demo_csvs = glob.glob(
        join(demo_folder, 'nationwidechildrens.org_clinical_patient_*.txt'))
    dfs_demo = []
    for f in demo_csvs:
        df = pd.read_csv(f, delimiter='\t')
        df = df.iloc[2:]
        age_col = 'days_to_birth' if 'days_to_birth' in df.columns else 'birth_days_to'
        df['age'] = df[age_col]
        dfs_demo.append(df)

    df_demo = pd.concat(dfs_demo)
    df_demo = df_demo[[SUBJ_ID_COL, demo_col]]
    if demo_col == 'race':
        # exclude racial groups with insufficient sample size
        df_demo = df_demo[~df_demo[demo_col].isin(EXCLUDE_RACES)]
    df_demo['sens_attr'] = df_demo.pop(demo_col)
    return df_demo


def compile_dataframe(labels, preds, probs, df_demo,demo_col):
    '''
    compile the dataframe
    input:
        labels: pd.Series, the ground truth labels
        preds: pd.Series, the predictions
        df_demo: pd.DataFrame, the demographic information
        demo_col: str, the demographic attribute
    output:
        df: pd.DataFrame, the compiled dataframe
    '''
    df = pd.concat([labels, preds, probs], axis=1).reset_index()
    df['slide'] = df.pop('index')
    df['pred'] = [i[1] for i in df['pred']]  # get the positive class
    # get TCGA ID
    df[SUBJ_ID_COL] = ['-'.join(i.split('-')[:3]) for i in df['slide']]
    # .reset_index(drop=True)
    df = df.merge(df_demo, on=SUBJ_ID_COL, how='left')
    ## process age
    
    NA_STRS = ['[Not Available]', '[Not Applicable]',
                '[Not Evaluated]', '[Completed]']
    df.replace(NA_STRS, np.nan, inplace=True)
    df = df.dropna(subset=['sens_attr'])
    if demo_col == 'age':
        df['sens_attr'] = df['sens_attr'].astype(float)/-365.25
    
    return df

def get_results_from_folder(folder, redo_pred=True,redo_th=0.5):
    '''
    get the results from the folder (for new data)
    input:
        folder: str, the folder containing the results
    output:
        df: pd.DataFrame, the results. It contains the following columns:
            - 'label': the ground truth label
            - 'pred': the prediction
            - 'sens_attr': the demographic attribute
        privileged_group: str, the privileged group
    '''
    attr = basename(folder).split(' ')[0]
    if attr == 'female':
        demo_col = 'gender'
    elif attr == 'white':
        demo_col = 'race'
    else:
        demo_col = 'age'
    # find data

    # label
    label_jsons = [
        join(folder, f'groundTruthResults_test{fold}_0.json') for fold in range(4)]
    # prediction
    # NOTE: this json file only contains binary prediction, not the probability
    # if you want to use the probability, you need to change the json file to load the probability
    pred_jsons = [
        join(folder, f'predResults_test{fold}_0.json') for fold in range(4)]
    
    prob_npys = [join(folder, f'AUCprob_test{fold}_0.npy') for fold in range(4)]
    assert all([os.path.exists(f) for f in label_jsons+pred_jsons+prob_npys])

    ######### process the buggy data
    # read the probability from the npy files
    prob_M = [np.load(x) for x in prob_npys]
    prob_M_all = np.concatenate(prob_M, axis=0)
    # the labels and predictions are accendentally saved in the last fold

    pred_d = [json.load(open(x)) for x in pred_jsons]
    label_d = [json.load(open(x)) for x in label_jsons]
    prob_d = {key:value for key,value in zip(label_d[-1].keys(),prob_M_all)} # borrow the keys from the label
    ## get the keys for each fold
    
    df = pd.DataFrame(pred_d[-1]).T
    df = df.reset_index()
    df['subj'] = df.pop('index')
    df['fold'] = -1
    covered_subjs = set()
    for fold in range(4):
        fold_subjs = set(label_d[fold].keys())
        df.loc[~df['subj'].isin(covered_subjs), 'fold'] = fold
        covered_subjs = covered_subjs.union(fold_subjs)
    fold_idxs = [df.loc[df['fold']==fold]['subj'] for fold in range(4)]



    # load data
    labels = [{key:label_d[-1][key] for key in idx} for idx in fold_idxs]
    probs = [{key:prob_d[key] for key in idx} for idx in fold_idxs]
    if redo_pred:
        preds = [{key:[0,1] if prob_d[key] > redo_th else [1,0] for key in idx} for idx in fold_idxs]
    else:
        preds = [{key:pred_d[-1][key] for key in idx} for idx in fold_idxs]

    # labels = [json.load(open(f)) for f in label_jsons]
    # preds = [json.load(open(f)) for f in pred_jsons]
    labels = [pd.Series(l, name='label') for l in labels]
    preds = [pd.Series(p, name='pred') for p in preds]
    probs = [pd.Series(p, name='prob') for p in probs]
    #####################
    
    # TCGA demographics
    df_demo = get_TCGA_demo_info(demo_col)
    # assemble the dataframe
    dfs = []
    for label, pred,prob in zip(labels, preds,probs):
        df = compile_dataframe(label, pred, prob,df_demo,demo_col)
        dfs.append(df)
    
    # # predefined privileged group
    
        
    return dfs, demo_col


def get_paired_bootstrap_dataframes(df_baseline, df_corrected, ID_col='slide'):
    '''
    get the paired bootstrap dataframe for pair-sampled test
    NOTE: 
        This is different from utils.get_bootstrap_stats, which is for independent-sampled test.
        This bootstrap only within the same ID.
    input:
        df_baseline: pd.DataFrame, the baseline results
        df_corrected: pd.DataFrame, the corrected results
    output:
        df_baseline_bootstrapped: pd.DataFrame, the bootstrapped baseline results
        df_corrected_bootstrapped: pd.DataFrame, the bootstrapped corrected results
    '''
    # get the paired bootstrap dataframe
    df = pd.concat([df_baseline, df_corrected], axis=0)
    df_baseline_bootstrapped = df.groupby(
        ID_col).sample(n=1).reset_index(drop=True)
    df_corrected_bootstrapped = df.groupby(
        ID_col).sample(n=1).reset_index(drop=True)
    return df_baseline_bootstrapped, df_corrected_bootstrapped


def fairness_improvement(
        df_baseline,
        df_corrected,
        privileged_group,
        add_perf_difference=True
):
    '''
    calculate the improvement of the corrected model over the baseline model
    input:
        df_baseline: pd.DataFrame, the baseline results
        df_corrected: pd.DataFrame, the corrected results
        privileged_group: str, the privileged group
    output:
        improvement: float, the improvement
    '''
    metrics_list = []
    METRIC_NAMES_DICT = get_metric_names(add_perf_difference=add_perf_difference)

    for df in [df_baseline, df_corrected]:
        metrics = FairnessMetrics(
            df[f'pred'].to_numpy(),
            df[f'prob'].to_numpy(),
            df['label'].to_numpy(),
            df['sens_attr'].astype(str).to_numpy(), previleged_group=privileged_group,add_perf_difference=add_perf_difference)

        metrics_list.append(pd.DataFrame(
            metrics).set_index(['sensitiveAttr','group_type'], drop=True))
    improvement_larger_better = metrics_list[1] - metrics_list[0]
    improvement_smaller_better = metrics_list[0] - metrics_list[1]
    improvement = improvement_larger_better[METRIC_NAMES_DICT['higher_better_metrics']].merge(
        improvement_smaller_better[METRIC_NAMES_DICT['lower_better_metrics']], left_index=True, right_index=True)
    # reorganize the columns
    improvement = improvement[METRIC_NAMES_DICT['perf_metrics']+METRIC_NAMES_DICT['fairness_metrics']]
    return improvement


def bootstrap_improvement_test(df_baseline, df_corrected, privileged_group, n_bootstrap=1000,add_perf_difference=True,ID_col='slide'):
    # raise NotImplementedError
    dfs_improvement = []
    # for i in tqdm(range(n_bootstrap)):
    for i in tqdm(range(n_bootstrap),miniters=n_bootstrap//10):
        df_baseline_bootstrapped, df_corrected_bootstrapped = get_paired_bootstrap_dataframes(
            df_baseline, df_corrected,ID_col=ID_col)
        improvement = fairness_improvement(
            df_baseline_bootstrapped, df_corrected_bootstrapped, privileged_group=privileged_group,add_perf_difference=add_perf_difference)
        dfs_improvement.append(improvement)
    df_improvement_bootstrap = pd.concat(dfs_improvement)
    # calculate actual improvement
    improvement = fairness_improvement(
        df_baseline, df_corrected, privileged_group=privileged_group,add_perf_difference=add_perf_difference)
    # calculate p-value
    df_p_better = improvement.copy()
    df_p_worse = improvement.copy()
    for i, row in improvement.iterrows():
        for col in improvement.columns:
            bootstrap_values = df_improvement_bootstrap[col].loc[i].dropna()
            p_better = (bootstrap_values >= row[col]).sum() / len(bootstrap_values)
            p_worse = (bootstrap_values <= row[col]).sum() / len(bootstrap_values)
            df_p_better.loc[i, col] = p_better
            df_p_worse.loc[i, col] = p_worse
    return improvement, df_p_better, df_p_worse

def fairness_improvement_groupLevel(df,add_perf_difference=True):
    METRIC_NAMES_DICT = get_metric_names(add_perf_difference=add_perf_difference)

    df_corrected = df.loc[df['model']=='corrected'].set_index(['fold'])
    df_baseline = df.loc[df['model']=='baseline'].set_index(['fold'])
    higher_better_metrics = METRIC_NAMES_DICT['higher_better_metrics']
    lower_better_metrics = METRIC_NAMES_DICT['lower_better_metrics']
    higher_better_metrics = [x if x not in METRIC_NAMES_DICT['perf_metrics'] else f'{x}_minority' for x in higher_better_metrics] # add minority to the performance metrics
    lower_better_metrics = [x if x not in METRIC_NAMES_DICT['perf_metrics'] else f'{x}_minority' for x in lower_better_metrics] # add minority to the performance metrics
    
    higher_better_metrics = list(set(df_baseline.columns).intersection(set(higher_better_metrics)))
    lower_better_metrics = list(set(df_baseline.columns).intersection(set(lower_better_metrics)))

    df_improv = df_baseline.copy()
    df_improv.drop(columns=['model'],inplace=True)
    df_improv[higher_better_metrics] = df_corrected[higher_better_metrics] - df_baseline[higher_better_metrics]
    df_improv[lower_better_metrics] = df_baseline[lower_better_metrics] - df_corrected[lower_better_metrics]
    df_mean_improv = df_improv.mean()
    # for smaller better metrics, the improvement is the difference between the baseline and corrected

    return df_improv, df_mean_improv
    

def bootstrap_improvement_test_groupLevel(dfs_baseline, dfs_corrected, privileged_group, n_bootstrap=1000,add_perf_difference=True):
    ## get the fairness for each group
    METRIC_NAMES_DICT = get_metric_names(add_perf_difference=add_perf_difference)

    baseline_metrics_list = []
    corrected_metrics_list = []
    for i, df in enumerate(dfs_baseline):
        fairResult = FairnessMetrics(
            df[f'pred'].to_numpy(),
            df[f'prob'].to_numpy(),
            df['label'].to_numpy(),
            df['sens_attr'].astype(str).to_numpy(), previleged_group=privileged_group,add_perf_difference=add_perf_difference)        
        fairResult = pd.DataFrame(fairResult)
        fairResult.insert(2, 'fold', i)

        baseline_metrics_list.append(fairResult)
    for i, df in enumerate(dfs_corrected):

        fairResult = FairnessMetrics(
            df[f'pred'].to_numpy(),
            df[f'prob'].to_numpy(),
            df['label'].to_numpy(),
            df['sens_attr'].astype(str).to_numpy(), previleged_group=privileged_group,add_perf_difference=add_perf_difference)        
        fairResult = pd.DataFrame(fairResult)
        fairResult.insert(2, 'fold', i)

        corrected_metrics_list.append(fairResult)
    df_baseline = pd.concat(baseline_metrics_list)
    df_corrected = pd.concat(corrected_metrics_list)
    df_baseline['model'] = 'baseline'
    df_corrected['model'] = 'corrected'
    ### only keep the minority group
    df_baseline = df_baseline.loc[df_baseline['group_type']=='minority'].reset_index(drop=True)
    df_corrected = df_corrected.loc[df_corrected['group_type']=='minority'].reset_index(drop=True)
    dropped_col = df_baseline[['sensitiveAttr','group_type']].iloc[0]
    df_baseline.drop(columns=['sensitiveAttr','group_type','N_0','N_1'],inplace=True)
    df_corrected.drop(columns=['sensitiveAttr','group_type','N_0','N_1'],inplace=True)
    
    for col in METRIC_NAMES_DICT['perf_metrics']:
        df_baseline[f'{col}_minority'] = df_baseline.pop(col)
        df_corrected[f'{col}_minority'] = df_corrected.pop(col)
        
    ###
    df = pd.concat([df_baseline, df_corrected]).reset_index(drop=True)
    ## calculate the improvement
    df_improv, df_mean_improv = fairness_improvement_groupLevel(df)
    df_mean_improv = pd.DataFrame(df_mean_improv).T
    ## bootstrap
    dfs_mean_improv_bootstrap = []
    for i in tqdm(range(n_bootstrap),miniters=n_bootstrap//10):
        # df_bootstrap = df.sample(frac=1,replace=True)
        df_bootstrap = df.groupby('fold').sample(frac=1,replace=True).reset_index()
        df_bootstrap[['fold','model']] = df[['fold','model']]
        df_improv_bootstrap, df_mean_improv_bootstrap = fairness_improvement_groupLevel(df_bootstrap)
        dfs_mean_improv_bootstrap.append(df_mean_improv_bootstrap)
    df_mean_improv_bootstrap = pd.concat(dfs_mean_improv_bootstrap,axis=1).T
    
    # calculate p-value
    df_p_better = df_mean_improv.copy()
    df_p_worse = df_mean_improv.copy()
    for col in df_mean_improv.columns:
        bootstrap_values = df_mean_improv_bootstrap[col].dropna()
        p_better = (bootstrap_values >= df_mean_improv[col].iloc[0]).sum() / len(bootstrap_values)
        p_worse = (bootstrap_values <= df_mean_improv[col].iloc[0]).sum() / len(bootstrap_values)
        df_p_better[col] = p_better
        df_p_worse[col] = p_worse
    for df in [df_mean_improv,df_p_better,df_p_worse]:
        df.insert(0,'sensitiveAttr',dropped_col['sensitiveAttr'])
        df.insert(1,'group_type',dropped_col['group_type'])
    return df_mean_improv, df_p_better, df_p_worse


def CV_bootstrap_improvement_test(
    dfs_baseline, dfs_corrected, privileged_group=None, n_bootstrap=1000,aggregate_method='fisher',add_perf_difference=True,
    ID_col='slide'):
    '''
    Estimate the improvement of the corrected model over the baseline model for all folds
    Input:
        dfs_baseline: list of pd.DataFrame, the baseline results
            Each pd.DataFrame contains the following columns:
                * sens_attr: sensitive attributes
                * prob: model probabilities
                * label: ground truth labels
                * pred:  model predictions
        dfs_corrected: list of pd.DataFrame, the corrected results
            Each pd.DataFrame contains the following columns:
                * sens_attr: sensitive attributes
                * prob: model probabilities
                * label: ground truth labels
                * pred:  model predictions
        n_bootstrap: int, number of bootstrap iterations
        aggregate_method: str, method to aggregate p-values. Options are:
            - 'concatenate': concatenate the input data
            - 'fisher', 'pearson', 'tippett', 'stouffer', 'mudholkar_george': methods to combine p-values, see scipy.stats.combine_pvalues for details
        privileged_group: str, the privileged group
        add_perf_difference: bool, whether to add performance difference as a fairness metric
        ID_col: str, the column name for the ID
        
    Output:
        df_p_better: pd.DataFrame, the p-values for significant improvement
        df_p_worse: pd.DataFrame, the p-values for significant worsening
        
    '''
    
    if aggregate_method == 'concatenate':
        # if the method is concatenate, we concatenate the data and return a single p-value
        df_baseline = pd.concat(dfs_baseline)
        df_corrected = pd.concat(dfs_corrected)
        df_improv, df_p_better, df_p_worse = \
            bootstrap_improvement_test(
            df_baseline, df_corrected, n_bootstrap=n_bootstrap, privileged_group=privileged_group,add_perf_difference=add_perf_difference,ID_col=ID_col)
        return df_improv, df_p_better, df_p_worse
    elif aggregate_method in ['fisher', 'pearson', 'tippett', 'stouffer', 'mudholkar_george']:
        # if the method is fisher or stouffer, we calculate the p-value for each fold
        dfs_improv = []
        dfs_p_better = []
        dfs_p_worse = []
        for i, (df_baseline, df_corrected) in enumerate(zip(dfs_baseline, dfs_corrected)):
            df_improv, df_p_better, df_p_worse = \
                bootstrap_improvement_test(
                df_baseline, df_corrected, n_bootstrap=n_bootstrap, privileged_group=privileged_group,add_perf_difference=add_perf_difference,ID_col=ID_col)
            df_improv.insert(0, 'fold', i)
            df_p_better.insert(0, 'fold', i)
            df_p_worse.insert(0, 'fold', i)
            dfs_improv.append(df_improv)
            dfs_p_better.append(df_p_better)
            dfs_p_worse.append(df_p_worse)
        ## concatenate the p-values

        df_p_better = pd.concat(dfs_p_better)
        df_p_worse = pd.concat(dfs_p_worse)
        df_improv = pd.concat(dfs_improv)
        ## aggregate the p-values
        df_p_combined = dfs_p_better[0].copy()
        for i, row in df_p_combined.iterrows():
            for col in df_p_combined.columns:
                pvals = df_p_better[col].loc[i]
                meta_res = combine_pvalues(pvals, method=aggregate_method, weights=None)
                df_p_combined[col].loc[i] = meta_res.pvalue
        df_p_combined['fold'] = f'{aggregate_method}_combined'
        df_p_better = pd.concat([df_p_better, df_p_combined])
        
        df_p_combined = dfs_p_worse[0].copy()
        for i, row in df_p_combined.iterrows():
            for col in df_p_combined.columns:
                pvals = df_p_worse[col].loc[i]
                meta_res = combine_pvalues(pvals, method=aggregate_method, weights=None)
                df_p_combined[col].loc[i] = meta_res.pvalue
        df_p_combined['fold'] = f'{aggregate_method}_combined'
        df_p_worse = pd.concat([df_p_worse, df_p_combined])
        return df_improv, df_p_better, df_p_worse
    elif aggregate_method == 'groupwise':
        # if the method is groupwise, we estimate the fairness metrics first, and then perform bootstraping on population level
        df_improv, df_p_better, df_p_worse = bootstrap_improvement_test_groupLevel(
            dfs_baseline, dfs_corrected, n_bootstrap=n_bootstrap, privileged_group=privileged_group,add_perf_difference=add_perf_difference)
        return df_improv, df_p_better, df_p_worse


def main(
    baseline_folder,corrected_folder,
    fix_privileged_group=True,
    n_bootstrap=1000,
    add_perf_difference=True,
    aggregate_method: Literal['concatenate','fisher', 'pearson', 'tippett', 'stouffer', 'mudholkar_george'] = 'fisher',
    insert_columns: dict = {}):
    '''
    Estimate the improvement of the corrected model over the baseline model for all folds
    Input:
        baseline_folder: str, the folder containing the baseline results
        corrected_folder: str, the folder containing the corrected results
        n_bootstrap: int, number of bootstrap iterations
        aggregate_method: str, method to aggregate p-values. Options are:
            - 'concatenate': concatenate the input data
            - 'fisher', 'pearson', 'tippett', 'stouffer', 'mudholkar_george': methods to combine p-values, see scipy.stats.combine_pvalues for details
    Output:
        df_p_better: pd.DataFrame, the p-values for significant improvement
        df_p_worse: pd.DataFrame, the p-values for significant worsening
    '''
    dfs_baseline, _ = get_results_from_folder(baseline_folder,redo_pred=args.redo_pred)
    dfs_corrected, demo_col = get_results_from_folder(corrected_folder,redo_pred=args.redo_pred)
    
    ## define the privileged group    
    PRIVILEGED_GROUP_DICT = {
        'gender': 'MALE',
        'race': "WHITE",
        'age': 'below'
    }

    if demo_col == 'age':
        # if the demographic attribute is age, we use the median age as the threshold
        # find median age
        age = [df['sens_attr'] for df in dfs_corrected] + [df['sens_attr'] for df in dfs_baseline]
        age = pd.concat(age)
        median_age = age.median()
        for i in range(len(dfs_corrected)):
            dfs_corrected[i]['sens_attr'] = [f'below{median_age:.1f}' if i < median_age else f'above{median_age:.1f}' for i in dfs_corrected[i]['sens_attr']]
            dfs_baseline[i]['sens_attr'] = [f'below{median_age:.1f}' if i < median_age else f'above{median_age:.1f}' for i in dfs_baseline[i]['sens_attr']]
        privileged_group = f'{PRIVILEGED_GROUP_DICT[demo_col]}{median_age:.1f}'
    else:
        # if the demographic attribute is race or gender, we use the predefined privileged group
        privileged_group = PRIVILEGED_GROUP_DICT[demo_col]
        
    
    # if the demographic attribute is race, we estimate the p-values for each minority group separately
    if demo_col == 'race':
        race_dfs_dict_baseline = {}
        race_dfs_dict_corrected = {}
        for i, min_race in enumerate(MINORITY_RACES):
            # filter the data to include only 2 races
            dfs_baseline_race = []
            dfs_corrected_race = []
            include_races = [privileged_group, min_race]
            for df in dfs_baseline:
                df_race = df.loc[df['sens_attr'].isin(include_races)].reset_index(drop=True).copy()
                dfs_baseline_race.append(df_race)
            for df in dfs_corrected:
                df_race = df.loc[df['sens_attr'].isin(include_races)].reset_index(drop=True).copy()
                dfs_corrected_race.append(df_race)
                
            min_count = [len(x.loc[x['sens_attr']==min_race]) for x in dfs_baseline_race]
            if min(min_count) == 0:
                print(f'Warning: {min_race} has 0 samples in some folds. Skipping this task.')
                continue
            race_dfs_dict_baseline[min_race] = dfs_baseline_race
            race_dfs_dict_corrected[min_race] = dfs_corrected_race
        race_dfs_dict_baseline['All'] = dfs_baseline
        race_dfs_dict_corrected['All'] = dfs_corrected
             
    if demo_col == 'race':
        ## if the demographic attribute is race, estimate the p-values for each minority group separately
        dfs_improv = []
        dfs_p_better = []
        dfs_p_worse = []
        for min_race in race_dfs_dict_baseline.keys():
            dfs_baseline = race_dfs_dict_baseline[min_race]
            dfs_corrected = race_dfs_dict_corrected[min_race]
            insert_columns_race = insert_columns.copy()
            insert_columns_race['sens_attr'] = f'{demo_col} ({privileged_group.lower()} vs. {min_race.lower()})'
            # df_p_worse, df_p_better, fairResult = CV_bootstrap_bias_test(
            #     dfs_race, privileged_group=privileged_group, n_bootstrap=n_bootstrap,aggregate_method=aggregate_method)
            use_privileged_group = privileged_group if args.fix_privileged_group is True else None
            df_improv, df_p_better, df_p_worse = CV_bootstrap_improvement_test(
                dfs_baseline, dfs_corrected, privileged_group=use_privileged_group, n_bootstrap=n_bootstrap,aggregate_method=aggregate_method)
        
            for i, (key, val) in enumerate(insert_columns_race.items()):
                df_improv.insert(i, key, val)
                df_p_better.insert(i, key, val)
                df_p_worse.insert(i, key, val)
            dfs_improv.append(df_improv)
            dfs_p_better.append(df_p_better)
            dfs_p_worse.append(df_p_worse)
        df_improv = pd.concat(dfs_improv)
        df_p_worse = pd.concat(dfs_p_worse)
        df_p_better = pd.concat(dfs_p_better)
    else:
        use_privileged_group = privileged_group if args.fix_privileged_group is True else None
        df_improv, df_p_better, df_p_worse = CV_bootstrap_improvement_test(
            dfs_baseline, dfs_corrected, privileged_group=use_privileged_group, n_bootstrap=n_bootstrap,aggregate_method=aggregate_method)
        
        for i, (key, val) in enumerate(insert_columns.items()):
            df_improv.insert(i, key, val)
            df_p_better.insert(i, key, val)
            df_p_worse.insert(i, key, val)
    return df_improv, df_p_better, df_p_worse

    


if __name__ == '__main__':
    #############################
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument('--baseline_folder', type=str, help='directory to the baseline results folder in google drive',
                        default='/n/data2/hms/dbmi/kyu/lab/NCKU/Fairness/baseline_model'
                        )
    parser.add_argument('--corrected_folder', type=str, help='directory to the corrected results folder in google drive',
                        default='/n/data2/hms/dbmi/kyu/lab/NCKU/Fairness/fairness_model'
                        )
    parser.add_argument('--output_folder', type=str, help='directory to the output folder in google drive',default='TCGA_P_improvement_temp')
    parser.add_argument('--n_bootstrap', type=int,
                        help='number of bootstrap iterations', default=100)
    parser.add_argument('--aggregate_method', type=str, choices=['groupwise','concatenate','fisher', 'pearson', 'tippett', 'stouffer', 'mudholkar_george'],
                        help='method to aggregate p-values', default='fisher')
    parser.add_argument('--proj_idx', type=int, help='the index of the project. If none, runn all', default=24)
    parser.add_argument('--add_perf_difference', action='store_true', help='add performance difference as a fairness metric', default=True)
    parser.add_argument('--redo_pred', action='store_true', help='redo the prediction', default=True)

    parser.add_argument('--fix_privileged_group', action='store_true', help='Fix the privilege group', default=False)
    parser.add_argument('--not_fix_privileged_group',help='Fix the privileged group',action='store_false',dest='fix_privileged_group')

    args = parser.parse_args()
    # end of parsing arguments
    # METRIC_NAMES_DICT = get_metric_names(add_perf_difference=args.add_perf_difference)
    

    print('='*50)
    print('Arguments:')
    print('='*50)

    for key,val in vars(args).items():
        print(f'{key}: {val}')
    print('='*50)

    #############################
    # list all the subfolders in the baseline folders
    baseline_subfolders = glob.glob(join(
        args.baseline_folder, '*ffpe')) + glob.glob(join(args.baseline_folder, '*frozen'))
    corrected_subfolders = glob.glob(join(
        args.corrected_folder, '*ffpe')) + glob.glob(join(args.corrected_folder, '*frozen'))
    baseline_subfolder_names = [basename(f) for f in baseline_subfolders]
    corrected_subfolder_names = [basename(f) for f in corrected_subfolders]
    # find matching folders
    df_baseline = pd.DataFrame(
        {'baseline_folder': baseline_subfolders, 'name': baseline_subfolder_names})
    df_corrected = pd.DataFrame(
        {'corrected_folder': corrected_subfolders, 'name': corrected_subfolder_names})
    df = df_baseline.merge(df_corrected, on='name', how='inner')
    ##
    for i, row in df.iterrows():
        if args.proj_idx is not None and i != args.proj_idx:
            continue
        baseline_folder = row['baseline_folder']
        corrected_folder = row['corrected_folder']
        proj = row['name']
        ###
        attr = proj.split(' ')[0]
        if attr == 'female':
            demo_col = 'gender'
        elif attr == 'white':
            demo_col = 'race'
        else:
            demo_col = 'age'
        task = ' '.join(proj.split(' ')[1:-1])
        sample_type = proj.split(' ')[-1]
        
        ###
        print(f'Processing {proj}')
        insert_columns = {'proj':proj,'task': task, 'sample_type': sample_type,'sens_attr': demo_col}

        df_improv, df_p_better, df_p_worse = main(
            baseline_folder, corrected_folder, n_bootstrap=args.n_bootstrap,
            fix_privileged_group=args.fix_privileged_group,
            aggregate_method=args.aggregate_method,
            add_perf_difference=args.add_perf_difference,
            insert_columns=insert_columns)
        ## save the p-values
        output_folder = join(args.output_folder,f'aggregate_{args.aggregate_method}')
        os.makedirs(output_folder, exist_ok=True)
        df_improv.to_csv(join(output_folder, f'improvement_{proj}({ args.n_bootstrap}samples).csv'))
        df_p_better.to_csv(join(output_folder, f'p_better_{proj}({ args.n_bootstrap}samples).csv'))
        df_p_worse.to_csv(join(output_folder, f'p_worse_{proj}({ args.n_bootstrap}samples).csv'))
