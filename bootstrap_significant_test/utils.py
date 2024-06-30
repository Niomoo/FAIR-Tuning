import pandas as pd
from sklearn import metrics
import numpy as np
# import albumentations as albu
import cv2
import random
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report
# argument library
import argparse

random.seed(24)

HIGHER_BETTER_COLS=['AUC', 'ACC', 'TPR', 'TNR', 'PPV', 'NPV', 'PQD', 'PQD(class)','PR','NR','BAcc',
                    'EPPV', 'ENPV', 'DPM(Positive)', 'DPM(Negative)', 'EOM(Positive)',
                    'EOM(Negative)', 'AUCRatio', 'OverAllAcc', 'OverAllAUC', 'TOTALACC',
                    'FAT_EO', 'FAT_ED', 'FAUCT_EO', 'FAUCT_ED']
LOWER_BETTER_COLS=['FPR', 'FNR', 'EOpp0', 'avgEOpp', 'EOpp1','EBAcc',
                    'EOdd', 'AUCDiff', 'TOTALACCDIF', 'ACCDIF']
PERF_COLS=['AUC', 'ACC', 'TPR', 'TNR', 'PPV', 'NPV','PR','NR','BAcc',
            'FPR', 'FNR', 'OverAllAcc', 'OverAllAUC', 'TOTALACC']
FAIRNESS_COLS=['PQD', 'PQD(class)', 'EPPV', 'ENPV', 'DPM(Positive)', 'DPM(Negative)', 'EOM(Positive)',
                'EOM(Negative)', 'AUCRatio', 'EOpp0', 'avgEOpp', 'EOpp1','EBAcc', 'EOdd', 'AUCDiff', 'TOTALACCDIF', 'ACCDIF',
                'FAT_EO', 'FAT_ED', 'FAUCT_EO', 'FAUCT_ED']

# maps the csv name to the TCGA project name
TCGA_NAME_DICT = {
    # tumor detection
    'LUAD_TumorDetection':  '04_LUAD',
    'CCRCC_TumorDetection':  '06_KIRC',
    'HNSC_TumorDetection':  '07_HNSC',
    'LSCC_TumorDetection':  '10_LUSC',
    # 'BRCA_TumorDetection':  '01_BRCA',
    'PDA_TumorDetection':  '11_PRAD',
    'UCEC_TumorDetection':  '05_UCEC',
    # cancer type classification
    'COAD_READ_512': '_COAD+READ',
    'KIRC_KICH_512': '_KIRC+KICH',
    'KIRP_KICH_512': '_KIRP+KICH',
    'KIRC_KIRP_512': '_KIRC+KIRP',   
    'LGG_GBM_512': '_GBM+LGG',
    'LUAD_LUSC_512': '_LUAD+LUSC',
    'COAD_READ': '_COAD+READ',
    'KIRC_KICH': '_KIRC+KICH',
    'KIRP_KICH': '_KIRP+KICH',
    'KIRC_KIRP': '_KIRC+KIRP',   
    'LGG_GBM': '_GBM+LGG',
    'LUAD_LUSC': '_LUAD+LUSC',
    # cancer subtype classification
    'Breast_ductal_lobular_512': '01_BRCA 1+1',
    'LUAD_BRONCHIOLO-ALVEOLAR_512': '04_LUAD 3+n',
    'Breast_ductal_lobular': '01_BRCA 1+1',
    'LUAD_BRONCHIOLO-ALVEOLAR': '04_LUAD 3+n',
    'LUAD_3_n': '04_LUAD 3+n',
}

def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations
    predicted : Matrix with predicted data, where rows are observations
    Returns
    ---------- 
    list type, with optimal cutoff value   
    """
    fpr, tpr, thresholds = metrics.roc_curve(target, predicted)

    # method 1
    # i = np.arange(len(tpr))
    # roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    # roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    # method 2
    # threshold = thresholds[np.argmin((1 - tpr) ** 2 + fpr ** 2)]

    # method 3
    threshold = thresholds[np.argmax(tpr - fpr)]

    return fpr, tpr, metrics.auc(fpr, tpr), threshold


def performance_metrics(cfs_mtx):
    all_num = (cfs_mtx[0][0]+cfs_mtx[0][1]+cfs_mtx[1][0]+cfs_mtx[1][1])
    tpr = [(cfs_mtx[0][0]/(cfs_mtx[0][0]+cfs_mtx[0][1])),
           (cfs_mtx[1][1]/(cfs_mtx[1][0]+cfs_mtx[1][1]))]
    # tpr (TP/P)
    tnr = [(cfs_mtx[1][1]/(cfs_mtx[1][0]+cfs_mtx[1][1])),
           (cfs_mtx[0][0]/(cfs_mtx[0][0]+cfs_mtx[0][1]))]
    # tnr (TN/N)
    fpr = [1-(cfs_mtx[1][1]/(cfs_mtx[1][0]+cfs_mtx[1][1])),
           1-(cfs_mtx[0][0]/(cfs_mtx[0][0]+cfs_mtx[0][1]))]
    # fpr 1-tnr = (FP/N)

    pp = [(cfs_mtx[0][0]+cfs_mtx[1][0])/all_num,
          (cfs_mtx[0][1]+cfs_mtx[1][1])/all_num]

    return tpr, tnr, fpr, pp, ['True Positive Rate', 'True Negative Rate', 'False Positive Rate', 'Predict Positive Rate']


def FairnessMetrics(predictions, probs, labels, sensitives,
                    previleged_group=None, unprevileged_group=None, add_perf_difference=False):
    '''
    Estimating fairness metrics
    Args:
    * predictions: numpy array, model predictions
    * probs: numpy array, model probabilities
    * labels: numpy array, ground truth labels
    * sensitives: numpy array, sensitive attributes
    * previleged_group: str, previleged group name. If None, the group with the best performance will be used.
    * unprevileged_group: str, unprevileged group name. If None, the group with the worst performance will be used.
    * add_perf_difference: bool, whether to add performance difference metrics

    Returns:
    * results: dict, performance metrics and fairness metrics
    '''
    AUC = []
    ACC = []
    TPR = []
    TNR = []
    PPV = []
    NPV = []
    PR = []
    NR = []
    FPR = []
    FNR = []
    TOTALACC = []
    N = []
    N_0 = []
    N_1 = []
    labels = labels.astype(np.int64)
    sensitives = [str(x) for x in sensitives]
    df = pd.DataFrame({'pred': predictions, 'prob': probs,
                      'label': labels, 'group': sensitives})

    uniSens = np.unique(sensitives)
    ## categorize the groups into majority and minority groups
    # group_types = []
    if previleged_group is not None:
        if unprevileged_group is None:
            # if only the previleged group is provided, we categorize the rest of the groups as unprevileged
            group_types = ['majority' if group == previleged_group else 'minority' for group in uniSens]
        else:
            # if both previleged and unprevileged groups are provided, we categorize the groups accordingly
            # for group in uniSens:
            #     g_type = 'majority' if group == previleged_group else 'minority' if group == unprevileged_group else 'unspecified'
            group_types = ['majority' if group == previleged_group else 'minority' if group == unprevileged_group else 'unspecified' for group in uniSens]
    elif unprevileged_group is not None:
        # if only the unprevileged group is provided, we categorize the rest of the groups as previleged
        group_types = ['minority' if group == unprevileged_group else 'majority' for group in uniSens]
        # for group in uniSens:
        # g_type = 'minority' if group == unprevileged_group else 'majority'
        # group_types.append(g_type)
    else:
        # if both previleged and unprevileged groups are not provided, set to unspecified
        group_types = ['unspecified' for group in uniSens]
        # for group in uniSens:
        #     g_type =  'unspecified'
        #     group_types.append(g_type)
    ##

    for modeSensitive in uniSens:
        modeSensitive = str(modeSensitive)
        df_sub = df.loc[df['group'] == modeSensitive]
        y_pred = df_sub['pred'].to_numpy()
        y_prob = df_sub['prob'].to_numpy()
        y_true = df_sub['label'].to_numpy()
        # y_pred = predictions[sensitives == modeSensitive]
        # y_prob = probs[sensitives == modeSensitive]
        # y_true = labels[sensitives == modeSensitive]

        if len(y_pred) == 0:
            continue
        cnf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        CR = classification_report(y_true, y_pred, labels=[
                                   0, 1], output_dict=True, zero_division=0)
        # AUC
        if len(set(y_true)) == 1:
            AUC.append(np.nan)
        else:
            AUC.append((metrics.roc_auc_score(y_true, y_prob)))
        N.append(CR['macro avg']['support'])
        N_0.append(CR['0']['support'])
        N_1.append(CR['1']['support'])
        # Overall accuracy for each class
        ACC.append(np.trace(cnf_matrix)/np.sum(cnf_matrix))
        # Sensitivity, hit rate, recall, or true positive rate
        TPR.append(CR['1']['recall'] if CR['1']['support'] > 0 else np.nan)
        # Specificity or true negative rate
        TNR.append(CR['0']['recall'] if CR['0']['support'] > 0 else np.nan)
        # Precision or positive predictive value
        PPV.append(CR['1']['precision'] if np.sum(
            cnf_matrix[:, 1]) > 0 else np.nan)
        # Negative predictive value
        NPV.append(CR['0']['precision'] if np.sum(
            cnf_matrix[:, 0]) > 0 else np.nan)
        # Fall out or false positive rate
        FPR.append(1-CR['0']['recall'] if CR['0']['support'] > 0 else np.nan)
        # False negative rate
        FNR.append(1-CR['1']['recall'] if CR['1']['support'] > 0 else np.nan)
        # Prevalence
        PR.append(np.sum(cnf_matrix[:, 1]) / np.sum(cnf_matrix))
        # Negative Prevalence
        NR.append(np.sum(cnf_matrix[:, 0]) / np.sum(cnf_matrix))
        # # False discovery rate
        # FDR = FP/(TP+FP)
        # total ACC
        TOTALACC.append(np.trace(cnf_matrix)/np.sum(cnf_matrix))

    OverAll_cnf_matrix = confusion_matrix(predictions, labels)
    OverAllACC = np.trace(OverAll_cnf_matrix)/np.sum(OverAll_cnf_matrix)
    try:
        OverAllAUC = metrics.roc_auc_score(labels, probs)
    except:
        OverAllAUC = np.nan

    AUC = np.array(AUC)
    ACC = np.array(ACC)
    TPR = np.array(TPR)
    TNR = np.array(TNR)
    PPV = np.array(PPV)
    NPV = np.array(NPV)
    PR = np.array(PR)
    NR = np.array(NR)
    FPR = np.array(FPR)
    FNR = np.array(FNR)
    TOTALACC = np.array(TOTALACC)

    df_perf = pd.DataFrame(
        {'AUC': AUC, 'ACC': ACC, 'TPR': TPR, 'TNR': TNR, 'PPV': PPV, 'NPV': NPV, 'BAcc': (np.array(TPR)+np.array(TNR))/2,
         'PR': PR, 'NR': NR, 'FPR': FPR, 'FNR': FNR, 'TOTALACC': TOTALACC,'OverAllAcc': OverAllACC,'Odd1': TPR+FPR,'Odd0':TNR+FNR,
         'OverAllAUC': OverAllAUC}, index=uniSens)
    lower_better_metrics = ['FPR', 'FNR']
    higher_better_metrics = ['TPR', 'TNR', 'NPV','BAcc',
                             'PPV', 'TOTALACC','OverAllAcc','OverAllAUC', 'AUC', 'ACC', 'PR', 'NR', 'Odd1', 'Odd0']

    if previleged_group is not None:
        perf_previleged = df_perf.loc[previleged_group]
    else:
        perf_previleged = pd.concat([
            df_perf[higher_better_metrics].max(),
            df_perf[lower_better_metrics].min()])
    if unprevileged_group is not None:
        perf_unprevileged = df_perf.loc[unprevileged_group]
    elif previleged_group is not None:
        perf_not_previleged = df_perf.drop(
            previleged_group)
        # perf_unprevileged = perf_not_previleged.min()
        perf_unprevileged = pd.concat([
            perf_not_previleged[higher_better_metrics].min(),
            perf_not_previleged[lower_better_metrics].max()])
    else:
        # perf_unprevileged = df_perf[higher_better_metrics].min()
        perf_unprevileged = pd.concat([
            df_perf[higher_better_metrics].min(),
            df_perf[lower_better_metrics].max()])

    perf_diff = perf_previleged - perf_unprevileged
    perf_ratio = perf_unprevileged / perf_previleged

    BAcc = (TPR + TNR) / 2
    alpha = 0.5
    avgEOpp = (perf_diff["TNR"] + perf_diff["TPR"]) / 2
    EOdd = -perf_diff["Odd1"]
    FAT_EO = 1 / (alpha * (1 / (1 - avgEOpp)) + (1 - alpha) * (1 / OverAllACC))
    FAT_ED = 1 / (alpha * (1 / (1 - EOdd)) + (1 - alpha) * (1 / OverAllACC))
    FAT_EO = np.array(FAT_EO)
    FAT_ED = np.array(FAT_ED)

    FAUCT_EO = 1 / (alpha * (1 / (1 - avgEOpp)) + (1 - alpha) * (1 / OverAllAUC))
    FAUCT_ED = 1 / (alpha * (1 / (1 - EOdd)) + (1 - alpha) * (1 / OverAllAUC))
    FAUCT_EO = np.array(FAUCT_EO)
    FAUCT_ED = np.array(FAUCT_ED)

    results = {
        "sensitiveAttr": uniSens,
        "group_type": group_types,
        "N_0": N_0,
        "N_1": N_1,
        "AUC": AUC,
        "ACC": ACC,
        "TPR": TPR,
        "TNR": TNR,
        "PPV": PPV,
        "NPV": NPV,
        "BAcc": BAcc,
        "PR": PR,
        "NR": NR,
        "FPR": FPR,
        "FNR": FNR,
        "EOpp0": perf_diff["TNR"],
        "avgEOpp": avgEOpp,
        "EOpp1": perf_diff["TPR"],
        "EBAcc": perf_diff["BAcc"],
        "EOdd": (-perf_diff["Odd1"]),
        "EOdd0": (-perf_diff["Odd0"]),
        "EOdd1": (-perf_diff["Odd1"]),
        "PQD": perf_ratio["TOTALACC"],
        "PQD(class)": perf_ratio["TOTALACC"],
        "EPPV": perf_ratio["PPV"],
        "ENPV": perf_ratio["NPV"],
        "DPM(Positive)": perf_ratio["PR"],
        "DPM(Negative)": perf_ratio["NR"],
        "EOM(Positive)": perf_ratio["TPR"],
        "EOM(Negative)": perf_ratio["TNR"],
        "AUCRatio": perf_ratio["AUC"],
        "AUCDiff": perf_diff["AUC"],
        "OverAllAcc": OverAllACC,
        "OverAllAUC": OverAllAUC,
        "TOTALACC": TOTALACC,
        "TOTALACCDIF": perf_diff["TOTALACC"],
        "ACCDIF": perf_diff["ACC"],
        "FAT_EO": FAT_EO.tolist(),
        "FAT_ED": FAT_ED.tolist(),
        "FAUCT_EO": FAUCT_EO.tolist(),
        "FAUCT_ED": FAUCT_ED.tolist(),
    }
    if add_perf_difference:
        results, new_cols = get_perf_diff(
            results,perf_metrics=PERF_COLS,
            privileged_group=previleged_group,demo_col='sensitiveAttr')
    return results

def get_metric_names(add_perf_difference=False):
    '''
    returns a dictionary of metric list
    '''
    metrics_list = {
        'higher_better_metrics': HIGHER_BETTER_COLS.copy(),
        'lower_better_metrics': LOWER_BETTER_COLS.copy(),
        'perf_metrics': PERF_COLS.copy(),
        'fairness_metrics': FAIRNESS_COLS.copy()
    }
    ## if add_perf_difference == True, we add the performance difference as a fairness metric
    if add_perf_difference:
        for col in PERF_COLS:
            metrics_list['fairness_metrics'].append(f'{col}_diff')
            if col in HIGHER_BETTER_COLS:
                metrics_list['lower_better_metrics'].append(f'{col}_diff')
            elif col in LOWER_BETTER_COLS:
                metrics_list['higher_better_metrics'].append(f'{col}_diff')
    metrics_list['higher_better_fairness_metrics'] = [x for x in metrics_list['fairness_metrics'] if x in metrics_list['higher_better_metrics']]
    metrics_list['lower_better_fairness_metrics'] = [x for x in metrics_list['fairness_metrics'] if x in  metrics_list['lower_better_metrics']]
    return metrics_list

# for inpath in BASES:
def get_perf_diff(fairResult,perf_metrics=PERF_COLS,privileged_group=None,demo_col = 'sensitiveAttr'):
    '''
    Add performance metric to the dataframe
    '''
    df = pd.DataFrame(fairResult)
    new_cols = []
    for col in perf_metrics:
        if col in df.keys():
            if privileged_group is not None:
                val_privileged = df.loc[df[demo_col]==privileged_group][col]
                val_others = df.loc[df[demo_col]!=privileged_group][col]
            else:
                val_privileged = df[col]
                val_others = df[col]
            val_privileged = val_privileged.max()
            val_others = val_others.min()
            
            
            # df[f'{col}Diff'] = val_privileged - val_others
            fairResult[col+'_diff'] = val_privileged - val_others
            new_cols.append(col+'_diff')
            
    return fairResult, new_cols


def FairnessMetricsMultiClass(predictions, labels, sensitives):
    # TODO: fix the bug like FairnessMetrics()
    raise NotImplementedError
    ACC = []
    TPR = []
    TNR = []
    PPV = []
    NPV = []
    PR = []
    NR = []
    FPR = []
    FNR = []
    TOTALACC = []

    uniSens = np.unique(sensitives)
    for modeSensitive in uniSens:
        y_pred = predictions[sensitives == modeSensitive]
        y_true = labels[sensitives == modeSensitive]
        cnf_matrix = confusion_matrix(y_true, y_pred)
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)
        # Overall accuracy for each class
        ACC.append(((TP+TN)/(TP+FP+FN+TN)).tolist())
        # Sensitivity, hit rate, recall, or true positive rate
        TPR.append((TP/(TP+FN)).tolist())
        # Specificity or true negative rate
        TNR.append((TN/(TN+FP)).tolist())
        # Precision or positive predictive value
        PPV.append((TP/(TP+FP)).tolist())
        # Negative predictive value
        NPV.append((TN/(TN+FN)).tolist())
        # Fall out or false positive rate
        FPR.append((FP/(FP+TN)).tolist())
        # False negative rate
        FNR.append((FN/(TP+FN)).tolist())
        # Prevalence
        PR.append(((TP+FP)/(TP+FP+FN+TN)).tolist()[0])
        # Negative Prevalence
        NR.append(((TN+FN)/(TP+FP+FN+TN)).tolist()[0])
        # # False discovery rate
        # FDR = FP/(TP+FP)
        # total ACC
        TOTALACC.append(np.diag(cnf_matrix).sum()/np.sum(cnf_matrix))

    OverAll_cnf_matrix = confusion_matrix(labels, predictions)
    OverAllACC = np.diag(OverAll_cnf_matrix).sum()/np.sum(OverAll_cnf_matrix)

    ACC = np.array(ACC)
    TPR = np.array(TPR)
    TNR = np.array(TNR)
    PPV = np.array(PPV)
    NPV = np.array(NPV)
    PR = np.array(PR)
    NR = np.array(NR)
    FPR = np.array(FPR)
    FNR = np.array(FNR)
    TOTALACC = np.array(TOTALACC)

    return {
        'ACC': ACC.tolist(),
        'TPR': TPR.tolist(),
        'TNR': TNR.tolist(),
        'PPV': PPV.tolist(),
        'NPV': NPV.tolist(),
        'PR': PR.tolist(),
        'NR': NR.tolist(),
        'FPR': FPR.tolist(),
        'FNR': FNR.tolist(),
        'EOpp0': (TNR.max(axis=0)-TNR.min(axis=0)).sum(),
        'EOpp1': (TPR.max(axis=0)-TPR.min(axis=0)).sum(),
        'EOdd': ((TPR+FPR).max(axis=0)-(TPR+FPR).min(axis=0)).sum(),
        'PQD': TOTALACC.min()/TOTALACC.max(),
        'PQD(class)': (ACC.min(axis=0)/ACC.max(axis=0)).mean(),
        'EPPV': (PPV.min(axis=0)/PPV.max(axis=0)).mean(),
        'ENPV': (NPV.min(axis=0)/NPV.max(axis=0)).mean(),
        'DPM': (PR.min(axis=0)/PR.max(axis=0)).mean(),
        'EOM': (TPR.min(axis=0)/TPR.max(axis=0)).mean(),
        'OverAllAcc': OverAllACC,
        'TOTALACC': TOTALACC.tolist(),
        'TOTALACCDIF': TOTALACC.max()-TOTALACC.min(),
        'ACCDIF': (ACC.max(axis=0)-ACC.min(axis=0)).mean()
    }
