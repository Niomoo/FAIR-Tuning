import pandas as pd
from sklearn import metrics
import numpy as np
import random
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import loralib as lora
import math
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
## argument library
import argparse

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
    tpr = [ (cfs_mtx[0][0]/(cfs_mtx[0][0]+cfs_mtx[0][1])), (cfs_mtx[1][1]/(cfs_mtx[1][0]+cfs_mtx[1][1]))]
    # tpr (TP/P)
    tnr = [ (cfs_mtx[1][1]/(cfs_mtx[1][0]+cfs_mtx[1][1])), (cfs_mtx[0][0]/(cfs_mtx[0][0]+cfs_mtx[0][1]))]
    # tnr (TN/N)
    fpr = [ 1-(cfs_mtx[1][1]/(cfs_mtx[1][0]+cfs_mtx[1][1])), 1-(cfs_mtx[0][0]/(cfs_mtx[0][0]+cfs_mtx[0][1]))]
    # fpr 1-tnr = (FP/N)

    pp = [(cfs_mtx[0][0]+cfs_mtx[1][0])/all_num, (cfs_mtx[0][1]+cfs_mtx[1][1])/all_num]

    return tpr, tnr, fpr, pp, ['True Positive Rate', 'True Negative Rate', 'False Positive Rate', 'Predict Positive Rate']

def FairnessMetrics(predictions, labels, sensitives):
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
    # predictions = [list(p)[0] for p in predictions]
    # labels = [list(l.astype(np.int64))[0] for l in labels]
    sensitives = [str(x) for x in sensitives]
    df = pd.DataFrame({'pred':predictions,'label':labels,'group': sensitives}) # used DataFrame because numpy string arrays have some issue with conversion

    uniSens = np.unique(sensitives)
    for modeSensitive in uniSens:
        modeSensitive = str(modeSensitive)
        df_sub = df.loc[df['group']==modeSensitive]
        y_pred = df_sub['pred'].to_numpy()
        # y_prob = df_sub['prob'].to_numpy()
        y_true = df_sub['label'].to_numpy()
        # y_pred = predictions[sensitives == modeSensitive]
        # y_prob = probs[sensitives == modeSensitive]
        # y_true = labels[sensitives == modeSensitive]

        if len(y_pred) == 0:
            continue
        cnf_matrix = confusion_matrix(y_true, y_pred,labels=[0,1])
        CR = classification_report(y_true, y_pred,labels=[0,1],output_dict=True)
        # AUC
        # if len(set(y_true)) == 1:
        #     AUC.append(float("NaN"))
        # else:
        #     AUC.append((metrics.roc_auc_score(y_true, y_prob)))
        N.append(CR['macro avg']['support'])
        N_0.append(CR['0']['support'])
        N_1.append(CR['1']['support'])
        # Overall accuracy for each class
        ACC.append(np.trace(cnf_matrix)/np.sum(cnf_matrix))
        # Sensitivity, hit rate, recall, or true positive rate
        TPR.append(CR['1']['recall'] if CR['1']['support']> 0 else np.nan)
        # Specificity or true negative rate
        TNR.append(CR['0']['recall'] if CR['0']['support']> 0 else np.nan)
        # Precision or positive predictive value
        PPV.append(CR['1']['precision'] if np.sum(cnf_matrix[:,1]) > 0 else np.nan)
        # Negative predictive value
        NPV.append(CR['0']['precision'] if np.sum(cnf_matrix[:,0]) > 0 else np.nan)
        # Fall out or false positive rate
        FPR.append(1-CR['0']['recall'] if CR['0']['support']> 0 else np.nan)
        # False negative rate
        FNR.append(1-CR['1']['recall'] if CR['1']['support']> 0 else np.nan)
        # Prevalence
        PR.append(np.sum(cnf_matrix[:,1]) / np.sum(cnf_matrix))
        # Negative Prevalence
        NR.append( np.sum(cnf_matrix[:,0]) / np.sum(cnf_matrix))
        # # False discovery rate
        # FDR = FP/(TP+FP)
        # total ACC
        TOTALACC.append(np.trace(cnf_matrix)/np.sum(cnf_matrix))

    OverAll_cnf_matrix = confusion_matrix(predictions, labels)
    OverAllACC = np.trace(OverAll_cnf_matrix)/np.sum(OverAll_cnf_matrix)
    # OverAllAUC = metrics.roc_auc_score(labels, probs)

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

    return {
        'sensitiveAttr': uniSens,
        'N_0': N_0,
        'N_1': N_1,
        # 'AUC': AUC,
        'ACC': ACC,
        'TPR': TPR,
        'TNR': TNR,
        'PPV': PPV,
        'NPV': NPV,
        'PR': PR,
        'NR': NR,
        'FPR': FPR,
        'FNR': FNR,
        'EOpp0': np.nanmax(TNR,axis=0)-np.nanmin(TNR,axis=0),
        'EOpp1': np.nanmax(TPR,axis=0)-np.nanmin(TPR,axis=0),
        'EOdd': np.nanmax(TPR+FPR) - np.nanmin(TPR+FPR),
        'EOdd0':  np.nanmax(TNR+FNR) - np.nanmin(TNR+FNR),
        'EOdd1': np.nanmax(TPR+FPR) - np.nanmin(TPR+FPR),
        'PQD': np.nanmin(TOTALACC)/ np.nanmax(TOTALACC),
        'PQD(class)': np.mean(np.nanmin(ACC)/np.nanmax(ACC)),
        'EPPV': np.nanmin(PPV)/np.nanmax(PPV),
        'ENPV': np.nanmin(NPV)/np.nanmax(NPV),
        'DPM(Positive)': np.nanmin(PR)/np.nanmax(PR),
        'DPM(Negative)': np.nanmin(NR)/np.nanmax(NR), 
        'EOM(Positive)': np.nanmin(TPR)/np.nanmax(TPR),
        'EOM(Negative)': np.nanmin(TNR)/np.nanmax(TNR),
        'OverAllAcc': OverAllACC,
        'TOTALACC': TOTALACC,
        'TOTALACCDIF': np.nanmax(TOTALACC)-np.nanmin(TOTALACC),
        'ACCDIF': np.nanmax(ACC)-np.nanmin(ACC)
    }

def FairnessMetricsMultiClass(predictions, labels, sensitives):
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

    OverAll_cnf_matrix = confusion_matrix(predictions, labels)
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
        'EOpp0': (TNR.max(axis = 0)-TNR.min(axis = 0)).sum(),
        'EOpp1': (TPR.max(axis = 0)-TPR.min(axis = 0)).sum(),
        'EOdd': ((TPR+FPR).max(axis = 0)-(TPR+FPR).min(axis = 0)).sum(),
        'PQD': TOTALACC.min()/TOTALACC.max(),
        'PQD(class)': (ACC.min(axis = 0)/ACC.max(axis = 0)).mean(),
        'EPPV': (PPV.min(axis = 0)/PPV.max(axis = 0)).mean(),
        'ENPV': (NPV.min(axis = 0)/NPV.max(axis = 0)).mean(), 
        'DPM': (PR.min(axis = 0)/PR.max(axis = 0)).mean(),
        'EOM': (TPR.min(axis = 0)/TPR.max(axis = 0)).mean(),
        'OverAllAcc': OverAllACC,
        'TOTALACC': TOTALACC.tolist(),
        'TOTALACCDIF': TOTALACC.max()-TOTALACC.min(),
        'ACCDIF': (ACC.max(axis = 0)-ACC.min(axis = 0)).mean()
    }

def RaceMetrics(predictions, labels):
    cnf_matrix = confusion_matrix(predictions, labels)
    OverAllACC = np.diag(cnf_matrix).sum()/np.sum(cnf_matrix)
    return OverAllACC

def SurvivalMetrics(predictedSurvivalTime, trueSurvivalTime, events, sensitives):
    # 1. C-index
    try:
        c_index = concordance_index(trueSurvivalTime, predictedSurvivalTime, events)
    except ZeroDivisionError:
        c_index = '--'
    
    # 2. Survival time
    med_days = np.median(predictedSurvivalTime)
    T_big = []
    T_small = []
    E_big = []
    E_small = []
    for i in range(0, len(predictedSurvivalTime)):
        if predictedSurvivalTime[i] > med_days:
            T_big.append(trueSurvivalTime[i])
            E_big.append(events[i])
        else:
            T_small.append(trueSurvivalTime[i])
            E_small.append(events[i])
    try:
        logrank_results = logrank_test(T_big, T_small, event_observed_A=E_big, event_observed_B=E_small)
    except:
        logrank_results = {'p_value': '--'}

    uniSens = np.unique(sensitives)
    c_index_sens = []
    T_big_sens = []
    T_small_sens = []
    E_big_sens = []
    E_small_sens = []
    p_big_sens = []
    p_small_sens = []
    logrank_results_sens = []
    c_index_sens_long = []
    c_index_sens_short = []
    for modeSensitive in uniSens:
        # 3. C-index by sensitive attribute
        predictedSurvivalTime_sub = predictedSurvivalTime[sensitives == modeSensitive]
        trueSurvivalTime_sub = trueSurvivalTime[sensitives == modeSensitive]
        events_sub = events[sensitives == modeSensitive]
        try:
            c_index_sens.append(concordance_index(trueSurvivalTime_sub, predictedSurvivalTime_sub, events_sub))
        except:
            c_index_sens.append('--')    
        # 4. Survival time by sensitive attribute
        med_days_sub = np.median(predictedSurvivalTime_sub)
        for i in range(0, len(predictedSurvivalTime_sub)):
            if predictedSurvivalTime_sub[i] > med_days_sub:
                T_big_sens.append(trueSurvivalTime_sub[i])
                E_big_sens.append(events_sub[i])
                p_big_sens.append(predictedSurvivalTime_sub[i])
            else:
                T_small_sens.append(trueSurvivalTime_sub[i])
                E_small_sens.append(events_sub[i])
                p_small_sens.append(predictedSurvivalTime_sub[i])
        logrank_results_sens.append(logrank_test(T_big_sens, T_small_sens, event_observed_A=E_big_sens, event_observed_B=E_small_sens))
        
        # 5. C-index of different term by sensitive attribute
        try:
            c_index_sens_long.append(concordance_index(T_big_sens, p_big_sens, E_big_sens))
        except ZeroDivisionError:
            c_index_sens_long.append('--')
        try:
            c_index_sens_short.append(concordance_index(T_small_sens, p_small_sens, E_small_sens))
        except ZeroDivisionError:
            c_index_sens_short.append('--')
    if len(c_index_sens) == 0:
        c_index_diff = '--'
    else:
        try:
            c_index_diff = max(c_index_sens) - min(c_index_sens)
        except TypeError:
            c_index_diff = '--'
    if len(c_index_sens_long) == 0:
        c_index_diff_long = '--'
    else:
        try:
            c_index_diff_long = max(c_index_sens_long) - min(c_index_sens_long)
        except TypeError:
            c_index_diff_long = '--'
    if len(c_index_sens_short) == 0:
        c_index_diff_short = '--'
    else:
        try:
            c_index_diff_short = max(c_index_sens_short) - min(c_index_sens_short)
        except TypeError:
            c_index_diff_short = '--'

    return {
        'c_index': c_index,
        'c_0': c_index_sens[0] if len(c_index_sens) > 1 else '--',
        'c_diff': c_index_diff,
        'c_1': c_index_sens[1] if len(c_index_sens) > 1 else '--',
        'c_long0': c_index_sens_long[0] if len(c_index_sens_long) > 1 else '--',
        'c_long_diff': c_index_diff_long,
        'c_long1': c_index_sens_long[1] if len(c_index_sens_long) > 1 else '--',
        'c_short0': c_index_sens_short[0] if len(c_index_sens_short) > 1 else '--',
        'c_short_diff': c_index_diff_short,
        'c_short1': c_index_sens_short[1] if len(c_index_sens_short) > 1 else '--',
        'logrank': logrank_results.p_value if type(logrank_results) != dict else '--',
        'logrank0': logrank_results_sens[0].p_value if len(logrank_results_sens) > 1 else '--',
        'logrank1': logrank_results_sens[1].p_value if len(logrank_results_sens) > 1 else '--',
    }


# ### ================================================== ####

def load_weights(model, path):
    pretrained_dict = torch.load(path)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

# ### ================================================== ####

def race_acc(race_c, race_i, r_labels):
    race_labels_c = r_labels
    race_labels_i = 1 - r_labels
    
    mask_c = race_c >= 0.5
    race_c[mask_c] = 1
    race_c[~mask_c] = 0
    acc_c = metrics.accuracy_score(race_labels_c, race_c)
    
    mask_i = race_i >= 0.5
    race_i[mask_i] = 1
    race_i[~mask_i] = 0
    acc_i = metrics.accuracy_score(race_labels_i, race_i)
    return acc_c, acc_i

# ### ================================================== ####


def replace_linear(model, rank=4, dropout=0.3):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            setattr(model, name, lora.Linear(module.in_features, module.out_features, r=rank, lora_dropout=dropout))
        else:
            replace_linear(module)

# ### =================================================== ####

class FairnessLoss(nn.Module):
    def __init__(self, fairness_weight):
        super(FairnessLoss, self).__init__()
        self.fairness_weight = fairness_weight

    def forward(self, pred_sens, sensitive_features):
        # print(pred_sens)
        sensitive_predictions = torch.sigmoid(pred_sens)

        # print(sensitive_predictions)
        # print(sensitive_features)
        # L1-loss
        error_rates = torch.abs(sensitive_predictions - sensitive_features.unsqueeze(1))
        max_error_rate = torch.mean(error_rates)

        # print("error_rate", error_rates)
        # print("max_error_rate", max_error_rate)
        loss = self.fairness_weight * max_error_rate
        # print(loss)
        return loss

# ### =================================================== ####

def weibull_loss(shape, scale, time, event):
    y = time
    u = event
    a = scale
    b = shape
    hazard0 = (y + 1e-35/a)**b
    hazard1 = (y + 1/a)**b
    return -torch.mean(u * torch.log(torch.exp(hazard1 - hazard0) - 1) - hazard1)

# ### =================================================== ####

def survival_loss_function(shape, scale, time, event, lengths, group):
    loss = 0.
    group_of_loss = {}
    group_length = {}
    lengths = list(lengths)
    for i in range(len(lengths)):
        # print(preds[i], sur_preds[i], targets[i], times[i])
        loss += weibull_loss(shape[i], scale[i], time[i], event[i]) 
        if group[i] not in group_of_loss:
            group_length[group[i]] = 1
            group_of_loss[group[i]] = weibull_loss(shape[i], scale[i], time[i], event[i])
        else:
            group_length[group[i]] += 1
            group_of_loss[group[i]] += weibull_loss(shape[i], scale[i], time[i], event[i])
    loss = loss / len(lengths)
    group_of_loss = {k: v / group_length[k] for k, v in group_of_loss.items()}
    # print('loss', loss)
    # print('group_of_loss', group_of_loss)
    return loss, group_of_loss

# ### =================================================== ####

def loss_function(loss_func, preds, targets, lengths, group):
    loss = 0.
    group_of_loss = {}
    group_length = {}
    lengths = list(lengths)
    for i in range(len(lengths)):
        loss += loss_func(preds[i, :lengths[i]], targets[i, :lengths[i]]) 
        if group[i] not in group_of_loss:
            group_length[group[i]] = 1
            group_of_loss[group[i]] = loss_func(preds[i, :lengths[i]], targets[i, :lengths[i]])
        else:
            group_length[group[i]] += 1
            group_of_loss[group[i]] += loss_func(preds[i, :lengths[i]], targets[i, :lengths[i]])
    loss = loss / len(lengths)
    group_of_loss = {k: v / group_length[k] for k, v in group_of_loss.items()}
    return loss, group_of_loss

# ### =================================================== ####

def batch_resample(batch_size, group_loss):
    group_weight = {}
    group_sampling_probability = {}
    group_samples = {}
    for group in group_loss:
        group_weight[group] = 1 / group_loss[group]
    for group in group_weight:
        group_sampling_probability[group] = group_weight[group] / sum(group_weight.values())
        if math.isnan(group_sampling_probability[group]):
            group_samples[group] = 0
        else:
            group_samples[group] = int(group_sampling_probability[group] * batch_size)
        if(group_samples[group] == 0):
            group_samples[group] = 1

    total = sum(group_samples.values())
    if(total < batch_size):
        group_samples[max(group_samples, key=group_samples.get)] += batch_size - total
    elif(total > batch_size):
        group_samples[max(group_samples, key=group_samples.get)] -= total - batch_size
    # print('group_samples_after: ', group_samples)
    return group_samples

# ### =================================================== ####

def eo_constraint(p, y, a):
    fpr = torch.abs(torch.sum(p * (1 - y) * a) / (torch.sum(a) + 1e-5) - torch.sum(p * (1 - y) * (1 - a)) / (
                torch.sum(1 - a) + 1e-5))
    fnr = torch.abs(torch.sum((1 - p) * y * a) / (torch.sum(a) + 1e-5) - torch.sum((1 - p) * y * (1 - a)) / (
                torch.sum(1 - a) + 1e-5))
    return fpr, fnr


def di_constraint(p, a):
    di = -1 * torch.min((torch.sum(a * p) / torch.sum(a)) / (torch.sum((1 - a) * p) / torch.sum((1 - a))),
                        (torch.sum((1 - a) * p) / torch.sum((1 - a))) / (torch.sum(a * p) / torch.sum(a)))
    return di


def dp_constraint(p, a):
    dp = torch.abs((torch.sum(a * p) / torch.sum(a)) - (torch.sum((1 - a) * p) / torch.sum((1 - a))))
    return dp


def ae_constraint(criterion, log_softmax, y, a):
    loss_p = criterion(log_softmax[a == 1], y[a == 1]) + 1e-6
    loss_n = criterion(log_softmax[a == 0], y[a == 0]) + 1e-6
    return torch.abs(loss_p - loss_n)


def mmf_constraint(criterion, log_softmax, y, a):
    # loss_p = criterion(log_softmax[a == 1], y[a == 1])
    # loss_n = criterion(log_softmax[a == 0], y[a == 0])
    # return torch.max(loss_p, loss_n)
    y_p_a = y + a
    y_m_a = y - a
    if len(y[y_p_a == 2]) > 0:
        loss_1 = criterion(log_softmax[y_p_a == 2], y[y_p_a == 2])  # (1, 1)
    else:
        loss_1 = torch.tensor(0.0).cuda()
    if len(y[y_p_a == 0]) > 0:
        loss_2 = criterion(log_softmax[y_p_a == 0], y[y_p_a == 0])  # (0, 0)
    else:
        loss_2 = torch.tensor(0.0).cuda()
    if len(y[y_m_a == 1]) > 0:
        loss_3 = criterion(log_softmax[y_m_a == 1], y[y_m_a == 1])  # (1, 0)
    else:
        loss_3 = torch.tensor(0.0).cuda()
    if len(y[y_m_a == -1]) > 0:
        loss_4 = criterion(log_softmax[y_m_a == -1], y[y_m_a == -1])  # (0, 1)
    else:
        loss_4 = torch.tensor(0.0).cuda()
    return torch.max(torch.max(loss_1, loss_2), torch.max(loss_3, loss_4))

def calculate_cindex(model, data_loader):
    predicted_survival_times = []
    true_survival_times = []
    events = []

    # Iterate over the data loader
    for batch in data_loader:
        with torch.no_grad():
            # Forward pass to get the predicted shape and scale parameters
            shape, scale = model(batch['images'])

            # Calculate the predicted survival times using the Weibull distribution
            predicted_survival_time = scale * torch.exp(torch.log(batch['time'] + 1e-8) / shape)

            predicted_survival_times.append(predicted_survival_time)
            true_survival_times.append(batch['time'])
            events.append(batch['event'])

    # Concatenate the predicted and true survival times
    predicted_survival_times = torch.cat(predicted_survival_times)
    true_survival_times = torch.cat(true_survival_times)
    events = torch.cat(events)

    # Calculate the C-Index using lifelines' concordance_index function
    c_index = concordance_index(true_survival_times, predicted_survival_times, events)

    return c_index
