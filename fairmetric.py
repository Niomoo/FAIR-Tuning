import numpy as np
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

class Metrics():
    def __init__(self, predictions = None, labels = None, sensitives = None, projectName = "", sensitiveGroupNames = None, savePath = "", saveConfusionMatrix = False, modify = False, verbose = False):
        if(predictions is not None):
            if(not isinstance(predictions, np.ndarray)):
                if(not isinstance(predictions, list)):
                    raise TypeError(f'sequence predictions: expected numpy.ndarray or list, {type(predictions)} found.')
                predictions = np.array(predictions)

        if(labels is not None):
            if(not isinstance(labels, np.ndarray)):
                if(not isinstance(labels, list)):
                    raise TypeError(f'sequence labels: expected numpy.ndarray or list, {type(labels)} found.')
                labels = np.array(labels)

        if(sensitives is not None):
            if(not isinstance(sensitives, np.ndarray)):
                if(not isinstance(sensitives, list)):
                    raise TypeError(f'sequence sensitives: expected numpy.ndarray or list, {type(sensitives)} found.')
                sensitives = np.array(sensitives)
            
        self.arrPredictions = predictions
        self.arrLabels = labels
        self.arrSensitives = sensitives
        self.strSavePath = savePath
        if(self.strSavePath != ''):
            self.bSaveConfusionMatrix = True
        self.bSaveConfusionMatrix = saveConfusionMatrix
        self.lsConfusionMetrics = []
        self.lsSensitves = []
        self.verbose = verbose
        self.lsMetrics = []
        self.strProjectName = projectName
        self.dictResults = None
        self.lsSensitiveGroupNames = sensitiveGroupNames
        self.lsMinMaxGroups = []
        self.lsGroupCount = []
        if(modify == False and self.arrPredictions is not None and self.arrLabels is not None and self.arrSensitives is not None):
            self.dictResults = self.FairnessMetrics()

    def update(self, predictions = None, labels = None, sensitives = None, savePath = "", projectName = ""):
        if(predictions is not None):
            if(not isinstance(predictions, np.ndarray)):
                if(not isinstance(predictions, list)):
                    raise TypeError(f'sequence predictions: expected numpy.ndarray or list, {type(predictions)} found.')
                predictions = np.array(predictions)

        if(labels is not None):
            if(not isinstance(labels, np.ndarray)):
                if(not isinstance(labels, list)):
                    raise TypeError(f'sequence labels: expected numpy.ndarray or list, {type(labels)} found.')
                labels = np.array(labels)

        if(sensitives is not None):
            if(not isinstance(sensitives, np.ndarray)):
                if(not isinstance(sensitives, list)):
                    raise TypeError(f'sequence sensitives: expected numpy.ndarray or list, {type(sensitives)} found.')
                sensitives = np.array(sensitives)

        if(predictions is None): return
        self.arrPredictions = predictions
        self.arrLabels = labels
        if(sensitives is None): return
        self.arrSensitives = sensitives
        if(savePath != ""): self.strSavePath = savePath
        if(projectName != ""): self.strProjectName = projectName
        self.dictResults = self.FairnessMetrics()

        # return self.dictResults

    def FairnessRule(self):
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
        ConMtx = []
        predictions = self.arrPredictions
        labels      = self.arrLabels
        sensitives  = self.arrSensitives

        OverAll_cnf_matrix = confusion_matrix(labels, predictions)
        ConMtx.append(OverAll_cnf_matrix)
        OverAllACC = np.diag(OverAll_cnf_matrix).sum()/np.sum(OverAll_cnf_matrix)

        uniSens = np.unique(sensitives)
        uniLabels = np.unique(labels)
        
        tpSensCount = []
        self.lsGroupCount = []
        for senidx, modeSensitive in enumerate(uniSens):
            y_pred = predictions[sensitives == modeSensitive]
            y_true = labels[sensitives == modeSensitive]
            tpSensCount.append((len(y_pred), senidx))
            self.lsGroupCount.append(len(y_pred))
            cnf_matrix = confusion_matrix(y_true, y_pred)
            ConMtx.append(cnf_matrix)
            FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
            FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
            TP = np.diag(cnf_matrix)
            TN = cnf_matrix.sum() - (FP + FN + TP)
            FP = FP.astype(float)
            FN = FN.astype(float)
            TP = TP.astype(float)
            TN = TN.astype(float)
            
            if(len(uniLabels) == 2):
                strNR = 'Negative'
                strPR = 'Positve'
                strTNR = 'TrueNegative'
                strTPR = 'TruePositive'
                # Overall accuracy for each class
                ACC.append(((TP+TN)/(TP+FP+FN+TN)).tolist()[0])
                # Sensitivity, hit rate, recall, or true positive rate
                TPR.append((TP/(TP+FN)).tolist()[0])
                # Specificity or true negative rate
                TNR.append((TN/(TN+FP)).tolist()[0])
                # Precision or positive predictive value
                PPV.append((TP/(TP+FP)).tolist()[0])
                # Negative predictive value
                NPV.append((TN/(TN+FN)).tolist()[0])
                # Fall out or false positive rate
                FPR.append((FP/(FP+TN)).tolist()[0])
                # False negative rate
                FNR.append((FN/(TP+FN)).tolist()[0])
                # Prevalence
                PR.append(((TP+FP)/(TP+FP+FN+TN)).tolist()[0])
                # Negative Prevalence
                NR.append(((TN+FN)/(TP+FP+FN+TN)).tolist()[0])
                # # False discovery rate
                # FDR = FP/(TP+FP)
                # total ACC
                TOTALACC.append(np.diag(cnf_matrix).sum()/np.sum(cnf_matrix))
            else:
                strNR = 'NR'
                strPR = 'PR'
                strTNR = 'TNR'
                strTPR = 'TPR'
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

        self.lsConfusionMetrics = ConMtx
        tpSensCount = sorted(tpSensCount)
        self.lsMinMaxGroups = ['' for _ in range(len(tpSensCount))]
        minimun = tpSensCount[0][0]
        for i in tpSensCount:
            if(i[0] == minimun): self.lsMinMaxGroups[i[1]] = '(m)'
            else: break
        tpSensCount.reverse()
        maximum = tpSensCount[0][0]
        for i in tpSensCount:
            if(i[0] == maximum): self.lsMinMaxGroups[i[1]] = '(M)'
            else: break

        alpha = 0.5
        avgEOpp = ((TNR.max(axis = 0)-TNR.min(axis = 0)).sum()+(TPR.max(axis = 0)-TPR.min(axis = 0)).sum())/2
        EOdd = ((TPR+FPR).max(axis = 0)-(TPR+FPR).min(axis = 0)).sum()
        FAT_EO = 1 / (alpha * (1 / (1 - avgEOpp)) + (1 - alpha) * (1 / OverAllACC))
        FAT_ED = 1 / (alpha * (1 / (1 - EOdd)) + (1 - alpha) * (1 / OverAllACC))
        FAT_EO = np.array(FAT_EO)
        FAT_ED = np.array(FAT_ED)

        if(len(uniLabels) == 2):
            return {
                'Count': self.lsGroupCount,
                '': self.lsMinMaxGroups,
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
                'avgEOpp': ((TNR.max(axis = 0)-TNR.min(axis = 0)).sum()+(TPR.max(axis = 0)-TPR.min(axis = 0)).sum())/2,
                'EOpp1': (TPR.max(axis = 0)-TPR.min(axis = 0)).sum(),
                'EOdd': ((TPR+FPR).max(axis = 0)-(TPR+FPR).min(axis = 0)).sum(),
                'PQD': TOTALACC.min()/TOTALACC.max(),
                # 'PQD(class)': (ACC.min(axis = 0)/ACC.max(axis = 0)).mean(),
                'EPPV': (PPV.min(axis = 0)/PPV.max(axis = 0)).mean(),
                'ENPV': (NPV.min(axis = 0)/NPV.max(axis = 0)).mean(),
                f'DPM({strPR})': (PR.min(axis = 0)/PR.max(axis = 0)).mean(),
                f'DPM({strNR})': (NR.min(axis = 0)/NR.max(axis = 0)).mean(),
                f'EOM({strTPR})': (TPR.min(axis = 0)/TPR.max(axis = 0)).mean(),
                f'EOM({strTNR})': (TNR.min(axis = 0)/TNR.max(axis = 0)).mean(),
                'OverAllAcc': OverAllACC,
                'TOTALACC': TOTALACC.tolist(),
                'TOTALACCDIF': TOTALACC.max()-TOTALACC.min(),
                # 'FAT_EO': FAT_EO.tolist(),
                # 'FAT_ED': FAT_ED.tolist(),
                # 'ACCDIF': (ACC.max(axis = 0)-ACC.min(axis = 0)).mean()
            }
        else:
            return {
                'Count': self.lsGroupCount,
                '': self.lsMinMaxGroups,
                'ACC': ACC.tolist(),
                'TPR': TPR.tolist(),
                'TNR': TNR.tolist(),
                # 'PPV': PPV.tolist(),
                # 'NPV': NPV.tolist(),
                # 'PR': PR.tolist(),
                # 'NR': NR.tolist(),
                'FPR': FPR.tolist(),
                'FNR': FNR.tolist(),
                'EOpp0': (TNR.max(axis = 0)-TNR.min(axis = 0)).sum(),
                'avgEOpp': ((TNR.max(axis = 0)-TNR.min(axis = 0)).sum()+(TPR.max(axis = 0)-TPR.min(axis = 0)).sum())/2,
                'EOpp1': (TPR.max(axis = 0)-TPR.min(axis = 0)).sum(),
                'EOdd': ((TPR+FPR).max(axis = 0)-(TPR+FPR).min(axis = 0)).sum(),
                'PQD': TOTALACC.min()/TOTALACC.max(),
                'PQD(class)': (ACC.min(axis = 0)/ACC.max(axis = 0)).mean(),
                # 'EPPV': (PPV.min(axis = 0)/PPV.max(axis = 0)).mean(),
                # 'ENPV': (NPV.min(axis = 0)/NPV.max(axis = 0)).mean(),
                f'DPM({strPR})': (PR.min(axis = 0)/PR.max(axis = 0)).mean(),
                f'DPM({strNR})': (NR.min(axis = 0)/NR.max(axis = 0)).mean(),
                f'EOM({strTPR})': (TPR.min(axis = 0)/TPR.max(axis = 0)).mean(),
                f'EOM({strTNR})': (TNR.min(axis = 0)/TNR.max(axis = 0)).mean(),
                'OverAllAcc': OverAllACC,
                'TOTALACC': TOTALACC.tolist(),
                'TOTALACCDIF': TOTALACC.max()-TOTALACC.min(),
                # 'FAT_EO': FAT_EO.tolist(),
                # 'FAT_ED': FAT_ED.tolist(),
                # 'ACCDIF': (ACC.max(axis = 0)-ACC.min(axis = 0)).mean()
            }

    def FairnessMetrics(self):
        # Two way to calculate fairness metrics
        # 1. init an object and FairnessMetrics()
        # 2. Metrics().FairnessMetrics(...)
        if(self.arrPredictions is None):
            raise ValueError("missing predictions.")            
        if(self.arrLabels is None):
            raise ValueError("missing lables.")
        if(self.arrSensitives is None):
            raise ValueError("missing sensitives.")
        if(self.bSaveConfusionMatrix == True and self.strSavePath == ''):
            raise ValueError("missing save confusion matrix png path.")
        
        self.dictResults = self.FairnessRule()
        lenMultiContains = [isinstance(self.dictResults[i], list) for i in self.dictResults.keys()]
        dfMulti = pd.DataFrame.from_dict(self.dictResults)
        self.lsMetrics = list(dfMulti[np.array(list(self.dictResults.keys()))[(~np.array(lenMultiContains))].tolist()].columns)
        if(self.bSaveConfusionMatrix):
            self.saveConfusionMatrix()
        if(self.verbose):
            print(self.__str__())
            
        return self.dictResults
    
    def getHeads(self):
        result = ""
        if(len(self.lsMetrics) == 0): return ""
        if(self.lsSensitiveGroupNames is None): ssg = range(len(self.dictResults['TOTALACC']))
        else: ssg = self.lsSensitiveGroupNames
        maxPrecision = max([len(i) for i in self.lsMetrics]+[len(f'Group(M) {i}') for i in ssg])
        # if(len(np.unique(self.arrSensitives)) == 2):
        if(True):
            # version 1.0
            # result = f"|{'Project':{maxPrecision}}|{'|'.join([f'{i:{maxPrecision}}' for i in self.lsMetrics[:-1]])}|"
            # result += f"{'Group1':{maxPrecision}}|{self.lsMetrics[-1]:{maxPrecision}}|{'Group2':{maxPrecision}}|"
            result = f"|{'Project':{maxPrecision}}|{'|'.join([f'{i:{maxPrecision}}' for i in self.lsMetrics[:]])}|"
            for ssgidx, i in enumerate(ssg):
                result += f"{f'Group{self.lsMinMaxGroups[ssgidx]} {i}':{maxPrecision}}|"
            result += '\n|'+'|'.join(['-'*maxPrecision for _ in range(len(self.lsMetrics)+len(self.dictResults['TOTALACC'])+1)])+'|'
        return result
    
    def getResults(self, markdownFormat = False):
        if(markdownFormat == False):
            return self.dictResults
        result = ""
        if(self.lsSensitiveGroupNames is None): ssg = range(len(self.dictResults['TOTALACC']))
        else: ssg = self.lsSensitiveGroupNames
        maxPrecision = max([len(i) for i in self.lsMetrics]+[len(f'Group(M) {i}') for i in ssg])
        # if(len(np.unique(self.arrSensitives)) == 2):
        if(True):
            if(len(self.lsMetrics) == 0): return ""

            # result = f"|{self.strProjectName:{maxPrecision}}|{'|'.join([f'{self.dictResults[i]:{maxPrecision}.4f}' for i in self.lsMetrics[:-1]])}|"
            # result += f"{self.dictResults['TOTALACC'][0]:{maxPrecision}.4f}|{self.dictResults[self.lsMetrics[-1]]:{maxPrecision}.4f}|{self.dictResults['TOTALACC'][1]:{maxPrecision}.4f}|"
            result = f"|{self.strProjectName:{maxPrecision}}|{'|'.join([f'{self.dictResults[i]:{maxPrecision}.4f}' for i in self.lsMetrics[:]])}|"
            result += '|'.join([f"{self.dictResults['TOTALACC'][i]:{maxPrecision}.4}" for i in range(len(self.dictResults['TOTALACC']))])+"|"
        return result
    
    def saveConfusionMatrix(self):
        if(len(self.lsSensitves) != len(self.lsConfusionMetrics)):
            self.lsSensitves = [i for i in range(len(self.lsConfusionMetrics))]
        for idx, confusionMatrix in enumerate(self.lsConfusionMetrics):
            _, _ = plot_confusion_matrix(conf_mat = confusionMatrix)
            plt.savefig(f'{self.strSavePath}_{self.lsSensitves[idx]}.png')
            plt.close()
    
    def __str__(self):
        if(self.dictResults is None): return ""
        msg = ''
        labels    = self.arrLabels
        sensitives  = self.arrSensitives
        numUniLabels = len(np.unique(labels))
        numUniSens = len(np.unique(sensitives))+1
        
        msg += "   |"
        for i in range(numUniSens):
            for j in range(numUniLabels):
                msg += f"{j:4}|"
            if(numUniSens > i+1):
                msg += f"{' '*2}|"
        msg += "\n"
        # msg += "\n"+"-"*(11*(numUniSens-1)+4*(1+numUniLabels*numUniSens))+"\n"

        for i in range(numUniLabels):
            msg += f"{i:3}|"
            for j in range(numUniSens):
                for k in range(numUniLabels):
                    msg += f"{self.lsConfusionMetrics[j][i][k]:4}|"
                if(numUniSens > j+1):
                    msg += f"{' '*2}|"
            msg += "\n"
        lenMultiContains = [isinstance(self.dictResults[i], list) for i in self.dictResults.keys()]
        dfMulti = pd.DataFrame.from_dict(self.dictResults)

        dfSingle = dfMulti[np.array(list(self.dictResults.keys()))[(~np.array(lenMultiContains))].tolist()].iloc[:1]
        dfMulti = dfMulti[np.array(list(self.dictResults.keys()))[lenMultiContains].tolist()]
        msg += dfMulti.to_string()
        msg += "\n"
        msg += dfSingle.to_string()
        return msg
    
    def __repr__(self):
        return f'Metrics(predictions = {self.arrPredictions}, labels = {self.arrLabels}, sensitives = {self.arrSensitives})'
    
def showMetrics(results):
    precision = [len(i[0]) for i in results.items()]
    table = "| Setting | " + " | ".join(results.keys()) + " |\n"
    table += "| -----" + " | -----".join(["-"] * len(results.keys())) + "|\n| proposed |"
    for result in results.items():
        try:
            row = f" {result[1]:.4f}" + " |"  
        except ValueError:
            row = f" {result[1]}" + " |"
        table += row
    row += " \n"
    return table