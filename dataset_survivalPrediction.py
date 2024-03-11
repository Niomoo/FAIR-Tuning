from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import Dataset, Sampler
import cv2, matplotlib, os, sys, random
import numpy as np
import pandas as pd
import glob
import torch

class generateDataSet():
    def __init__(self, cancer, sensitive, fold, seed = 24):
        '''

        '''
        # if(cancer != None and len(cancer) < 1): raise ValueError(f'cancer less than 1.')
        self.cancer = cancer
        self.sensitive = sensitive
        self.fold = fold
        self.seed = seed
        self.intDiagnosticSlide = 0
        self.intTumor = 0 
        self.strClinicalInformationPath = './survival_clinical_information/'
        self.strEmbeddingPath = './AI_fairness/' # path to embeddings
        self.sort = False
        self.dfDistribution = None
        self.dfRemoveDupDistribution = None
        self.dictInformation = {}
        if self.cancer != None and self.sensitive != None and self.fold != None:
            self.dfClinicalInformation = self.fClinicalInformation()
        else:
            self.dfClinicalInformation = None

    def fClinicalInformation(self):
        df = pd.DataFrame({})
        for c in self.cancer:
            part = pd.read_pickle(glob.glob(f'{self.strClinicalInformationPath}/{c}_clinical_information.pkl')[0])
            df = pd.concat([df, part], ignore_index = True)
        return df
    
    def fReduceDataFrame(self, df):
        dfClinicalInformation = df.copy()
        mask = (df['days_to_death'] == '\'--') & (df['days_to_last_follow_up'] == '\'--')
        dfClinicalInformation = dfClinicalInformation[~mask].reset_index(drop = True)
        dfClinicalInformation['event'] = dfClinicalInformation['days_to_death'].apply(lambda x: 1 if x != '\'--' else 0)  # 1: death 0: alive
        # print(dfClinicalInformation['event'].value_counts())
        mask2 = dfClinicalInformation['days_to_death'] != '\'--'
        dfClinicalInformation.loc[mask2, 'T'] = dfClinicalInformation.loc[mask2, 'days_to_death']
        dfClinicalInformation.loc[~mask2, 'T'] = dfClinicalInformation.loc[~mask2, 'days_to_last_follow_up']

        stages = ['Stage I', 'Stage IA', 'Stage IB', 'Stage II', 'Stage IIA', 'Stage IIB', 'Stage IIC']
        mask3 = dfClinicalInformation['ajcc_pathologic_stage'].isin(stages)
        dfClinicalInformation = dfClinicalInformation.loc[mask3]

        dfClinicalInformation = dfClinicalInformation[['case_submitter_id', list(self.sensitive.keys())[0], 'T', 'event', 'ajcc_pathologic_stage']]

        dfClinicalInformation.columns = ['case_submitter_id', 'sensitive', 'T', 'event', 'stage']
        return dfClinicalInformation
    
    def fTransLabel(self, df):
        pass

    def fTransSensitive(self, df):
        substrings = self.sensitive[list(self.sensitive.keys())[0]]
        df = df[[any(x in y for x in substrings) for y in df['sensitive'].tolist()]]
        return df
    
    def updateDataFrame(self):
        self.dfClinicalInformation = self.fClinicalInformation()

    def getDistribution(self):
        return self.dfDistribution
    def getRemoveDupDistribution(self):
        return self.dfRemoveDupDistribution
    def getInformation(self):
        return self.dictInformation
        
    def train_valid_test(self, split=1.0, lora=False):
        if self.dfClinicalInformation is None:
            self.updateDataFrame()
        dfClinicalInformation = self.dfClinicalInformation.copy()

        lsDownloadPath = glob.glob(f'{self.strEmbeddingPath}/*.pt')
        lsDownloadFoldID = [s.split('/')[-1][:-3] for s in lsDownloadPath]

        lsDownloadCaseSubmitterId = [s[:12] for s in lsDownloadFoldID]
        dfClinicalInformation = self.fReduceDataFrame(dfClinicalInformation.drop_duplicates(subset = 'case_submitter_id', ignore_index = True))
        dfDownload = pd.DataFrame({
            'case_submitter_id': lsDownloadCaseSubmitterId,
            'folder_id': lsDownloadFoldID
        })
        dfClinicalInformation = pd.merge(dfClinicalInformation, dfDownload, on = "case_submitter_id")
        
        # if(self.intDiagnosticSlide == 0):
        #     dfClinicalInformation = dfClinicalInformation[['DX' in s[20:22] for s in dfClinicalInformation['folder_id'].tolist()]].reset_index(drop = True)
        # elif(self.intDiagnosticSlide == 1):
        #     dfClinicalInformation = dfClinicalInformation[['DX' not in s[20:22] for s in dfClinicalInformation['folder_id'].tolist()]].reset_index(drop = True)

        le = LabelEncoder()
        dfClinicalInformation.event =  le.fit_transform(dfClinicalInformation.event.values)
        leLabel = le.classes_
        self.dictInformation['event'] = leLabel

        scaler = MinMaxScaler()
        dfClinicalInformation['T'] = scaler.fit_transform(dfClinicalInformation['T'].values.reshape(-1, 1))

        dfClinicalInformation = self.fTransSensitive(dfClinicalInformation).reset_index(drop = True)
        dfClinicalInformation.sensitive = le.fit_transform(dfClinicalInformation.sensitive.values)
        leSensitive = le.classes_
        self.dictInformation['sensitive'] = leSensitive

        stage_map = {
            'Stage I': 0,
            'Stage IA': 0,
            'Stage IB': 0,
            'Stage II': 1,
            'Stage IIA': 1,
            'Stage IIB': 1,
            'Stage IIC': 1
        }
        dfClinicalInformation['stage'] = dfClinicalInformation['stage'].map(stage_map)

        dfDummy = dfClinicalInformation.drop_duplicates(subset = 'case_submitter_id', ignore_index = True).copy()
        dfDummy['fold'] = (10*np.array(dfDummy['sensitive'].tolist())+np.array(dfDummy['event'])).tolist()
        dfDummy.fold = le.fit_transform(dfDummy.fold.values)
        # print(dfDummy.groupby(['fold', 'event', 'sensitive']).size())
        foldNum = [0 for _ in range(int(len(dfDummy.index)))]
        if self.fold == 1:
            train, valitest = train_test_split(dfDummy, train_size = 0.6, random_state = self.seed, shuffle = True, stratify = dfDummy['fold'].tolist())
            vali, test = train_test_split(valitest, test_size = 0.5, random_state = self.seed, shuffle = True, stratify = valitest['fold'].tolist())
            # split training data
            if split < 1.0:
                train, remainder = train_test_split(train, train_size = split, random_state = self.seed, shuffle = True, stratify = train['fold'].tolist())
                if lora:
                    # vali
                    bags = [list(remainder.index), list(vali.index), list(test.index)]
                    print("The length of training set for fine-tuning: ", len(train))
                else:
                    bags = [list(train.index), list(vali.index), list(test.index)]
                    print("The length of training set: ", len(remainder))
            else:
                bags = [list(train.index), list(vali.index), list(test.index)]

        elif self.fold == 2:
            ab, cd = train_test_split(dfDummy, train_size = 0.5, random_state = self.seed, shuffle = True, stratify = dfDummy['fold'].tolist())
            a, b = train_test_split(ab, test_size = 0.5, random_state = self.seed, shuffle = True, stratify = ab['fold'].tolist())
            c, d = train_test_split(cd, train_size = 0.5, random_state = self.seed, shuffle = True, stratify = cd['fold'].tolist())
            bags = [list(a.index), list(b.index), list(c.index), list(d.index)]

        for fid, indices in enumerate(bags):
            for idx in indices:
                foldNum[idx] = fid
        dfDummy['fold'] = foldNum
        self.dfRemoveDupDistribution = dfDummy.groupby(['fold', 'event', 'sensitive']).size()

        dfDummy = dfDummy[['case_submitter_id', 'fold']]
        dfClinicalInformation = pd.merge(dfClinicalInformation, dfDummy, on = "case_submitter_id")
        dfClinicalInformation['path'] = [f'{self.strEmbeddingPath}{p}.pt' for p in dfClinicalInformation['folder_id']]
        self.dfDistribution = dfClinicalInformation.groupby(['fold', 'event', 'sensitive']).size()
        return dfClinicalInformation

        
class CancerDataset(Dataset):
    def __init__(self, df, fold_idx, split_type='kfold', exp_idx=0):
        self.df = df
        self.fold_idx = fold_idx
        self.split_type = split_type
        self.exp_idx = exp_idx
        
    def __getitem__(self, idx):
        if self.split_type == 'kfold':
            if self.fold_idx == 0:
                row = self.df[self.df['fold'].isin([(4-self.exp_idx)%4, (4-self.exp_idx+1)%4])].reset_index(drop=True).loc[idx]
                sample = torch.load(row.path)
                group = (row.sensitive, row.event)
                return sample, len(sample), row.sensitive, row.event, row['T'], group, row.stage
            elif self.fold_idx == 1:
                row = self.df[self.df['fold'].isin([(4-self.exp_idx+2)%4])].reset_index(drop=True).loc[idx]
                sample = torch.load(row.path)
                group = (row.sensitive, row.event)
                return sample, len(sample), row.sensitive, row.event, row['T'], group, row.stage
            elif self.fold_idx == 2:
                row = self.df[self.df['fold'].isin([(4-self.exp_idx+3)%4])].reset_index(drop=True).loc[idx]
                sample = torch.load(row.path)
                group = (row.sensitive, row.event)
                return sample, len(sample), row.sensitive, row.event, row['T'], group, row.stage
                        
        elif self.split_type == 'vanilla':
            if self.fold_idx == 0:
                row = self.df[self.df['fold'].isin([0])].reset_index(drop=True).loc[idx]
                sample = torch.load(row.path)
                group = (row.sensitive)
                return sample, len(sample), row.sensitive, row.event, row['T'], group, row.stage
            elif self.fold_idx == 1:
                row = self.df[self.df['fold'].isin([1])].reset_index(drop=True).loc[idx]
                sample = torch.load(row.path)
                group = (row.sensitive)
                return sample, len(sample), row.sensitive, row.event, row['T'], group, row.stage
            elif self.fold_idx == 2:
                row = self.df[self.df['fold'].isin([2])].reset_index(drop=True).loc[idx]
                sample = torch.load(row.path)
                group = (row.sensitive)
                return sample, len(sample), row.sensitive, row.event, row['T'], group, row.stage
                
    def __len__(self):
        fold_counts = self.df['fold'].value_counts()
        if self.split_type == 'kfold':
            if self.exp_idx == 0:
                if self.fold_idx == 0:  
                    return fold_counts.loc[[0, 1]].sum()
                elif self.fold_idx == 1:
                    return fold_counts.get(2, 0)
                elif self.fold_idx == 2:
                    return fold_counts.get(3, 0)
            elif self.exp_idx == 1:
                if self.fold_idx == 0:  
                    return fold_counts.loc[[3, 0]].sum()
                elif self.fold_idx == 1:
                    return fold_counts.get(1, 0)
                elif self.fold_idx == 2:
                    return fold_counts.get(2, 0)
            elif self.exp_idx == 2:
                if self.fold_idx == 0:  
                    return fold_counts.loc[[2, 3]].sum()
                elif self.fold_idx == 1:
                    return fold_counts.get(0, 0)
                elif self.fold_idx == 2:
                    return fold_counts.get(1, 0)
            elif self.exp_idx == 3:  
                if self.fold_idx == 0:  
                    return fold_counts.loc[[1, 2]].sum()
                elif self.fold_idx == 1:
                    return fold_counts.get(3, 0)
                elif self.fold_idx == 2:
                    return fold_counts.get(0, 0)
            else:
                return None
        elif self.split_type == 'vanilla':
            if self.fold_idx == 0:
                return fold_counts.get(0, 0)
            elif self.fold_idx == 1:
                return fold_counts.get(1, 0)  
            elif self.fold_idx == 2:
                return fold_counts.get(2, 0)
            else:
                return None
            
    def get_groups(self):
        targets = set(self.df['event'].values)
        sensitives = set(self.df['sensitive'].values)
        return [s for s in sensitives]

def get_datasets(df, split_type, exp_idx, reweight=False):
    if split_type == 'kfold':
        train_ds = CancerDataset(df, 0, split_type=split_type, exp_idx=exp_idx)
        if reweight:
            train_ds = ReweightDataset(train_ds, df, 0, split_type=split_type, exp_idx=exp_idx)
        val_ds = CancerDataset(df, 1, split_type=split_type, exp_idx=exp_idx)  
        test_ds = CancerDataset(df, 2, split_type=split_type, exp_idx=exp_idx)
    elif split_type == 'vanilla':
        train_ds = CancerDataset(df, 0, split_type=split_type)
        if reweight:
            train_ds = ReweightDataset(train_ds, df, 0, split_type=split_type)
        val_ds = CancerDataset(df, 1, split_type=split_type)
        test_ds = CancerDataset(df, 2, split_type=split_type)  
        
    return train_ds, val_ds, test_ds

    
def collate_fn(batch):
    samples, lengths, sensitives, events, times, groups, stage = zip(*batch)
    max_len = max(lengths)
    padded_slides = []
    for i in range(0, len(samples)):
        pad = (0,0,0, max_len-lengths[i])
        padded_slide = torch.nn.functional.pad(samples[i], pad)
        padded_slides.append(padded_slide)
    
    padded_slides = torch.stack(padded_slides)

    return padded_slides, lengths, torch.tensor(sensitives), torch.tensor(events), torch.tensor(times), groups, stage

class BalancedSampler(Sampler):
    def __init__(self, data_source, batch_size, resample=False, group_nums=None):
        self.data_source = data_source
        self.group_indices = {group: [] for group in data_source.get_groups()}
        for i in range(len(data_source)):
            group = data_source[i][2]
            self.group_indices[group].append(i)
        self.total_size = sum(len(group) for group in self.group_indices.values())
        self.batch_size = batch_size
        self.batch_num = self.total_size // self.batch_size
        self.resample = resample
        self.group_nums = group_nums
    
    def __iter__(self):
        batch_indices = []
        if self.resample == False:
            for i in range(self.batch_num):
                indices = []
                for group in self.group_indices:
                    indices.append(random.choice(self.group_indices[group]))
                while len(indices) < self.batch_size:
                    group = random.choice(list(self.group_indices.keys()))
                    indices.append(random.choice(self.group_indices[group]))
                batch_indices.append(indices)
        else:
            for i in range(self.batch_num):
                indices = []
                for group in self.group_nums:
                    indices += random.choices(self.group_indices[group], k=self.group_nums[group])
                batch_indices.append(indices)
        return iter(batch_indices)
            

    def __len__(self):
        return self.total_size // self.batch_size

class ReweightDataset(Dataset):
    def __init__(self, dataset, df, fold_idx, split_type='kfold', exp_idx=0):
        self.df = df
        self.fold_idx = fold_idx
        self.split_type = split_type
        self.exp_idx = exp_idx
        self.original_data = dataset
        self.reweight_data = self.reweight()

    def __getitem__(self, idx):
        return self.reweight_data[idx]

    def __len__(self):
        return len(self.reweight_data)

    def reweight(self):
        # get the number of samples in each group
        group_nums = {}
        samples = {}
        for i in range(len(self.original_data)):
            group = self.original_data[i][2]
            group_nums[group] = group_nums.get(group, 0) + 1
            samples[group] = samples.get(group, []) + [self.original_data[i]]
        min_count = min(group_nums.values())
        # reduce the number of samples in each group to the minimum count
        balanced = []
        for group, count in group_nums.items():
            while len(self.get_group_data(balanced, group)) < min_count:
                group_data = random.sample(samples[group], min_count)
                balanced.extend(group_data)
        return balanced
    
    def get_groups(self):
        targets = set(self.df['event'].values)
        sensitives = set(self.df['sensitive'].values)
        return [(s) for s in sensitives]
    
    def get_group_data(self, dataset, group):
        group_data = []
        for sample in dataset:
            if sample[2] == group:
                group_data.append(sample)
        return group_data