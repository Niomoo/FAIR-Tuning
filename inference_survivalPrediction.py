import os
import sys
import ast
import argparse
from sklearn.preprocessing import MinMaxScaler
import torch
import glob
from torch.utils.data import DataLoader
from tqdm import tqdm
from network_survivalPrediction import WeibullModel
import numpy as np
import pandas as pd
from util import replace_linear, SurvivalMetrics, FairnessMetricsMultiClass, Find_Optimal_Cutoff
import numpy as np
import loralib as lora
from pathlib import Path
from dataset_survivalPrediction import generateDataSet, get_datasets
from fairmetric import *
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

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
        "--lora", 
        action='store_true', 
        help="For LoRA finetuning."
    )
    parser.add_argument(
        "--reweight", 
        action='store_true', 
        help="For Last-layer finetuning."
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
    # Dataset preparation
    data = generateDataSet(cancer = args.cancer, sensitive = eval(args.fair_attr), fold = args.partition, seed = args.seed)
    df = data.train_valid_test()
    num_classes = len(df["event"].unique())
    auroc = 0.

    predicted_survival_times = []
    true_survival_times = []
    events = []
    senAttrs = []
    stages = []

    if args.partition == 1:
        _, _, test_ds = get_datasets(df, "vanilla", None)
        test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, pin_memory=True, pin_memory_device=args.device)

        cancer_folder = "_".join(args.cancer)
        model_names = os.listdir(args.model_path + f"{cancer_folder}_{args.partition}/")
        subfolders = [folder for folder in model_names if os.path.isdir(os.path.join(args.model_path + f"{cancer_folder}_{args.partition}/", folder))]
        model_indexes = [int(name.split('-')[0]) for name in subfolders]
        max_index = max(model_indexes)

        if args.lora:
            model = WeibullModel(ft=True)
            weight_path = glob.glob(args.model_path + f"{cancer_folder}_{args.partition}/{max_index}-*/model.pt")[0]
            model.load_state_dict(torch.load(weight_path), strict=False)
            lora_names = os.listdir(args.model_path + f"{cancer_folder}_{args.partition}_lora/")
            subfolders = [folder for folder in lora_names if os.path.isdir(os.path.join(args.model_path + f"{cancer_folder}_{args.partition}_lora/", folder))]
            lora_indexes = [int(name.split('-')[0]) for name in subfolders]
            if args.weight_path == "":
                max_lora_index = max(lora_indexes)
            else:
                max_lora_index = int(args.weight_path)
            lora_path = glob.glob(args.model_path + f"{cancer_folder}_{args.partition}_lora/{max_lora_index}-*_lora/lora.pt")[0]
            model.load_state_dict(torch.load(lora_path), strict=False)
            result_path = Path(lora_path).parent / "result.csv"
            fig_path = Path(lora_path).parent / "survival_curve.png"
            fig_path2 = Path(lora_path).parent / "survival_curve_stage.png"
            fig_path3 = Path(lora_path).parent / "survival_curve_black.png"
            # fig_path4 = Path(lora_path).parent / "boxplot.png"

        elif args.reweight:
            model = WeibullModel(ft=True)
            cancer_folder = "_".join(args.cancer)
            reweight_names = os.listdir(args.model_path + f"{cancer_folder}_{args.partition}_reweight/")
            subfolders = [folder for folder in reweight_names if os.path.isdir(os.path.join(args.model_path + f"{cancer_folder}_{args.partition}_reweight/", folder))]
            reweight_indexes = [int(name.split('-')[0]) for name in subfolders]
            if args.weight_path == "":
                max_reweight_index = max(reweight_indexes)
            else:
                max_reweight_index = int(args.weight_path)
            reweight_path = glob.glob(args.model_path + f"{cancer_folder}_{args.partition}_reweight/{max_reweight_index}-*_reweight/model.pt")[0]
            model.load_state_dict(torch.load(reweight_path), strict=False)
            result_path = Path(reweight_path).parent / "result.csv"
            fig_path = Path(reweight_path).parent / "survival_curve.png"
            fig_path2 = Path(reweight_path).parent / "survival_curve_stage.png"
            fig_path3 = Path(reweight_path).parent / "survival_curve_black.png"
            # fig_path4 = Path(reweight_path).parent / "boxplot.png"

        elif not args.lora and not args.reweight:
            model = WeibullModel(ft=False)
            if args.weight_path != "":
                max_index = int(args.weight_path)
            weight_path = glob.glob(args.model_path + f"{cancer_folder}_{args.partition}/{max_index}-*/model.pt")[0]
            model.load_state_dict(torch.load(weight_path), strict=False)
            result_path = Path(weight_path).parent / "result.csv"
            fig_path = Path(weight_path).parent / "survival_curve.png"
            fig_path2 = Path(weight_path).parent / "survival_curve_stage.png"
            fig_path3 = Path(weight_path).parent / "survival_curve_black.png"
            # fig_path4 = Path(weight_path).parent / "boxplot.png"

        group_counts = test_ds.get_groups()
        groups = {}
        for i in range(len(group_counts)):
            groups[group_counts[i]] = 0
        for i in range(len(test_ds)):
            group = test_ds[i][5]
            groups[group] += 1
        print(args.cancer)
        print(groups)

        model.eval().to(args.device)

        test_pbar = tqdm(enumerate(test_dl), colour="blue", total=len(test_dl))
        with torch.no_grad():
            for idx, data in test_pbar:
                wsi_embeddings, lengths, sensitive, event, time, group, stage = data
                test_shape_scale = model(wsi_embeddings.to(args.device), sensitive.to(args.device))
                test_shape, test_scale = test_shape_scale[:, 0], test_shape_scale[:, 1]
                # predicted_survival_time = weibull_min.ppf(0.5, scale=test_scale.cpu(), c=test_shape.cpu())
                predicted_survival_time = test_scale * torch.exp(torch.log(time.to(args.device)+ 1e-8) / test_shape)

                predicted_survival_times.append(predicted_survival_time.detach().cpu().numpy())
                # predictions.append(torch.argmax(test_shape.detach().cpu(), dim=1).numpy())
                true_survival_times.append(time.detach().cpu().numpy())
                events.append(event.detach().cpu().numpy())
                senAttrs.append(sensitive.detach().cpu().numpy())
                stages.append(stage.detach().cpu().numpy())
    
    elif args.partition == 2:
        for curr_fold in range(4):
            _, _, test_ds = get_datasets(df, "kfold", curr_fold)
            test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, pin_memory=True, pin_memory_device=args.device)
            cancer_folder = "_".join(args.cancer)
            model_names = os.listdir(args.model_path + f"{cancer_folder}_{args.partition}/")
            subfolders = [folder for folder in model_names if os.path.isdir(os.path.join(args.model_path + f"{cancer_folder}_{args.partition}/", folder))]
            model_indexes = [int(name.split('-')[0]) for name in subfolders]
            max_index = max(model_indexes)

            if args.lora:
                model = WeibullModel(ft=True)
                cancer_folder = "_".join(args.cancer)
                weight_path = glob.glob(args.model_path + f"{cancer_folder}_{args.partition}/{max_index}*_{curr_fold}/model.pt")[0]
                model.load_state_dict(torch.load(weight_path), strict=False)
                lora_names = os.listdir(args.model_path + f"{cancer_folder}_{args.partition}_lora/")
                subfolders = [folder for folder in lora_names if os.path.isdir(os.path.join(args.model_path + f"{cancer_folder}_{args.partition}_lora/", folder))]
                lora_indexes = [int(name.split('-')[0]) for name in subfolders]
                if args.weight_path == "":
                    max_lora_index = max(lora_indexes)
                else:
                    max_lora_index = int(args.weight_path)
                lora_path = glob.glob(args.model_path + f"{cancer_folder}_{args.partition}_lora/{max_lora_index}*_{curr_fold}_lora/lora.pt")[0]
                model.load_state_dict(torch.load(lora_path), strict=False)
                result_path = Path(lora_path).parent.parent / f"{max_lora_index}-result.csv"
                fig_path = Path(lora_path).parent.parent / f"{max_lora_index}-survival_curve.png"
                fig_path2 = Path(lora_path).parent / f"{max_lora_index}-survival_curve_stage.png"
                fig_path3 = Path(lora_path).parent / f"{max_lora_index}-survival_curve_black.png"
                fig_path4 = Path(lora_path).parent / f"{max_lora_index}-boxplot.png"

            elif args.reweight:
                model = WeibullModel(ft=True)
                cancer_folder = "_".join(args.cancer)
                reweight_names = os.listdir(args.model_path + f"{cancer_folder}_{args.partition}_reweight/")
                subfolders = [folder for folder in reweight_names if os.path.isdir(os.path.join(args.model_path + f"{cancer_folder}_{args.partition}_reweight/", folder))]
                reweight_indexes = [int(name.split('-')[0]) for name in subfolders]
                if args.weight_path == "":
                    max_reweight_index = max(reweight_indexes)
                else:
                    max_reweight_index = int(args.weight_path)
                reweight_path = glob.glob(args.model_path + f"{cancer_folder}_{args.partition}_reweight/{max_reweight_index}-*_reweight/model.pt")[0]
                model.load_state_dict(torch.load(reweight_path), strict=False)
                result_path = Path(reweight_path).parent.parent / f"{max_reweight_index}-result.csv"
                fig_path = Path(reweight_path).parent.parent / f"{max_reweight_index}-survival_curve.png"
                fig_path2 = Path(reweight_path).parent.parent / f"{max_reweight_index}-survival_curve_stage.png"
                fig_path3 = Path(reweight_path).parent.parent / f"{max_reweight_index}-survival_curve_black.png"
                fig_path4 = Path(reweight_path).parent.parent / f"{max_reweight_index}-boxplot.png"

            elif not args.lora and not args.reweight:
                model = WeibullModel(ft=False)
                if args.weight_path != "":
                    max_index = int(args.weight_path)
                weight_path = glob.glob(args.model_path + f"{cancer_folder}_{args.partition}/{max_index}*_{curr_fold}/model.pt")[0]
                model.load_state_dict(torch.load(weight_path), strict=False)
                result_path = Path(weight_path).parent.parent / f"{max_index}-result.csv"
                fig_path = Path(weight_path).parent.parent / f"{max_index}-survival_curve.png"
                fig_path2 = Path(weight_path).parent.parent / f"{max_index}-survival_curve_stage.png"
                fig_path3 = Path(weight_path).parent.parent / f"{max_index}-survival_curve_black.png"
                fig_path4 = Path(weight_path).parent.parent / f"{max_index}-boxplot.png"

            model.eval().to(args.device)

            test_pbar = tqdm(enumerate(test_dl), colour="blue", total=len(test_dl))

            with torch.no_grad():
                for _, data in test_pbar:
                    wsi_embeddings, lengths, sensitive, event, time, group, stage = data
                    test_shape_scale = model(wsi_embeddings.to(args.device), sensitive.to(args.device))
                    test_shape, test_scale = test_shape_scale[:, 0], test_shape_scale[:, 1]

                    predicted_survival_time = test_scale * torch.exp(torch.log(time.to(args.device)+ 1e-8) / test_shape)

                    predicted_survival_times.append(predicted_survival_time.detach().cpu().numpy())
                    true_survival_times.append(time.detach().cpu().numpy())
                    events.append(event.detach().cpu().numpy())
                    senAttrs.append(sensitive.detach().cpu().numpy())
                    stages.append(stage.detach().cpu().numpy())

    # npLabels = np.concatenate(labels)
    # npPredictions = np.concatenate(predictions)
    npPredictedSurvivalTime = np.concatenate(predicted_survival_times)
    npTrueSurvivalTime = np.concatenate(true_survival_times)
    npEvents = np.concatenate(events)
    npSenAttrs = np.concatenate(senAttrs)
    npStages = np.concatenate(stages)
    nan_mask = np.isnan(npPredictedSurvivalTime) | np.isnan(npTrueSurvivalTime)
    npPredictedSurvivalTime = npPredictedSurvivalTime[~nan_mask]
    npTrueSurvivalTime = npTrueSurvivalTime[~nan_mask]
    npEvents = npEvents[~nan_mask]
    npSenAttrs = npSenAttrs[~nan_mask]
    npStages = npStages[~nan_mask]
    scaler = MinMaxScaler()
    # max_time = np.max(npTrueSurvivalTime)
    # npPredictedSurvivalTime = np.where(npPredictedSurvivalTime == np.Inf, max_time, npPredictedSurvivalTime)
    med_days = np.median(npPredictedSurvivalTime)

    if med_days == np.Inf:
        max_time = np.max(npTrueSurvivalTime)
        npPredictedSurvivalTime = np.where(npPredictedSurvivalTime == np.Inf, max_time, npPredictedSurvivalTime)
        med_days = np.median(npPredictedSurvivalTime)
    T_big = []
    T_small = []
    E_big = []
    E_small = []
    T_stage1 = []
    T_stage2 = []
    E_stage1 = []
    E_stage2 = []
    for i in range(0, len(npPredictedSurvivalTime)):
        if npPredictedSurvivalTime[i] > med_days:
            T_big.append(npTrueSurvivalTime[i])
            E_big.append(npEvents[i])
        else:
            T_small.append(npTrueSurvivalTime[i])
            E_small.append(npEvents[i])
        if npStages[i] == 0:
            T_stage1.append(npTrueSurvivalTime[i])
            E_stage1.append(npEvents[i])
        elif npStages[i] == 1:
            T_stage2.append(npTrueSurvivalTime[i])
            E_stage2.append(npEvents[i])
    kmf = KaplanMeierFitter()
    plt.figure(1)
    fig, ax = plt.subplots(1)
    kmf.fit(T_big, event_observed=E_big, label='Longer-term survivors')
    kmf.plot(show_censors=True, ax=ax)
    kmf.fit(T_small, event_observed=E_small, label='Shorter-term survivors')
    kmf.plot(show_censors=True, ax=ax)
    plt.savefig(fig_path)
    print(f"Save survival curve to:{fig_path}")

    kmf2 = KaplanMeierFitter()
    plt.figure(2)
    fig2, ax2 = plt.subplots(1)
    kmf2.fit(T_stage2, event_observed=E_stage2, label='Stage II')
    kmf2.plot(show_censors=True, ax=ax2)
    kmf2.fit(T_stage1, event_observed=E_stage1, label='Stage I')
    kmf2.plot(show_censors=True, ax=ax2)
    plt.savefig(fig_path2)
    print(f"Save stage survival curve to:{fig_path2}")
    stage_logrank = logrank_test(T_stage2, T_stage1, event_observed_A=E_stage2, event_observed_B=E_stage1)
    print("stage_logrank", stage_logrank.p_value)
    # fpr, tpr, auroc, threshold = Find_Optimal_Cutoff(npLabels, npPredictions)
    # predictions = torch.ge(torch.tensor(npPredictions), threshold).int()
    results = SurvivalMetrics(npPredictedSurvivalTime, npTrueSurvivalTime, npEvents, npSenAttrs)

    # kmf3 = KaplanMeierFitter()
    # plt.figure(3)
    # fig3, ax3 = plt.subplots(1)
    # kmf3.fit(npTrueSurvivalTime[npSenAttrs == 0], event_observed=npEvents[npSenAttrs == 0], label='Black')
    # kmf3.plot(show_censors=True, ax=ax3)
    # kmf3.fit(npTrueSurvivalTime[npSenAttrs == 1], event_observed=npEvents[npSenAttrs == 1], label='White')
    # kmf3.plot(show_censors=True, ax=ax3)
    # plt.savefig(fig_path3)
    # print(f"Save survival curve to:{fig_path3}")
    np_Predicted_survival_times = npPredictedSurvivalTime[npSenAttrs == 0]
    np_True_survival_times = npTrueSurvivalTime[npSenAttrs == 0]
    np_Events = npEvents[npSenAttrs == 0]
    med_days_black = np.median(np_Predicted_survival_times)
    T_big_black = []
    T_small_black = []
    E_big_black = []
    E_small_black = []
    for i in range(0,len(np_Predicted_survival_times)):
        if np_Predicted_survival_times[i] > med_days_black:
            T_big_black.append(np_True_survival_times[i])
            E_big_black.append(np_Events[i])
        else:
            T_small_black.append(np_True_survival_times[i])
            E_small_black.append(np_Events[i])
    kmf3 = KaplanMeierFitter()
    plt.figure(3)
    fig3, ax3 = plt.subplots(1)
    kmf3.fit(T_big_black, event_observed=E_big_black, label="Longer-term survivors")
    kmf3.plot(show_censors=True, ax=ax3)
    kmf3.fit(T_small_black, event_observed=E_small_black, label="Shorter-term survivors")
    kmf3.plot(show_censors=True, ax=ax3)
    plt.savefig(fig_path3)

    # survival_times = [npTrueSurvivalTime[npSenAttrs == 0], npTrueSurvivalTime[npSenAttrs == 1]]
    # plt.figure(4)
    # fig4, ax4 = plt.subplots(1)
    # plt.boxplot(survival_times, labels=['Black', 'White'])
    # plt.xlabel('Race')
    # plt.ylabel('Survival Time')
    # plt.title('Comparison of Survival Time between Black and White Data')
    # plt.savefig(fig_path4)

    temp = {
        # "AUROC": auroc,
        # "Threshold": threshold,
    }
    results = {**results}
    df = pd.DataFrame(results, index=[0])
    df.T.to_csv(result_path)
    print(f"Save results to:{result_path}")  

    # fmtc = Metrics(predictions = survivorsGroup, labels = npEvents, sensitives = npSenAttrs, projectName = "proposed", verbose = True)
    # markdown = fmtc.getResults(markdownFormat=True)
    # if auroc != 0: markdown += f"{auroc:.4f}|"
    table = showMetrics(results)
    print(table)

    del model, results

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("GPU")
    else:
        print("CPU")
    args = parse_args()
    main(args)
