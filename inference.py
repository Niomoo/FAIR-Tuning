import os
import sys
import ast
import argparse
import torch
import glob
from torch.utils.data import DataLoader
from tqdm import tqdm
from network import ClfNet
import numpy as np
import pandas as pd
from util import replace_linear, FairnessMetrics, FairnessMetricsMultiClass, Find_Optimal_Cutoff
import numpy as np
import loralib as lora
from pathlib import Path
from dataset import generateDataSet, get_datasets
from fairmetric import *

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
    num_classes = len(df["label"].unique())
    auroc = 0.

    if args.partition == 1:
        _, _, test_ds = get_datasets(df, "vanilla", None)
        test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, pin_memory=True, pin_memory_device=args.device)

        cancer_folder = "_".join(args.cancer)
        model_names = os.listdir(args.model_path + f"{cancer_folder}_{args.partition}/")
        subfolders = [folder for folder in model_names if os.path.isdir(os.path.join(args.model_path + f"{cancer_folder}_{args.partition}/", folder))]
        model_indexes = [int(name.split('-')[0]) for name in subfolders]
        max_index = max(model_indexes)
        
        if args.lora:
            model = ClfNet(classes=num_classes, ft=True)
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
        
        elif args.reweight:
            model = ClfNet(classes=num_classes, ft=True)
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


        elif not args.lora and not args.reweight:
            model = ClfNet(classes=num_classes, ft=False)
            if args.weight_path != "":
                max_index = int(args.weight_path)
            weight_path = glob.glob(args.model_path + f"{cancer_folder}_{args.partition}/{max_index}-*/model.pt")[0]
            model.load_state_dict(torch.load(weight_path), strict=False)
            result_path = Path(weight_path).parent / "result.csv"

        model.eval().to(args.device)
        
        test_pbar = tqdm(enumerate(test_dl), colour="blue", total=len(test_dl))
        predictions = []
        labels = []
        senAttrs = []
        
        with torch.no_grad():
            for _, data in test_pbar:
                wsi_embeddings, lengths, sensitive, label, group = data
                test_cancer_pred = model(wsi_embeddings.to(args.device), sensitive.to(args.device))

                predictions.append(torch.argmax(test_cancer_pred.detach().cpu(), dim=1).numpy())
                labels.append(label.detach().cpu().numpy())
                senAttrs.append(sensitive.detach().cpu().numpy())
        
        if num_classes > 2:
            results = FairnessMetricsMultiClass(np.array(predictions), np.array(labels), senAttrs)
            pd.DataFrame(results).T.to_csv(result_path)
            print(f"Save results to:{result_path}")
                
        elif num_classes == 2:
            fpr, tpr, auroc, threshold = Find_Optimal_Cutoff(np.array(labels), np.array(predictions))
            predictions = torch.ge(torch.tensor(predictions), threshold).int()
            results = FairnessMetrics(np.array(predictions), np.array(labels), senAttrs)
            temp = {"AUROC": auroc,
                    "Threshold": threshold}
            results = {**temp, **results}
            pd.DataFrame(results).T.to_csv(result_path)
            print(f"Save results to:{result_path}")  

        fmtc = Metrics(predictions = np.array(predictions), labels = np.array(labels), sensitives = senAttrs, projectName = "proposed", verbose = True)
        markdown = fmtc.getResults(markdownFormat=True) 
        if auroc != 0: markdown += f"{auroc:.4f}|"
        print(markdown)

        del model, results

    elif args.partition == 2:
        predictions = []
        labels = []
        senAttrs = []
        for curr_fold in range(4):
            _, _, test_ds = get_datasets(df, "kfold", curr_fold)
            test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, pin_memory=True, pin_memory_device=args.device)
            cancer_folder = "_".join(args.cancer)
            model_names = os.listdir(args.model_path + f"{cancer_folder}_{args.partition}/")
            subfolders = [folder for folder in model_names if os.path.isdir(os.path.join(args.model_path + f"{cancer_folder}_{args.partition}/", folder))]
            model_indexes = [int(name.split('-')[0]) for name in subfolders]
            max_index = max(model_indexes)

            if args.lora:
                model = ClfNet(classes=num_classes, ft=True)
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

            elif args.reweight:
                model = ClfNet(classes=num_classes, ft=True)
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
            
            elif not args.lora and not args.reweight:
                model = ClfNet(classes=num_classes, ft=False)
                if args.weight_path != "":
                    max_index = int(args.weight_path)
                weight_path = glob.glob(args.model_path + f"{cancer_folder}_{args.partition}/{max_index}*_{curr_fold}/model.pt")[0]
                model.load_state_dict(torch.load(weight_path), strict=False)
                result_path = Path(weight_path).parent.parent / f"{max_index}-result.csv"

            model.eval().to(args.device)

            test_pbar = tqdm(enumerate(test_dl), colour="blue", total=len(test_dl))

            with torch.no_grad():
                for _, data in test_pbar:
                    wsi_embeddings, lengths, sensitive, label, group = data
                    test_cancer_pred = model(wsi_embeddings.to(args.device), sensitive.to(args.device))

                    predictions.append(torch.argmax(test_cancer_pred.detach().cpu(), dim=1).numpy())
                    labels.append(label.detach().cpu().numpy())
                    senAttrs.append(sensitive.detach().cpu().numpy())
            
        if num_classes > 2:
            results = FairnessMetricsMultiClass(np.array(predictions), np.array(labels), senAttrs)
            pd.DataFrame(results).T.to_csv(result_path)
            print(f"Save results to:{result_path}")

        elif num_classes == 2:
            fpr, tpr, auroc, threshold = Find_Optimal_Cutoff(np.array(labels), np.array(predictions))
            predictions = torch.ge(torch.tensor(predictions), threshold).int()
            results = FairnessMetrics(np.array(predictions), np.array(labels), senAttrs)
            temp = {"AUROC": auroc,
                    "Threshold": threshold}
            results = {**temp, **results}
            pd.DataFrame(results).T.to_csv(result_path)
            print(f"Save results to:{result_path}")
        
        fmtc = Metrics(predictions = np.array(predictions), labels = np.array(labels), sensitives = senAttrs, projectName = "proposed", verbose = True)
        markdown = fmtc.getResults(markdownFormat=True) 
        if auroc != 0: markdown += f"{auroc:.4f}|"
        print(markdown)

        del model, results

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("GPU")
    else:
        print("CPU")
    args = parse_args()
    main(args)
