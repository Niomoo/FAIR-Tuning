import os
import sys
import ast
import argparse
import torch
import wandb
import glob
from torch.utils.data import DataLoader
from tqdm import tqdm
from network import ClfNet, WeibullModel
from util import *
import numpy as np
from pathlib import Path
from dataset import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler

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
        help="Downstream task: '1:cancer classification, 2:tumor detection, 3:survival prediction",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/",
        help="Path to save weights.",
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        default="",
        help="Path to stage 1 pretrained weights.",
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=100, 
        help="Epochs for training."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1, 
        help="Batch size for sampling images."
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-4, 
        help="Learning rate for training."
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
        help="Sample a balanced dataset."
    )
    parser.add_argument(
        "--fair_lambda",
        type=float,
        default=0.5,
        help="Parameter for fairness loss."
    )
    parser.add_argument(
        "--constraint",
        type=str,
        default="",
        help="Fairness constraint for fine-tuning"
    )
    parser.add_argument(
        "--selection",
        type=str,
        default="EOdd",
        help="Model selection strategy for fine-tuning."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cpu or cuda",
    )
    parser.add_argument(
        "--acc_grad",
        type=int,
        default=1,
        help="Accumulation gradient."
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate for the model"
    )
    parser.add_argument(
        "--scheduler_gamma",
        type=float,
        default=1,
        help="Gamma for scheduler"
    )
    parser.add_argument(
        "--scheduler_step",
        type=float,
        default=10,
        help="Steps for scheduler"
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=1.0,
        help="Split ratio for training set"
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args


def main(args):
    # Wandb settings
    cancer_folder = "_".join(args.cancer)
    if not os.path.exists(args.model_path + f"{cancer_folder}_{args.partition}/"):
        os.makedirs(args.model_path + f"{cancer_folder}_{args.partition}/")
    model_names = os.listdir(args.model_path + f"{cancer_folder}_{args.partition}/")
    subfolders = [folder for folder in model_names if os.path.isdir(os.path.join(args.model_path + f"{cancer_folder}_{args.partition}/", folder))]
    if not subfolders:
        max_index = 0
    else:
        model_indexes = [int(name.split('-')[0]) for name in subfolders]
        max_index = max(model_indexes)       

    wandb_id = wandb.util.generate_id()
    if args.partition == 1:
        name = wandb_id
        if not args.reweight:                             # 1 w/o reweight
            max_index += 1                                     
    elif args.partition == 2:
        name = wandb_id + f"_fold_{args.curr_fold}"
        if not args.reweight:                             # 2 w/o reweight
            if args.curr_fold == 0:
                max_index += 1
    group = "_".join(args.cancer)+ "_" + str(args.partition)

    if args.reweight:
        name += "_reweight"
        group += "_reweight"
        if not os.path.exists(args.model_path + f"{cancer_folder}_{args.partition}_reweight/"):
            os.makedirs(args.model_path + f"{cancer_folder}_{args.partition}_reweight/")
        reweight_names = os.listdir(args.model_path + f"{cancer_folder}_{args.partition}_reweight/")
        subfolders = [folder for folder in reweight_names if os.path.isdir(os.path.join(args.model_path + f"{cancer_folder}_{args.partition}_reweight/", folder))]
        if not subfolders:
            max_reweight_index = 1
        else:
            reweight_indexes = [int(name.split('-')[0]) for name in subfolders]
            max_reweight_index = max(reweight_indexes)
            if args.partition == 1:                                 # 1 reweight
                max_reweight_index += 1
            else:                                                   # 2 reweight
                if args.curr_fold == 0:
                    max_reweight_index += 1 

        name = str(max_reweight_index) + "-base-" + name
    else:
        name = str(max_index) + "-base-" + name

    wandb.init(project='FAIR-Tuning', 
               entity='jennyliu-lyh', 
               config=args, 
               name=name, 
               id=wandb_id,
               group=group)

    # Dataset preparation
    data = generateDataSet(cancer = args.cancer, sensitive = eval(args.fair_attr), fold = args.partition, task = args.task, seed = args.seed)
    df = data.train_valid_test(args.split_ratio)

    if args.task == 3:
        num_classes = 2
    else:
        num_classes = len(df["label"].unique())

    if args.partition == 1:
        train_ds, val_ds, test_ds = get_datasets(df, args.task, "vanilla", None, args.reweight)
    elif args.partition == 2:
        train_ds, val_ds, test_ds = get_datasets(df, args.task, "kfold", args.curr_fold)

    train_dl = DataLoader(train_ds, collate_fn=collate_fn, batch_size=args.batch_size, pin_memory=False)
    val_dl = DataLoader(val_ds, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=False, pin_memory=False)
    test_dl = DataLoader(test_ds, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=False, pin_memory=False)

    if not args.reweight:
        if args.task == 1 or args.task == 2:
            model = ClfNet(classes=num_classes)
        elif args.task == 3:
            model = WeibullModel()
        model.train()

    if args.reweight:
        if args.task == 1 or args.task == 2:
            model = ClfNet(classes=num_classes, ft=True)
        elif args.task == 3:
            model = WeibullModel(ft=True)
        cancer_folder = "_".join(args.cancer)
        if args.weight_path != "":
            weight_path = glob.glob(args.model_path + f"{cancer_folder}_{args.partition}/{args.weight_path}-*/model.pt")[0]
        else:
            if args.partition == 1:
                weight_path = glob.glob(args.model_path + f"{cancer_folder}_{args.partition}/{max_index}-*/model.pt")[0]
            elif args.partition == 2:
                weight_path = glob.glob(args.model_path + f"{cancer_folder}_{args.partition}/{max_index}-*_{args.curr_fold}/model.pt")[0]
        model.load_state_dict(torch.load(weight_path), strict=False)
        print(f"Weights path:{weight_path}")
        print("Loaded pretrained weights.")

    model = model.to(args.device)

    # Settings
    gradient_accumulation_steps = args.acc_grad
    parameters_to_update = []
    if args.reweight:
        for n, p in model.named_parameters():
            if n.startswith('fc_target.6'):
                p.requires_grad = True
                parameters_to_update.append(p)
            else:
                p.requires_grad = False
        print("Params to learn:" + str(len(parameters_to_update)))
        if args.task == 1 or args.task == 2:
            optimizer = torch.optim.Adam(parameters_to_update, lr=args.lr)
        elif args.task == 3:
            optimizer = torch.optim.RMSprop(parameters_to_update, lr=args.lr, weight_decay=1e-4)
    else:
        for n, p in model.named_parameters():
            p.requires_grad = True
            parameters_to_update.append(p)
        print("Params to learn:" + str(len(parameters_to_update)))
        if args.task == 1 or args.task == 2:
            optimizer = torch.optim.Adam(parameters_to_update, lr=args.lr)
        elif args.task == 3:
            optimizer = torch.optim.RMSprop(parameters_to_update, lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.scheduler) # constant lr
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    loss_fn = torch.nn.CrossEntropyLoss()
    fairness_loss_fn = FairnessLoss(args.fair_lambda)

    # Output folder
    model_save_path = args.model_path + f"{group}/{name}"
    os.makedirs(model_save_path, exist_ok=True)

    epoch_record = 0
    performance_record = 0.
    fairness_record = 9999.
    best_performance = 0.
    best_fairness = 9999.
    group_samples = {}
    
    model.train()

    # Training/evaluation process
    for epoch in range(args.epochs):
        train_loss = 0.
        fair_loss = 0.
        group_loss = 0.
        overall_loss = 0.
        total_train_loss = 0.
        total_fair_loss = 0.

        pbar = tqdm(enumerate(train_dl), colour="yellow", total=len(train_dl))

        # for sample in train_dl:
        #     emb, length, sensitive, label, groups = sample
        #     group_count = {}
        #     for batch_group in groups:
        #         if batch_group in group_count:
        #             group_count[batch_group] += 1
        #         else:
        #             group_count[batch_group] = 1
        #     print(group_count)

        for idx, data in pbar:

            if args.task == 3:
                wsi_embeddings, lengths, sensitive, event, time, group, stage = data
                shape_scale = model(wsi_embeddings.to(args.device), lengths)
                shape, scale = shape_scale[:, 0], shape_scale[:, 1]
                train_loss, group_of_loss = survival_loss_function(shape, scale, time.float().to(args.device), torch.nn.functional.one_hot(event, num_classes).float().to(args.device), lengths, group)
            else:
                wsi_embeddings, lengths, sensitive, label, group = data
                cancer_pred = model(wsi_embeddings.to(args.device), sensitive.to(args.device), lengths)
                train_loss, group_of_loss = loss_function(loss_fn, cancer_pred, torch.nn.functional.one_hot(label, num_classes).float().to(args.device), lengths, group)
            train_loss = train_loss / gradient_accumulation_steps

            if args.reweight:
                if args.task == 3:
                    x, y, a = (wsi_embeddings.to(args.device), event.to(args.device), sensitive.to(args.device))
                    if args.constraint == "":
                        fair_loss = 0
                    else:
                        predicted_survival_time = scale * torch.exp(torch.log(time.to(args.device) + 1e-8) / shape)

                else:
                    x, y, a = wsi_embeddings.to(args.device), label.to(args.device), sensitive.to(args.device)
                    log_softmax, softmax = torch.nn.functional.log_softmax(cancer_pred, dim=1), torch.nn.functional.softmax(cancer_pred, dim=1)
                    if args.constraint == '':
                        fair_loss = 0
                    elif args.constraint == 'MMF':
                        fair_loss = args.fair_lambda * mmf_constraint(loss_fn, log_softmax, y, a)
                    else:
                        if args.constraint == 'EO':
                            fpr, fnr = eo_constraint(softmax[:, 1], y, a)
                            loss_fairness = fpr + fnr
                        elif args.constraint == 'DI':
                            loss_fairness = di_constraint(softmax[:, 1], a)
                        elif args.constraint == 'DP':
                            loss_fairness = dp_constraint(softmax[:, 1], a)
                        elif args.constraint == 'AE':
                            loss_fairness = ae_constraint(loss_fn, log_softmax, y, a)
                        fair_loss = args.fair_lambda * loss_fairness
            #     fair_loss = fairness_loss_fn(cancer_pred, torch.nn.functional.one_hot(label, num_classes).float().to(args.device), lengths, group)

            total_loss = train_loss + fair_loss
            total_loss.backward()

            group_losses = sum(group_of_loss.values())

            if(idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if torch.isnan(total_loss):
                pass
            else:
                overall_loss += total_loss.detach().cpu().numpy()
            if torch.isnan(train_loss):
                pass
            else:
                total_train_loss += train_loss.detach().cpu().numpy()
            if torch.isnan(group_losses):
                pass
            else:
                group_loss = group_losses.detach().cpu().numpy()
            total_fair_loss += fair_loss

            avg_overall_loss = overall_loss / len(train_dl)
            avg_train_loss = total_train_loss / len(train_dl)
            avg_train_fair_loss = total_fair_loss / len(train_dl)
            avg_group_loss = group_loss / len(train_dl)

            pbar.set_description(
                f"Iter:{epoch+1:3}/{args.epochs:3} "
                f"Avg_loss:{avg_train_loss:.4f} "
                f"Fair_loss:{avg_train_fair_loss:.4f} "
                f"Group_loss:{avg_group_loss:.4f} ")
            pbar.update()         
        scheduler.step()

        model.eval()
        eval_pbar = tqdm(enumerate(val_dl), colour="blue", total=len(val_dl))
        eval_loss = 0.0
        eval_fair_loss = 0.0
        eval_group_loss = 0.0
        eval_overall_loss = 0.0
        total_eval_loss = 0.0
        total_eval_fair_loss = 0.0
        predicted_survival_time = 0.0
        predictions = []
        predicted_survival_times = []
        true_survival_times = []
        labels = []
        events = []
        senAttrs = []
        with torch.no_grad():
            for _, data in eval_pbar:

                if args.task == 3:
                    wsi_embeddings, lengths, sensitive, event, time, group, stage = data
                    eval_shape_scale = model(wsi_embeddings.to(args.device), lengths)
                    eval_shape, eval_scale = eval_shape_scale[:, 0], eval_shape_scale[:, 1]
                    eval_loss, eval_group_of_loss = survival_loss_function(eval_shape, eval_scale, time.float().to(args.device), torch.nn.functional.one_hot(event, num_classes) .float() .to(args.device), lengths, group)
                else:
                    wsi_embeddings, lengths, sensitive, label, group = data
                    eval_cancer_pred = model(wsi_embeddings.to(args.device), sensitive.to(args.device), lengths)
                    eval_loss, eval_group_of_loss = loss_function(loss_fn, eval_cancer_pred, torch.nn.functional.one_hot(label, num_classes).float().to(args.device), lengths, group)

                if args.reweight:
                    if args.task == 3:
                        if args.constraint == "":
                            eval_fair_loss = 0
                    else:
                        x, y, a = wsi_embeddings.to(args.device), label.to(args.device), sensitive.to(args.device)
                        log_softmax, softmax = torch.nn.functional.log_softmax(eval_cancer_pred, dim=1), torch.nn.functional.softmax(eval_cancer_pred, dim=1)
                        if args.constraint == '':
                            eval_fair_loss = 0
                        elif args.constraint == 'MMF':
                            eval_fair_loss = args.fair_lambda * mmf_constraint(loss_fn, log_softmax, y, a)
                        else:
                            if args.constraint == 'EO':
                                fpr, fnr = eo_constraint(softmax[:, 1], y, a)
                                eval_loss_fairness = fpr + fnr
                            elif args.constraint == 'DI':
                                eval_loss_fairness = di_constraint(softmax[:, 1], a)
                            elif args.constraint == 'DP':
                                eval_loss_fairness = dp_constraint(softmax[:, 1], a)
                            elif args.constraint == 'AE':
                                eval_loss_fairness = ae_constraint(loss_fn, log_softmax, y, a)
                            eval_fair_loss = args.fair_lambda * eval_loss_fairness

                eval_total_loss = eval_loss + eval_fair_loss
                eval_group_losses = sum(eval_group_of_loss.values())

                if torch.isnan(eval_total_loss):
                    pass
                else:
                    eval_overall_loss += eval_total_loss.detach().cpu().numpy()
                if torch.isnan(eval_loss):
                    pass
                else:
                    total_eval_loss += eval_loss.detach().cpu().numpy()
                eval_group_loss = eval_group_losses.detach().cpu().numpy()
                total_eval_fair_loss += eval_fair_loss

                avg_eval_overall_loss = eval_overall_loss / len(val_dl)
                avg_eval_loss = total_eval_loss / len(val_dl)
                avg_eval_fair_loss = total_eval_fair_loss / len(val_dl)
                avg_eval_group_loss = eval_group_loss / len(val_dl)

                eval_pbar.set_description(
                    f"Iter:{epoch+1:3}/{args.epochs:3} "
                    f"Avg_loss:{avg_eval_loss:.4f} "
                    f"Fair_loss:{avg_eval_fair_loss:.4f} "
                    f"Group_loss:{avg_eval_group_loss:.4f} ")
                eval_pbar.update()

                if args.task == 3:
                    predicted_survival_time = eval_scale * torch.exp(torch.log(time.to(args.device) + 1e-8) / eval_shape)
                    predicted_survival_times.append(predicted_survival_time.detach().cpu().numpy())
                    true_survival_times.append(time.detach().cpu().numpy())
                    events.append(event.detach().cpu().numpy())
                else:
                    predictions.append(torch.argmax(eval_cancer_pred.detach().cpu(), dim=1).numpy())
                    labels.append(label.detach().cpu().numpy())
                senAttrs.append(sensitive.detach().cpu().numpy())

        if num_classes > 2:
            npLabels = np.concatenate(labels)
            npPredictions = np.concatenate(predictions)
            npSenAttrs = np.concatenate(senAttrs)

            results = FairnessMetricsMultiClass(npPredictions, npLabels, npSenAttrs)
            acc = results["OverAllAcc"]
            fairness = results["EOdd"]
            criterion = 0                   # 0: performance 1: fairness 2: both
            if args.selection == "EOdd":
                fairness = results["EOdd"]
                criterion = 1
            elif args.selection == "avgEOpp":
                fairness = (results["EOpp0"] + results["EOpp1"]) / 2
                criterion = 1
            elif args.selection == "OverAllAcc":
                criterion = 0

            if not args.reweight:
                if acc > best_performance:
                    best_performance = acc
                    # Model save
                    torch.save(model.state_dict(), Path(model_save_path) / "model.pt")
                    epoch_record = epoch
                    performance_record = best_performance
                    print(f"Epoch:{epoch_record}, OverallAcc:{performance_record}")
            elif args.reweight:
                if criterion == 0:
                    if acc > best_performance:
                        best_performance = acc
                        torch.save(model.state_dict(), Path(model_save_path) / "model.pt")
                        epoch_record = epoch
                        performance_record = best_performance
                        print(f"Epoch:{epoch_record}, {args.selection}:{performance_record}")
                elif criterion == 1:
                    if fairness > 0 and fairness < best_fairness:
                        best_fairness = fairness
                        torch.save(model.state_dict(), Path(model_save_path) / "model.pt")
                        epoch_record = epoch
                        fairness_record = best_fairness
                        print(f"Epoch:{epoch_record}, {args.selection}:{fairness_record}")

            temp = {"Avg_Loss(train)": avg_train_loss,
                    "Avg_Loss(valid)": avg_eval_loss,
                    "Group(M) Majority": results["TOTALACC"][1],
                    "Group(m) Minority": results["TOTALACC"][0],
                    "Fair_Loss(train)": avg_train_fair_loss,
                    "Fair_Loss(valid)": avg_eval_fair_loss,
                    "Group_Loss(train)": avg_group_loss,
                    "Group_Loss(valid)": avg_eval_group_loss,
                    "Overall Loss(train)": avg_overall_loss,
                    "Overall Loss(valid)": avg_eval_overall_loss,
            } 
            wandb_record = {**temp, **results}
            wandb.log(wandb_record)

        elif num_classes == 2 and args.task != 3:
            npLabels = np.concatenate(labels)
            npPredictions = np.concatenate(predictions)
            npSenAttrs = np.concatenate(senAttrs)
            fpr, tpr, auroc, threshold = Find_Optimal_Cutoff(npLabels, npPredictions)
            predictions = torch.ge(torch.tensor(npPredictions), threshold).int()
            results = FairnessMetrics(npPredictions, npLabels, npSenAttrs)

            criterion = 0                   # 0: performance 1: fairness 2: both
            fairness = results["EOdd"]
            if args.selection == "EOdd":
                fairness = results["EOdd"]
                criterion = 1
            elif args.selection == "avgEOpp":
                fairness = (results["EOpp0"] + results["EOpp1"]) / 2
                criterion = 1
            elif args.selection == "AUROC":
                criterion = 0

            if not args.reweight:
                if auroc > best_performance:
                    best_performance = auroc
                    # Model save
                    torch.save(model.state_dict(), Path(model_save_path) / "model.pt")
                    epoch_record = epoch
                    performance_record = best_performance
                    print(f"Epoch:{epoch_record}, AUROC:{performance_record}")
            elif args.reweight:
                if criterion == 0:
                    if auroc > best_performance:
                        best_performance = auroc
                        torch.save(model.state_dict(), Path(model_save_path) / "model.pt")
                        epoch_record = epoch
                        performance_record = best_performance
                        print(f"Epoch:{epoch_record}, AUROC:{performance_record}")
                elif criterion == 1:
                    if fairness > 0 and fairness < best_fairness                :
                        best_fairness = fairness
                        torch.save(model.state_dict(), Path(model_save_path) / "model.pt")
                        epoch_record = epoch
                        fairness_record = best_fairness
                        print(f"Epoch:{epoch_record}, {args.selection}:{fairness_record}")

            temp = {"AUROC": auroc,
                    "Threshold": threshold,
                    "Avg_Loss(train)": avg_train_loss,
                    "Avg_Loss(valid)": avg_eval_loss,
                    "Fair_Loss(train)": avg_train_fair_loss,
                    "Fair_Loss(valid)": avg_eval_fair_loss,
                    "Group_Loss(train)": avg_group_loss,
                    "Group_Loss(valid)": avg_eval_group_loss,
                    "Overall Loss(train)": avg_overall_loss,
                    "Overall Loss(valid)": avg_eval_overall_loss,
            }
            wandb_record = {**temp, **results}
            wandb.log(wandb_record)

        elif num_classes == 2 and args.task == 3:
            try:
                npPredictedSurvivalTime = np.concatenate(predicted_survival_times)
            except ValueError:
                npPredictedSurvivalTime = predicted_survival_times
            try:
                npTrueSurvivalTime = np.concatenate(true_survival_times)
            except ValueError:
                npTrueSurvivalTime = true_survival_times
            try:
                npEvents = np.concatenate(events)
            except ValueError:
                npEvents = events
            try:
                npSenAttrs = np.concatenate(senAttrs)
            except ValueError:
                npSenAttrs = senAttrs
            try:
                nan_mask = np.isnan(npPredictedSurvivalTime) | np.isnan(npTrueSurvivalTime)
                npPredictedSurvivalTime = npPredictedSurvivalTime[~nan_mask]
                npTrueSurvivalTime = npTrueSurvivalTime[~nan_mask]
                npEvents = npEvents[~nan_mask]
                npSenAttrs = npSenAttrs[~nan_mask]
            except TypeError:
                pass
            
            results = SurvivalMetrics(npPredictedSurvivalTime, npTrueSurvivalTime, npEvents, npSenAttrs)
            
            criterion = 0  # 0: performance 1: fairness 2: both
            fairness = results["c_diff"]
            if args.selection == "c_diff":
                fairness = results["c_diff"]
                criterion = 1
            elif args.selection == "c_a":
                fairness = (results["c_long_diff"] + results["c_short_diff"]) / 2
                criterion = 1
            elif args.selection == "c_index":
                criterion = 0
            
            if not args.reweight:
                if (type(results["c_index"]) != str) and results["c_index"] > 0.0:
                    if results["c_index"] > best_performance:
                        best_performance = results["c_index"]
                        # Model save
                        torch.save(model.state_dict(), Path(model_save_path) / "model.pt")
                        epoch_record = epoch
                        performance_record = best_performance
                        print(f"Epoch:{epoch_record}, C-index:{performance_record}")
            elif args.reweight:
                if criterion == 0:
                    if (type(results["c_index"]) != str) and results["c_index"] > 0.0:
                        if results["c_index"] > best_performance:
                            best_performance = results["c_index"]
                            torch.save(model.state_dict(), Path(model_save_path) / "model.pt")
                            epoch_record = epoch
                            performance_record = best_performance
                            print(f"Epoch:{epoch_record}, C-index:{performance_record}")
                elif criterion == 1:
                    if (type(fairness) != str) and fairness > 0.0:
                        if epoch == 0 or fairness < best_fairness:
                            best_fairness = fairness
                            torch.save(model.state_dict(), Path(model_save_path) / "model.pt")
                            epoch_record = epoch
                            fairness_record = best_fairness
                            print(f"Epoch:{epoch_record}, {args.selection}:{fairness_record}")

            temp = {
                # "AUROC": auroc,
                # "Threshold": threshold,
                "Avg_Loss(train)": avg_train_loss,
                "Avg_Loss(valid)": avg_eval_loss,
                "Fair_Loss(train)": avg_train_fair_loss,
                "Fair_Loss(valid)": avg_eval_fair_loss,
                "Group_Loss(train)": avg_group_loss,
                "Group_Loss(valid)": avg_eval_group_loss,
                "Overall Loss(train)": avg_overall_loss,
                "Overall Loss(valid)": avg_eval_overall_loss,
            }
            wandb_record = {**temp, **results}
            wandb.log(wandb_record)
   
    wandb.finish()

    print(f"Epoch:{epoch_record}, Performance:{performance_record}")
    print(f"Epoch:{epoch_record}, {args.selection}:{fairness_record}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
