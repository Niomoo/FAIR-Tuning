import os
import sys
import ast
import argparse
import torch
import wandb
import glob
from torch.utils.data import DataLoader
from tqdm import tqdm
from network import ClfNet
from util import *
import numpy as np
from pathlib import Path
from dataset_tumorDetection import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler

def parse_args(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cancer",
        nargs="+",
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
        default="./models_base/",
        help="Path to save weights.",
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        default="",
        help="Path to LoRA pretrained weights.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Epochs for training.")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for sampling images."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for training."
    )
    parser.add_argument(
        "--curr_fold", type=int, default=0, help="For k-fold experiments, current fold."
    )
    parser.add_argument(
        "--partition",
        type=int,
        default=1,
        help="Data partition method:'1:train/valid/test(6:2:2), 2:k-folds'.",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=0, 
        help="Random seed for data partition."
    )
    parser.add_argument(
        "--lora", 
        action="store_true",
        help="For LoRA finetuning."
    )
    parser.add_argument(
        "--reweight", 
        action="store_true",
        help="Sample a balanced dataset."
    )
    parser.add_argument(
        "--rank", 
        type=int, 
        default=4, 
        help="Rank for LoRA."
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
        help="Model selection strategy for LoRA.",
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
        help="Dropout for LoRA"
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
    model_names = os.listdir(args.model_path + f"{cancer_folder}_{args.partition}/")
    subfolders = [folder for folder in model_names if os.path.isdir(os.path.join(args.model_path + f"{cancer_folder}_{args.partition}/", folder))]
    if not subfolders:
        max_index = 0
    else:
        model_indexes = [int(name.split("-")[0]) for name in subfolders]
        max_index = max(model_indexes)

    wandb_id = wandb.util.generate_id()
    if args.partition == 1:
        name = wandb_id
        if not args.lora and not args.reweight:  # 1 w/o LoRA
            max_index += 1
    elif args.partition == 2:
        name = wandb_id + f"_fold_{args.curr_fold}"
        if not args.lora and not args.reweight:  # 2 w/o LoRA
            if args.curr_fold == 0:
                max_index += 1
    group = "_".join(args.cancer) + "_" + str(args.partition)
    if args.lora:
        name += "_lora"
        group += "_lora"
        lora_names = os.listdir(args.model_path + f"{cancer_folder}_{args.partition}_lora/")
        subfolders = [folder for folder in lora_names if os.path.isdir(os.path.join(args.model_path + f"{cancer_folder}_{args.partition}_lora/", folder))]
        if not subfolders:
            max_lora_index = 0
        else:
            lora_indexes = [int(name.split("-")[0]) for name in subfolders]
            max_lora_index = max(lora_indexes)
            if args.partition == 1:  # 1 LoRA
                max_lora_index += 1
            else:  # 2 LoRA
                if args.curr_fold == 0:
                    max_lora_index += 1

        name = str(max_lora_index) + "-base-" + name
    elif args.reweight:
        name += "_reweight"
        group += "_reweight"
        reweight_names = os.listdir(args.model_path + f"{cancer_folder}_{args.partition}_reweight/")
        subfolders = [folder for folder in reweight_names if os.path.isdir(os.path.join(args.model_path + f"{cancer_folder}_{args.partition}_reweight/", folder))]
        if not subfolders:
            max_reweight_index = 0
        else:
            reweight_indexes = [int(name.split("-")[0]) for name in subfolders]
            max_reweight_index = max(reweight_indexes)
            if args.partition == 1:  # 1 reweight
                max_reweight_index += 1
            else:  # 2 reweight
                if args.curr_fold == 0:
                    max_reweight_index += 1

        name = str(max_reweight_index) + "-base-" + name
    else:
        name = str(max_index) + "-base-" + name

    wandb.init(
        project="Fairness_TumorDetection",
        entity="jennyliu-lyh",
        config=args,
        name=name,
        id=wandb_id,
        group=group,
    )

    # Dataset preparation
    data = generateDataSet(
        cancer=args.cancer,
        sensitive=eval(args.fair_attr),
        fold=args.partition,
        seed=args.seed,
    )
    df = data.train_valid_test(args.split_ratio, args.lora)
    num_classes = len(df["label"].unique())

    if args.partition == 1:
        train_ds, val_ds, test_ds = get_datasets(df, "vanilla", None, args.reweight)
    elif args.partition == 2:
        train_ds, val_ds, test_ds = get_datasets(df, "kfold", args.curr_fold)

    train_dl = DataLoader(train_ds, collate_fn=collate_fn, batch_size=args.batch_size, pin_memory=False)
    val_dl = DataLoader(val_ds, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=False, pin_memory=False)
    test_dl = DataLoader(test_ds, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=False, pin_memory=False)

    if not args.lora:
        model = ClfNet(classes=num_classes)
        model.train()

    elif args.lora:
        model = ClfNet(classes=num_classes, ft=True)
        replace_linear(model, rank=args.rank, dropout=args.dropout)

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
        lora.mark_only_lora_as_trainable(model)
        print("Set LoRA parameters trainable; freeze other parameters.")

    if args.reweight:
        model = ClfNet(classes=num_classes, ft=True)
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
    if args.reweight:
        # params = list(model.fc_target.parameters())
        params = list(model.fc_target[-1].parameters())
        optimizer = torch.optim.Adam(params, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.scheduler) # constant lr
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    loss_fn = torch.nn.CrossEntropyLoss()
    fairness_loss_fn = FairnessLoss(args.fair_lambda)

    # Output folder
    model_save_path = args.model_path + f"{group}/{name}"
    os.makedirs(model_save_path, exist_ok=True)

    epoch_record = 0
    performance_record = 0.0
    fairness_record = 9999.0
    best_performance = 0.0
    best_fairness = 9999.0
    group_samples = {}

    # Training/evaluation process
    for epoch in range(args.epochs):
        train_loss = 0.0
        fair_loss = 0.0
        group_loss = 0.0
        overall_loss = 0.0
        total_train_loss = 0.0
        total_fair_loss = 0.0

        if args.lora and not args.reweight:
            if epoch == 0:
                sampler = BalancedSampler(train_ds, args.batch_size)
                train_dl = DataLoader(train_ds, collate_fn=collate_fn, pin_memory=False, batch_sampler=sampler)
            else:
                sampler = BalancedSampler(train_ds, args.batch_size, True, group_samples)
                train_dl = DataLoader(train_ds, collate_fn=collate_fn, pin_memory=False, batch_sampler=sampler) 

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

        for _, data in pbar:
            optimizer.zero_grad()
            wsi_embeddings, lengths, sensitive, label, group = data
            if not args.lora:
                model.train()
            elif args.lora:
                lora.mark_only_lora_as_trainable(model)

            cancer_pred = model(wsi_embeddings.to(args.device), sensitive.to(args.device), lengths)

            train_loss, group_of_loss = loss_function(loss_fn, cancer_pred, torch.nn.functional.one_hot(label, num_classes).float().to(args.device), lengths, group)
            train_loss = train_loss / gradient_accumulation_steps
            if args.reweight:
                x, y, a = wsi_embeddings.to(args.device), label.to(args.device), sensitive.to(args.device)
                log_softmax, softmax = torch.nn.functional.log_softmax(cancer_pred, dim=1), torch.nn.functional.softmax(cancer_pred, dim=1)
                if args.constraint == '':
                    fair_loss = 0
                elif args.constraint == 'MMF':
                    fair_loss = args.fair_lambda * mmf_constraint(loss_fn, log_softmax, y, a)
                else:
                    if args.constraint == "EO":
                        fpr, fnr = eo_constraint(softmax[:, 1], y, a)
                        loss_fairness = fpr + fnr
                    elif args.constraint == "DI":
                        loss_fairness = di_constraint(softmax[:, 1], a)
                    elif args.constraint == "DP":
                        loss_fairness = dp_constraint(softmax[:, 1], a)
                    elif args.constraint == "AE":
                        loss_fairness = ae_constraint(loss_fn, log_softmax, y, a)
                    fair_loss = args.fair_lambda * loss_fairness
            #     fair_loss = fairness_loss_fn(cancer_pred, torch.nn.functional.one_hot(label, num_classes).float().to(args.device), lengths, group)

            total_loss = train_loss + fair_loss
            total_loss.backward()

            group_losses = sum(group_of_loss.values())
            if args.lora and not args.reweight:
                group_samples = batch_resample(args.batch_size, group_of_loss)

            if (epoch + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
            # optimizer.step()

            overall_loss += total_loss.detach().cpu().numpy()
            total_train_loss += train_loss.detach().cpu().numpy()
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
                f"Group_loss:{avg_group_loss:.4f} "
            )
            pbar.update()
        scheduler.step()

        model.eval()
        eval_pbar = tqdm(enumerate(val_dl), colour="blue", total=len(val_dl))
        eval_fair_loss = 0.0
        total_eval_loss = 0.0
        total_eval_fair_loss = 0.0
        eval_overall_loss = 0.0
        predictions = []
        race_predictions = []
        labels = []
        senAttrs = []
        with torch.no_grad():
            for _, data in eval_pbar:
                wsi_embeddings, lengths, sensitive, label, group = data
                eval_cancer_pred = model(wsi_embeddings.to(args.device), sensitive.to(args.device), lengths)
                eval_loss, group_of_loss = loss_function(loss_fn, eval_cancer_pred, torch.nn.functional.one_hot(label, num_classes).float().to(args.device), lengths, group)

                if args.reweight:
                    x, y, a = wsi_embeddings.to(args.device), label.to(args.device), sensitive.to(args.device)
                    log_softmax, softmax = torch.nn.functional.log_softmax(eval_cancer_pred, dim=1), torch.nn.functional.softmax(eval_cancer_pred, dim=1)
                    if args.constraint == '':
                        eval_fair_loss = 0
                    elif args.constraint == 'MMF':
                        eval_fair_loss = args.fair_lambda * mmf_constraint(loss_fn, log_softmax, y, a)
                    else:
                        if args.constraint == "EO":
                            fpr, fnr = eo_constraint(softmax[:, 1], y, a)
                            eval_loss_fairness = fpr + fnr
                        elif args.constraint == "DI":
                            eval_loss_fairness = di_constraint(softmax[:, 1], a)
                        elif args.constraint == "DP":
                            eval_loss_fairness = dp_constraint(softmax[:, 1], a)
                        elif args.constraint == 'AE':
                            eval_loss_fairness = ae_constraint(loss_fn, log_softmax, y, a)
                        eval_fair_loss = args.fair_lambda * eval_loss_fairness

                total_loss = eval_loss + eval_fair_loss
                group_losses = sum(group_of_loss.values())

                eval_overall_loss += total_loss.detach().cpu().numpy()
                total_eval_loss += eval_loss.detach().cpu().numpy()
                group_eval_loss = group_losses.detach().cpu().numpy()
                total_eval_fair_loss += eval_fair_loss

                avg_eval_overall_loss = eval_overall_loss / len(val_dl)
                avg_eval_loss = total_eval_loss / len(val_dl)
                avg_eval_fair_loss = total_eval_fair_loss / len(val_dl)
                avg_eval_group_loss = group_eval_loss / len(val_dl)

                eval_pbar.set_description(
                    f"Iter:{epoch+1:3}/{args.epochs:3} "
                    f"Avg_loss:{avg_eval_loss:.4f} "
                    f"Fair_loss:{avg_eval_fair_loss:.4f} "
                    f"Group_loss:{avg_eval_group_loss:.4f} "
                )
                eval_pbar.update()

                predictions.append(torch.argmax(eval_cancer_pred.detach().cpu(), dim=1).numpy())
                labels.append(label.detach().cpu().numpy())
                senAttrs.append(sensitive.detach().cpu().numpy())

        npLabels = np.concatenate(labels)
        npPredictions = np.concatenate(predictions)
        npSenAttrs = np.concatenate(senAttrs)
        fpr, tpr, auroc, threshold = Find_Optimal_Cutoff(npLabels, npPredictions)
        predictions = torch.ge(torch.tensor(npPredictions), threshold).int()
        results = FairnessMetrics(npPredictions, npLabels, npSenAttrs)

        criterion = 0  # 0: performance 1: fairness 2: both
        fairness = results["EOdd"]
        if args.selection == "EOdd":
            fairness = results["EOdd"]
            criterion = 1
        elif args.selection == "avgEOpp":
            fairness = (results["EOpp0"] + results["EOpp1"]) / 2
            criterion = 1
        elif args.selection == "AUROC":
            criterion = 0

        if not args.lora and not args.reweight:
            if auroc > best_performance:
                best_performance = auroc
                # Model save
                torch.save(model.state_dict(), Path(model_save_path) / "model.pt")
                epoch_record = epoch
                performance_record = best_performance
                print(f"Epoch:{epoch_record}, AUROC:{performance_record}")
        elif args.lora:
            if criterion == 0:
                if auroc > best_performance:
                    best_performance = auroc
                    torch.save(lora.lora_state_dict(model), Path(model_save_path) / "./lora.pt")
                    epoch_record = epoch
                    performance_record = best_performance
                    print(f"Epoch:{epoch_record}, AUROC:{performance_record}")
            elif criterion == 1:
                if epoch == 0 or (fairness > 0 and fairness < best_fairness):
                    best_fairness = fairness
                    torch.save(lora.lora_state_dict(model), Path(model_save_path) / "./lora.pt")
                    epoch_record = epoch
                    fairness_record = best_fairness
                    print(f"Epoch:{epoch_record}, {args.selection}:{fairness_record}")
        elif args.reweight:
            if criterion == 0:
                if auroc > best_performance:
                    best_performance = auroc
                    torch.save(model.state_dict(), Path(model_save_path) / "model.pt")
                    epoch_record = epoch
                    performance_record = best_performance
                    print(f"Epoch:{epoch_record}, AUROC:{performance_record}")
            elif criterion == 1:
                if epoch == 0 or (fairness > 0 and fairness < best_fairness):
                    best_fairness = fairness
                    torch.save(model.state_dict(), Path(model_save_path) / "model.pt")
                    epoch_record = epoch
                    fairness_record = best_fairness
                    print(f"Epoch:{epoch_record}, {args.selection}:{fairness_record}")

        temp = {
            "AUROC": auroc,
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
    wandb.finish()
    print(f"Epoch:{epoch_record}, Performance:{performance_record}")
    print(f"Epoch:{epoch_record}, {args.selection}:{fairness_record}")


if __name__ == "__main__":
    args = parse_args()
    main(args)