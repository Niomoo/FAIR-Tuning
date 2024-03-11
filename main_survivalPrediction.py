import os
import sys
import ast
import argparse
import torch
import wandb
import glob
from torch.utils.data import DataLoader
from tqdm import tqdm
from network_survivalPrediction import WeibullModel
from util import *
import numpy as np
import loralib as lora
from pathlib import Path
from dataset_survivalPrediction import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index


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
        "--seed", type=int, default=0, help="Random seed for data partition."
    )
    parser.add_argument("--lora", action="store_true", help="For LoRA finetuning.")
    parser.add_argument(
        "--reweight", action="store_true", help="Sample a balanced dataset."
    )
    parser.add_argument("--rank", type=int, default=4, help="Rank for LoRA.")
    parser.add_argument(
        "--fair_lambda", type=float, default=0.5, help="Parameter for fairness loss."
    )
    parser.add_argument(
        "--constraint", type=str, default="", help="Fairness constraint for fine-tuning"
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
        "--acc_grad", type=int, default=1, help="Accumulation gradient."
    )
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout for LoRA")
    parser.add_argument(
        "--scheduler_gamma", type=float, default=1, help="Gamma for scheduler"
    )
    parser.add_argument(
        "--scheduler_step", type=float, default=10, help="Steps for scheduler"
    )
    parser.add_argument(
        "--split_ratio", type=float, default=1.0, help="Split ratio for training set"
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def clip_gradient(grad):
    return torch.clamp(grad, -0.1, 1)


def main(args):
    # Wandb settings
    cancer_folder = "_".join(args.cancer)
    model_names = os.listdir(args.model_path + f"{cancer_folder}_{args.partition}/")
    subfolders = [
        folder
        for folder in model_names
        if os.path.isdir(
            os.path.join(args.model_path + f"{cancer_folder}_{args.partition}/", folder)
        )
    ]
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
        lora_names = os.listdir(
            args.model_path + f"{cancer_folder}_{args.partition}_lora/"
        )
        subfolders = [
            folder
            for folder in lora_names
            if os.path.isdir(
                os.path.join(
                    args.model_path + f"{cancer_folder}_{args.partition}_lora/", folder
                )
            )
        ]
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
        reweight_names = os.listdir(
            args.model_path + f"{cancer_folder}_{args.partition}_reweight/"
        )
        subfolders = [
            folder
            for folder in reweight_names
            if os.path.isdir(
                os.path.join(
                    args.model_path + f"{cancer_folder}_{args.partition}_reweight/",
                    folder,
                )
            )
        ]
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
        project="Fairness_SurvivalPrediction_C-index",
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
    num_classes = len(df["event"].unique())

    if args.partition == 1:
        train_ds, val_ds, test_ds = get_datasets(df, "vanilla", None, args.reweight)
    elif args.partition == 2:
        train_ds, val_ds, test_ds = get_datasets(df, "kfold", args.curr_fold)

    train_dl = DataLoader(
        train_ds, collate_fn=collate_fn, batch_size=args.batch_size, pin_memory=False
    )
    val_dl = DataLoader(
        val_ds,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=False,
    )
    test_dl = DataLoader(
        test_ds,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=False,
    )

    if not args.lora:
        model = WeibullModel()
        model.train()

    elif args.lora:
        model = WeibullModel(ft=True)
        replace_linear(model, rank=args.rank, dropout=args.dropout)

        cancer_folder = "_".join(args.cancer)
        if args.weight_path != "":
            weight_path = glob.glob(
                args.model_path
                + f"{cancer_folder}_{args.partition}/{args.weight_path}-*/model.pt"
            )[0]
        else:
            if args.partition == 1:
                weight_path = glob.glob(
                    args.model_path
                    + f"{cancer_folder}_{args.partition}/{max_index}-*/model.pt"
                )[0]
            elif args.partition == 2:
                weight_path = glob.glob(
                    args.model_path
                    + f"{cancer_folder}_{args.partition}/{max_index}-*_{args.curr_fold}/model.pt"
                )[0]
        model.load_state_dict(torch.load(weight_path), strict=False)
        print(f"Weights path:{weight_path}")
        print("Loaded pretrained weights.")
        lora.mark_only_lora_as_trainable(model)
        print("Set LoRA parameters trainable; freeze other parameters.")

    if args.reweight:
        model = WeibullModel(ft=True)
        cancer_folder = "_".join(args.cancer)
        if args.weight_path != "":
            weight_path = glob.glob(
                args.model_path
                + f"{cancer_folder}_{args.partition}/{args.weight_path}-*/model.pt"
            )[0]
        else:
            if args.partition == 1:
                weight_path = glob.glob(
                    args.model_path
                    + f"{cancer_folder}_{args.partition}/{max_index}-*/model.pt"
                )[0]
            elif args.partition == 2:
                weight_path = glob.glob(
                    args.model_path
                    + f"{cancer_folder}_{args.partition}/{max_index}-*_{args.curr_fold}/model.pt"
                )[0]
        model.load_state_dict(torch.load(weight_path), strict=False)
        print(f"Weights path:{weight_path}")
        print("Loaded pretrained weights.")

    model = model.to(args.device)

    # Settings
    gradient_accumulation_steps = args.acc_grad
    parameters_to_update = []
    if args.reweight:
        # params = list(model.fc[-1].parameters())
        for n, param in model.named_parameters():
            if n.startswith('fc.6'):
                param.requires_grad = True
                # param.register_hook(clip_gradient)
                parameters_to_update.append(param)
            else:
                param.requires_grad = False
        print("Params to learn:" + str(len(parameters_to_update)))
        optimizer = torch.optim.RMSprop(parameters_to_update, lr=args.lr, weight_decay=1e-4)
    else:
        for n, param in model.named_parameters():
            param.requires_grad = True
            parameters_to_update.append(param)
        print("Params to learn:" + str(len(parameters_to_update)))
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.scheduler) # constant lr
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma
    )

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

        # if args.reweight:
        #     if epoch == 0:
        #         sampler = BalancedSampler(train_ds, args.batch_size)
        #         train_dl = DataLoader(train_ds, collate_fn=collate_fn, pin_memory=False, batch_sampler=sampler)
        #     else:
        #         sampler = BalancedSampler(train_ds, args.batch_size, True, group_samples)
        #         train_dl = DataLoader(train_ds, collate_fn=collate_fn, pin_memory=False, batch_sampler=sampler)

        pbar = tqdm(enumerate(train_dl), colour="yellow", total=len(train_dl))

        # for sample in train_dl:
        #     emb, length, sensitive, event, time, groups = sample
        #     group_count = {}
        #     for batch_group in groups:
        #         if batch_group in group_count:
        #             group_count[batch_group] += 1
        #         else:
        #             group_count[batch_group] = 1
        #     print(group_count)

        for idx, data in pbar:
            wsi_embeddings, lengths, sensitive, event, time, group, stage = data

            shape_scale = model(wsi_embeddings.to(args.device), lengths)
            shape, scale = shape_scale[:, 0], shape_scale[:, 1]
            # print("shape_scale", shape_scale)
            # if torch.isnan(shape).any() or torch.isnan(scale).any():
            # print("nan", shape_scale)
            train_loss, group_of_loss = survival_loss_function(shape, scale, time.float().to(args.device), torch.nn.functional.one_hot(event, num_classes).float().to(args.device), lengths, group)
            train_loss = train_loss / gradient_accumulation_steps
            if args.reweight:
                x, y, a = (wsi_embeddings.to(args.device), event.to(args.device), sensitive.to(args.device))
                if args.constraint == "":
                    fair_loss = 0
                else:
                    predicted_survival_time = scale * torch.exp(torch.log(time.to(args.device) + 1e-8) / shape)

            total_loss = train_loss + fair_loss
            total_loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            group_losses = sum(group_of_loss.values())
            if args.lora and not args.reweight:
                group_samples = batch_resample(args.batch_size, group_of_loss)

            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f'Parameter: {name}, Gradient: {param.grad}')

            if (idx + 1) % gradient_accumulation_steps == 0:
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
                f"Group_loss:{avg_group_loss:.4f} "
            )
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
        predicted_survival_times = []
        true_survival_times = []
        events = []
        senAttrs = []
        with torch.no_grad():
            for idx, data in eval_pbar:
                wsi_embeddings, lengths, sensitive, event, time, group, stage = data
                eval_shape_scale = model(wsi_embeddings.to(args.device), lengths)
                eval_shape, eval_scale = eval_shape_scale[:, 0], eval_shape_scale[:, 1]
                eval_loss, eval_group_of_loss = survival_loss_function(eval_shape, eval_scale, time.float().to(args.device), torch.nn.functional.one_hot(event, num_classes) .float() .to(args.device), lengths, group)
                eval_loss = eval_loss / gradient_accumulation_steps
                if args.reweight:
                    # x, y, a = wsi_embeddings.to(args.device), event.to(args.device), sensitive.to(args.device)
                    # log_softmax, softmax = torch.nn.functional.log_softmax(eval_shape, dim=1), torch.nn.functional.softmax(eval_shape, dim=1)
                    if args.constraint == "":
                        eval_fair_loss = 0
                #     elif args.constraint == 'MMF':
                #         eval_fair_loss = args.fair_lambda * mmf_constraint(loss_fn, log_softmax, y, a)
                #     else:
                #         if args.constraint == "EO":
                #             fpr, fnr = eo_constraint(softmax[:, 1], y, a)
                #             eval_loss_fairness = fpr + fnr
                #         elif args.constraint == "DI":
                #             eval_loss_fairness = di_constraint(softmax[:, 1], a)
                #         elif args.constraint == "DP":
                #             eval_loss_fairness = dp_constraint(softmax[:, 1], a)
                #         elif args.constraint == 'AE':
                #             eval_loss_fairness = ae_constraint(loss_fn, log_softmax, y, a)
                #         eval_fair_loss = args.fair_lambda * eval_loss_fairness

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
                    f"Group_loss:{avg_eval_group_loss:.4f} "
                )
                eval_pbar.update()

                # prediction = np.resize(eval_shape_scale.cpu(), (len(eval_shape_scale), 2))
                # npResult = np.concatenate((np.array(time.cpu())[:, np.newaxis], prediction), axis=1)
                predicted_survival_time = eval_scale * torch.exp(torch.log(time.to(args.device) + 1e-8) / eval_shape)
                # predicted_survival_time = weibull_min.ppf(0.5, scale=npResult[1], c=npResult[2])

                # predictions.append(torch.argmax(eval_shape.detach().cpu(), dim=1).numpy())
                predicted_survival_times.append(predicted_survival_time.detach().cpu().numpy())
                true_survival_times.append(time.detach().cpu().numpy())
                events.append(event.detach().cpu().numpy())
                senAttrs.append(sensitive.detach().cpu().numpy())
        # npPredictions = np.concatenate(predictions)
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
        # fpr, tpr, auroc, threshold = Find_Optimal_Cutoff(npLabels, npPredictions)
        # predictions = torch.ge(torch.tensor(npPredictions), threshold).int()
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
        if not args.lora and not args.reweight:
            if (type(results["c_index"]) != str) and results["c_index"] > 0.0:
                if results["c_index"] > best_performance:
                    best_performance = results["c_index"]
                    # Model save
                    torch.save(model.state_dict(), Path(model_save_path) / "model.pt")
                    epoch_record = epoch
                    performance_record = best_performance
                    print(f"Epoch:{epoch_record}, C-index:{performance_record}")
        # elif args.lora:
        #     if criterion == 0:
        #         if auroc > best_performance:
        #             best_performance = auroc
        #             torch.save(lora.lora_state_dict(model), Path(model_save_path) / "./lora.pt")
        #             epoch_record = epoch
        #             performance_record = best_performance
        #             print(f"Epoch:{epoch_record}, AUROC:{performance_record}")
        #     elif criterion == 1:
        #         if epoch == 0 or (fairness > 0 and fairness < best_fairness):
        #             best_fairness = fairness
        #             torch.save(lora.lora_state_dict(model), Path(model_save_path) / "./lora.pt")
        #             epoch_record = epoch
        #             fairness_record = best_fairness
        #             print(f"Epoch:{epoch_record}, {args.selection}:{fairness_record}")
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
