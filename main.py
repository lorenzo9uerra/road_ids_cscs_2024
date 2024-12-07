import torch
import numpy as np
import wandb
from util.data import (
    load_frames_car_hacking,
    load_frames_road,
    load_multiclass_car_hacking,
    load_multiclass_road,
    load_binary_in_vehicle,
    load_multiclass_in_vehicle,
    load_binary_road,
    load_frames_in_vehicle,
    load_binary_car_hacking,
)
from util.trainers import train, test
from util.models import LSTMModel, DCNN
from util.parser import get_parser
import yaml


def init_wandb(
    model_name,
    dataset,
    num_epochs,
    learning_rate,
    classification_type,
    label=None,
    threshold=0.5,
):
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="ids4can",
        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "architecture": model_name,
            "dataset": dataset,
            "epochs": num_epochs,
            "classification_type": classification_type,
            "threshold": threshold,
        },
        reinit=True,
    )
    if label:
        run.config.update({"label": label})
    return run


device = "cuda" if torch.cuda.is_available() else "cpu"

# https://pytorch.org/docs/stable/notes/randomness.html
np.random.seed(1)
torch.manual_seed(1)

if __name__ == "__main__":
    args = get_parser()
    dataset = args.dataset
    classification_type = args.classification_type
    model_name = args.model

    # Initialization
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    if model_name == "LSTM":
        test_percent = config[model_name][dataset]["test_percent"]
        sequence_length = config[model_name][dataset]["sequence_length"]
        num_epochs = config[model_name][dataset]["num_epochs"]
        learning_rate = config[model_name][dataset]["learning_rate"]
        hidden_size = config[model_name][dataset]["hidden_size"]
        num_layers = config[model_name][dataset]["num_layers"]
        input_size = config[model_name][dataset]["input_size"]
        labels_dict = config[model_name][dataset]["labels_dict"]
        threshold = config[model_name][dataset]["threshold"]

        if classification_type == "multiclass":
            label_name = None
            activation = "softmax"
            num_classes = len(labels_dict)
            loss_fn = torch.nn.CrossEntropyLoss()
        elif classification_type == "binary":
            label_name = labels_dict[args.label]
            activation = "sigmoid"
            num_classes = 1
            loss_fn = torch.nn.BCELoss()

        model = LSTMModel(
            input_size, hidden_size, num_layers, num_classes, device, activation
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        if classification_type == "binary":
            if dataset == "road":
                train_loader, test_loader = load_binary_road(
                    args.label, labels_dict, test_percent
                )
                dataset = dataset + "_" + labels_dict[args.label]
            elif "in-vehicle" in dataset:
                train_loader, test_loader = load_binary_in_vehicle(
                    dataset.split("_")[1], test_percent
                )
            elif dataset == "car-hacking":
                train_loader, test_loader = load_binary_car_hacking(
                    args.label, labels_dict, test_percent
                )
        elif classification_type == "multiclass":
            if dataset == "road":
                train_loader, test_loader = load_multiclass_road(test_percent)
            elif "in-vehicle" in dataset:
                train_loader, test_loader = load_multiclass_in_vehicle(
                    dataset.split("_")[1], test_percent
                )
            elif dataset == "car-hacking":
                train_loader, test_loader = load_multiclass_car_hacking(test_percent)

    elif model_name == "DCNN":
        test_percent = config[model_name][dataset]["test_percent"]
        num_epochs = config[model_name][dataset]["num_epochs"]
        learning_rate = config[model_name][dataset]["learning_rate"]
        threshold = config[model_name][dataset]["threshold"]
        labels_dict = config[model_name][dataset]["labels_dict"]
        weight_decay = config[model_name][dataset]["weight_decay"]
        sequence_length, input_size = None, None

        activation = "sigmoid"
        label_name = labels_dict[args.label]
        num_classes = 1
        if dataset == "road":
            train_loader, test_loader = load_frames_road(
                args.label, labels_dict, test_percent
            )
        elif "in-vehicle" in dataset:
            train_loader, test_loader = load_frames_in_vehicle(
                args.label, dataset.split("_")[1], test_percent
            )
        elif dataset == "car-hacking":
            train_loader, test_loader = load_frames_car_hacking(
                args.label, labels_dict, test_percent
            )
        loss_fn = torch.nn.BCELoss()
        model = DCNN().to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    # Start run
    run = init_wandb(
        model_name,
        dataset,
        num_epochs,
        learning_rate,
        classification_type,
        label_name,
        threshold,
    )
    if not args.test:
        train(
            model,
            train_loader,
            num_epochs,
            optimizer,
            sequence_length=sequence_length,
            input_size=input_size,
            loss_fn=loss_fn,
            device=device,
            wandb_run=run,
            checkpoint=f"checkpoints/{model_name}_{dataset}_{classification_type}{'_'+label_name if label_name else ''}.ckpt",
            retrain=args.retrain,
        )
    else:
        print("Loading model from checkpoint")
        chk = torch.load(
            f"checkpoints/{model_name}_{dataset}_{classification_type}{'_'+label_name if label_name else ''}.ckpt",
            weights_only=True,
        )
        model.load_state_dict(chk["model_state_dict"])
        optimizer.load_state_dict(chk["optimizer_state_dict"])
    test(
        model,
        test_loader,
        sequence_length=sequence_length,
        input_size=input_size,
        full_report=True,
        threshold=threshold,
        target_names=(
            list(labels_dict.values())
            if classification_type == "multiclass"
            else ["normal", label_name]
        ),
        device=device,
        label_name=dataset if not label_name else f"{dataset} {label_name}",
        wandb_run=run,
    )
wandb.finish()
