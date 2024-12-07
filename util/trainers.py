import torch
import wandb
from sklearn.metrics import classification_report
from tqdm import tqdm
import os


def train(
    model,
    train_loader,
    num_epochs,
    optimizer,
    loss_fn,
    wandb_run,
    input_size=None,
    sequence_length=None,
    device="cuda",
    validation_loader=None,
    retrain=False,
    checkpoint="checkpoints/model.ckpt",
):
    if not retrain and os.path.exists(checkpoint):
        print("Loading model from checkpoint")
        chk = torch.load(checkpoint, weights_only=True)
        model.load_state_dict(chk["model_state_dict"])
        optimizer.load_state_dict(chk["optimizer_state_dict"])
        start_epoch = chk["epoch"]
    else:
        start_epoch = 0

    if start_epoch >= num_epochs:
        print("Model already trained for the specified number of epochs")
        return

    print("Training the model from epoch", start_epoch + 1)
    if model.name == "LSTM":
        for epoch in range(start_epoch, num_epochs):
            model.train()
            running_loss = 0.0

            for i, (inputs, labels) in enumerate(tqdm(train_loader)):
                inputs = inputs.reshape(-1, sequence_length, input_size).to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                if isinstance(loss_fn, torch.nn.BCELoss):
                    outputs = outputs.squeeze()
                    labels = labels.float()
                elif isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                    outputs = outputs.view(-1, 12)

                loss = loss_fn(outputs, labels)
                running_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(
                "Epoch [{}/{}], Loss: {:.6f}".format(
                    epoch + 1, num_epochs, running_loss / (i + 1)
                )
            )
            wandb_run.log({"Loss": running_loss / (i + 1)})
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint,
            )

            if validation_loader:
                running_vloss = 0.0
                model.eval()

                # Disable gradient computation and reduce memory consumption.
                with torch.no_grad():
                    for i, (vinputs, vlabels) in enumerate(validation_loader):
                        vinputs = vinputs.reshape(-1, sequence_length, input_size).to(
                            device
                        )
                        vlabels = vlabels.to(device)
                        voutputs = model(vinputs)
                        if isinstance(loss_fn, torch.nn.BCELoss):
                            voutputs = voutputs.squeeze()
                            vlabels = vlabels.float()
                        loss = loss_fn(outputs, labels)
                        running_vloss += vloss.item()

                avg_vloss = running_vloss / (i + 1)
                wandb_run.log({"VLoss": avg_vloss})
    elif model.name == "DCNN":
        for epoch in range(start_epoch, num_epochs):
            model.train()
            running_loss = 0.0

            for i, (inputs, labels) in enumerate(tqdm(train_loader)):
                inputs = inputs.unsqueeze(1).to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs.squeeze(), labels)
                running_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(
                "Epoch [{}/{}], Loss: {:.6f}".format(
                    epoch + 1, num_epochs, running_loss / (i + 1)
                )
            )
            wandb_run.log({"Loss": running_loss / (i + 1)})
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint,
            )

            if validation_loader:
                running_vloss = 0.0
                model.eval()

                # Disable gradient computation and reduce memory consumption.
                with torch.no_grad():
                    for i, (vinputs, vlabels) in enumerate(validation_loader):
                        vinputs = vinputs.unsqueeze(1).to(device)
                        vlabels = vlabels.to(device)
                        voutputs = model(vinputs)
                        vloss = loss_fn(
                            voutputs.squeeze(),
                            vlabels,
                        )
                        loss = loss_fn(outputs, labels)
                        running_vloss += vloss.item()

                avg_vloss = running_vloss / (i + 1)
                wandb_run.log({"VLoss": avg_vloss})


@torch.no_grad()
def test(
    model,
    test_loader,
    wandb_run,
    sequence_length=None,
    input_size=None,
    threshold=0.5,
    full_report=False,
    target_names=None,
    device="cuda",
    label_name=None,
):
    print("Testing the model")
    model = model.to(device)
    model.eval()
    tp, tn, fp, fn = 0, 0, 0, 0
    if full_report:
        predictions_list = []
        labels_list = []
    if model.name == "LSTM":
        for inputs, labels in test_loader:
            inputs = inputs.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)

            outputs = model(inputs).squeeze()
            if model.num_classes != 1:
                predictions = torch.argmax(outputs, dim=1)
            else:
                predictions = outputs > threshold

            tp += ((predictions == labels) & (labels != 0)).sum().item()
            tn += ((predictions == labels) & (labels == 0)).sum().item()
            fp += ((predictions != labels) & (labels == 0)).sum().item()
            fn += ((predictions != labels) & (labels != 0)).sum().item()
            if full_report:
                predictions_list.append(predictions.cpu())
                labels_list.append(labels.cpu())
    elif model.name == "DCNN":
        for inputs, labels in test_loader:
            inputs = inputs.unsqueeze(1).to(device)
            labels = labels.to(device)

            outputs = model(inputs).squeeze()
            predictions = outputs > threshold

            tp += ((predictions == labels) & (labels != 0)).sum().item()
            tn += ((predictions == labels) & (labels == 0)).sum().item()
            fp += ((predictions != labels) & (labels == 0)).sum().item()
            fn += ((predictions != labels) & (labels != 0)).sum().item()
            if full_report:
                predictions_list.append(predictions.cpu())
                labels_list.append(labels.cpu())

    if tp == 0:
        precision = 0.0
        recall = 0.0
        f1_score = 0.0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)

    if label_name:
        print(f"Results for {label_name}:\n")
    print(f"tp: {tp}\ntn: {tn}\nfp: {fp}\nfn: {fn}\n")
    print("Precision: {:.6f}".format(precision))
    print("Recall: {:.6f}".format(recall))
    print("F1-Score: {:.6f}\n".format(f1_score))

    if full_report:
        predictions_list = [
            pred.unsqueeze(0) if pred.dim() == 0 else pred for pred in predictions_list
        ]
        all_predictions = torch.cat(predictions_list)
        all_labels = torch.cat(labels_list)
        classification_rep = classification_report(
            all_labels,
            all_predictions,
            digits=6,
            target_names=target_names,
            zero_division=0,
        )
        print(classification_rep)
        wandb_run.log(
            {
                "Confusion Matix": wandb.plot.confusion_matrix(
                    y_true=all_labels.tolist(),
                    preds=all_predictions.tolist(),
                    class_names=target_names,
                )
            }
        )
    wandb_run.log({"precision": precision, "recall": recall, "f1_score": f1_score})
