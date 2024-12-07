from binascii import unhexlify
import os
from bitstring import BitArray
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import torch
from torch.utils.data import Dataset
import re


class PandasDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.iloc[:, :-1].values.astype(np.float32)
        self.labels = dataframe.iloc[:, -1].values.astype(np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        inputs = torch.tensor(self.data[index])
        labels = torch.tensor(self.labels[index], dtype=torch.long)
        return inputs, labels


class FrameDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return torch.tensor(self.X[index]), torch.tensor(self.y[index])


def pad_id(id_bits):
    while len(id_bits) != 29:
        id_bits.insert(0, 0)
    return id_bits


def build_frames(ids, labels, sequence_length=29):
    X, y = [], []
    ids = [BitArray(unhexlify(hex(x)[2:].upper().rjust(4, "0"))).bin for x in ids]
    num_samples = len(ids) - (len(ids) % sequence_length)
    for i in range(0, num_samples, sequence_length):
        sequence = []
        for j in ids[i : i + sequence_length]:
            sequence.append(pad_id([int(x) for x in list(j)]))
        label = 0 if all(labels[i : i + sequence_length] == 0) else 1
        X.append(sequence)
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    return X, y


def reformat_car_hacking_row(row):
    if int(row["DLC"]) != 8:
        row["label"] = row["DATA" + str(int(row["DLC"]))]
        for i in range(int(row["DLC"])):
            row["DATA" + str(i)] = int(row["DATA" + str(i)], 16)
        for i in range(int(row["DLC"]), 8):
            row["DATA" + str(i)] = 0
    else:
        for i in range(8):
            row["DATA" + str(i)] = int(row["DATA" + str(i)], 16)
    return row


def load_binary_road(label, labels_dict, test_percent):
    directory_in_str = "data/road/"
    directory = os.fsencode(directory_in_str)

    dfs = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        regex = labels_dict[label] + r".{0,2}\.csv"
        pattern = re.compile(regex)
        if pattern.match(filename):
            # print(filename)
            dfs.append(
                pd.read_csv(
                    directory_in_str + filename,
                    names=[
                        "timestamp",
                        "ID",
                        "DATA0",
                        "DATA1",
                        "DATA2",
                        "DATA3",
                        "DATA4",
                        "DATA5",
                        "DATA6",
                        "DATA7",
                        "label",
                    ],
                )
            )
    df = pd.concat(dfs)
    df.drop(["timestamp"], axis=1, inplace=True)
    df["label"] = np.where(df["label"] == label, 1, 0)
    train_df, test_df = train_test_split(df, test_size=test_percent)
    smote = SMOTE(sampling_strategy={1: 100000}, random_state=42)
    X, y = smote.fit_resample(train_df.drop(["label"], axis=1), train_df["label"])
    train_df = pd.concat([X, y], axis=1)
    train_df.reset_index(drop=True, inplace=True)
    train_df = train_df.astype(np.float32)
    test_df = test_df.astype(np.float32)
    train_loader = torch.utils.data.DataLoader(
        PandasDataset(train_df), batch_size=512, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        PandasDataset(test_df), batch_size=512, shuffle=False
    )
    return train_loader, test_loader


def load_binary_in_vehicle(vehicle, test_percent):
    # this performs binary classification over the entire vehicle dataset, without distinguishing between the different types of attacks
    df = pd.read_csv(
        "data/in-vehicle_dataset/in-vehicle_" + vehicle + ".csv",
        dtype={"Time": float, "ID": str, "Length": int, "Label": float, "Type": str},
    )
    df.drop(["Time", "Type"], axis=1, inplace=True)
    df["Data"] = df["Data"].str.split(" ")
    df["Data"] = df["Data"].apply(lambda x: [int(i, 16) for i in x])
    df["ID"] = df["ID"].apply(lambda x: int(x, 16))
    df.drop(["Length"], axis=1, inplace=True)
    df[["DATA0", "DATA1", "DATA2", "DATA3", "DATA4", "DATA5", "DATA6", "DATA7"]] = (
        pd.DataFrame(df["Data"].tolist())
    )
    df.drop(["Data"], axis=1, inplace=True)
    df.fillna(0.0, inplace=True)

    train_df, test_df = train_test_split(df, test_size=test_percent)
    smote = SMOTE(sampling_strategy={x: 100000 for x in range(1, 4)}, random_state=42)
    X, y = smote.fit_resample(train_df.drop(["Label"], axis=1), train_df["Label"])
    train_df = pd.concat([X, y], axis=1)
    train_df.reset_index(drop=True, inplace=True)
    # prepare for binary classification
    train_df["Label"] = np.where(train_df["Label"] != 0, 1, 0)
    test_df["Label"] = np.where(test_df["Label"] != 0, 1, 0)
    train_df = train_df.astype(np.float32)
    test_df = test_df.astype(np.float32)
    # train_df has Label as the last column, which we will index in the PandasDataframe class, but test doesn't, so we need to move it
    test_df["Label"] = test_df.pop("Label")
    train_loader = torch.utils.data.DataLoader(
        PandasDataset(train_df), batch_size=512, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        PandasDataset(test_df), batch_size=512, shuffle=False
    )
    return train_loader, test_loader


def load_binary_car_hacking(label, labels_dict, test_percent):
    filenames = {
        "DoS": "DoS_dataset.csv",
        "Fuzzing": "Fuzzy_dataset.csv",
        "Gear Spoofing": "gear_dataset.csv",
        "RPM Spoofing": "RPM_dataset.csv",
    }
    if os.path.exists(
        f"data/processed/car-hacking-dataset/{filenames[labels_dict[label]]}"
    ):
        df = pd.read_csv(
            f"data/processed/car-hacking-dataset/{filenames[labels_dict[label]]}"
        )
    else:
        df = pd.read_csv(
            f"data/car-hacking-dataset/{filenames[labels_dict[label]]}",
            names=[
                "timestamp",
                "ID",
                "DLC",
                "DATA0",
                "DATA1",
                "DATA2",
                "DATA3",
                "DATA4",
                "DATA5",
                "DATA6",
                "DATA7",
                "label",
            ],
        )
        df.drop(["timestamp"], axis=1, inplace=True)
        df = df.apply(reformat_car_hacking_row, axis=1)
        df["ID"] = df["ID"].apply(lambda x: int(x, 16))
        df["label"] = np.where(df["label"] == "R", 0, 1)
        df.fillna(0.0, inplace=True)
        if not os.path.exists("data/processed/car-hacking-dataset/"):
            os.makedirs("data/processed/car-hacking-dataset/")
        df.to_csv(
            f"data/processed/car-hacking-dataset/{filenames[labels_dict[label]]}",
            index=False,
        )

    train_df, test_df = train_test_split(df, test_size=test_percent, random_state=42)
    # prepare for binary classification
    train_df = train_df.astype(np.float32)
    test_df = test_df.astype(np.float32)
    train_loader = torch.utils.data.DataLoader(
        PandasDataset(train_df), batch_size=512, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        PandasDataset(test_df), batch_size=512, shuffle=False
    )
    return train_loader, test_loader


def load_multiclass_road(test_percent):
    df = pd.read_csv("data/road/road_1.csv.gz")
    df.drop(["timestamp"], axis=1, inplace=True)
    train_df, test_df = train_test_split(df, test_size=test_percent)
    smote = SMOTE(sampling_strategy={x: 100000 for x in range(1, 12)}, random_state=42)
    X, y = smote.fit_resample(train_df.drop(["label"], axis=1), train_df["label"])
    train_df = pd.concat([X, y], axis=1)
    train_df.reset_index(drop=True, inplace=True)
    train_loader = torch.utils.data.DataLoader(
        PandasDataset(train_df), batch_size=512, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        PandasDataset(test_df), batch_size=512, shuffle=False
    )
    return train_loader, test_loader


def load_multiclass_in_vehicle(vehicle, test_percent):
    df = pd.read_csv(
        "data/in-vehicle_dataset/in-vehicle_" + vehicle + ".csv",
        dtype={"Time": float, "ID": str, "Length": int, "Label": float, "Type": str},
    )
    df.drop(["Time", "Type"], axis=1, inplace=True)
    df["Data"] = df["Data"].str.split(" ")
    df["Data"] = df["Data"].apply(lambda x: [int(i, 16) for i in x])
    df["ID"] = df["ID"].apply(lambda x: int(x, 16))
    df.drop(["Length"], axis=1, inplace=True)
    df[["DATA0", "DATA1", "DATA2", "DATA3", "DATA4", "DATA5", "DATA6", "DATA7"]] = (
        pd.DataFrame(df["Data"].tolist())
    )
    df.drop(["Data"], axis=1, inplace=True)
    df.fillna(0.0, inplace=True)
    train_df, test_df = train_test_split(df, test_size=test_percent)
    smote = SMOTE(sampling_strategy={x: 100000 for x in range(1, 4)}, random_state=42)
    X, y = smote.fit_resample(train_df.drop(["Label"], axis=1), train_df["Label"])
    train_df = pd.concat([X, y], axis=1)
    train_df.reset_index(drop=True, inplace=True)
    train_df = train_df.astype(np.float32)
    test_df = test_df.astype(np.float32)
    # train_df has Label as the last column, which we will index in the PandasDataframe class, but test doesn't, so we need to move it
    test_df["Label"] = test_df.pop("Label")
    train_loader = torch.utils.data.DataLoader(
        PandasDataset(train_df), batch_size=512, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        PandasDataset(test_df), batch_size=512, shuffle=False
    )
    return train_loader, test_loader


def load_multiclass_car_hacking(test_percent):
    df = pd.read_csv("data/car-hacking-dataset.csv")
    df.drop(["timestamp"], axis=1, inplace=True)
    df.fillna(0.0, inplace=True)
    print(df["label"].value_counts())
    train_df, test_df = train_test_split(df, test_size=test_percent)
    # Unnecessary
    # smote = SMOTE(sampling_strategy={x: 100000 for x in range(1, 5)}, random_state=42)
    # X, y = smote.fit_resample(train_df.drop(["label"], axis=1), train_df["label"])
    # train_df = pd.concat([X, y], axis=1)
    # train_df.reset_index(drop=True, inplace=True)
    train_df = train_df.astype(np.float32)
    test_df = test_df.astype(np.float32)

    # train_df has Label as the last column, which we will index in the PandasDataframe class, but test doesn't, so we need to move it
    train_loader = torch.utils.data.DataLoader(
        PandasDataset(train_df), batch_size=512, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        PandasDataset(test_df), batch_size=512, shuffle=False
    )
    return train_loader, test_loader


def load_frames_road(label, labels_dict, test_percent):
    if (
        os.path.exists(f"data/frames/road/{labels_dict[label]}_features_train.npy")
        and os.path.exists(f"data/frames/road/{labels_dict[label]}_labels_train.npy")
        and os.path.exists(f"data/frames/road/{labels_dict[label]}_features_test.npy")
        and os.path.exists(f"data/frames/road/{labels_dict[label]}_labels_test.npy")
    ):
        print("Loading frames Data")
        X_train = np.load(f"data/frames/road/{labels_dict[label]}_features_train.npy")
        y_train = np.load(f"data/frames/road/{labels_dict[label]}_labels_train.npy")
        X_test = np.load(f"data/frames/road/{labels_dict[label]}_features_test.npy")
        y_test = np.load(f"data/frames/road/{labels_dict[label]}_labels_test.npy")
    else:
        directory_in_str = "data/road/"
        directory = os.fsencode(directory_in_str)

        dfs = []
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if "masquerade" not in labels_dict[label]:
                regex = labels_dict[label] + r".{0,2}\.csv"
            else:
                regex = (
                    labels_dict[label].split("_masquerade")[0]
                    + r".{0,2}_masquerade\.csv"
                )
            pattern = re.compile(regex)
            if pattern.match(filename):
                dfs.append(
                    pd.read_csv(
                        directory_in_str + filename,
                        names=[
                            "timestamp",
                            "ID",
                            "DATA0",
                            "DATA1",
                            "DATA2",
                            "DATA3",
                            "DATA4",
                            "DATA5",
                            "DATA6",
                            "DATA7",
                            "label",
                        ],
                    )
                )
        df = pd.concat(dfs, ignore_index=True)
        df["label"] = np.where(df["label"] == label, 1, 0)
        X, y = df["ID"].to_numpy(), df["label"].to_numpy()

        X, y = build_frames(X, y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_percent, random_state=42
        )

        # X_train has shape == [n_samples, 29, 29] but we need to reshape it to [n_samples, 29*29] for SMOTE
        # X_train = X_train.reshape(-1, 29*29)
        # smote = SMOTE(sampling_strategy={1: int(1e5)}, random_state=42)
        # X_train, y_train = smote.fit_resample(X_train, y_train)
        # Reshape back to [n_samples, 29, 29]
        # X_train = X_train.reshape(-1, 29, 29)

        if not os.path.exists("data/frames/road/"):
            os.makedirs("data/frames/road/")
        np.save(f"data/frames/road/{labels_dict[label]}_features_train.npy", X_train)
        np.save(f"data/frames/road/{labels_dict[label]}_labels_train.npy", y_train)
        np.save(f"data/frames/road/{labels_dict[label]}_features_test.npy", X_test)
        np.save(f"data/frames/road/{labels_dict[label]}_labels_test.npy", y_test)

    train_loader = torch.utils.data.DataLoader(
        FrameDataset(X_train, y_train), batch_size=64, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        FrameDataset(X_test, y_test), batch_size=64, shuffle=False
    )
    return train_loader, test_loader


def load_frames_in_vehicle(label, vehicle, test_percent):
    if (
        os.path.exists(f"data/frames/in-vehicle/{label}_{vehicle}_features_train.npy")
        and os.path.exists(f"data/frames/in-vehicle/{label}_{vehicle}_labels_train.npy")
        and os.path.exists(
            f"data/frames/in-vehicle/{label}_{vehicle}_features_test.npy"
        )
        and os.path.exists(f"data/frames/in-vehicle/{label}_{vehicle}_labels_test.npy")
    ):
        print("Loading frames Data")
        X_train = np.load(
            f"data/frames/in-vehicle/{label}_{vehicle}_features_train.npy"
        )
        y_train = np.load(f"data/frames/in-vehicle/{label}_{vehicle}_labels_train.npy")
        X_test = np.load(f"data/frames/in-vehicle/{label}_{vehicle}_features_test.npy")
        y_test = np.load(f"data/frames/in-vehicle/{label}_{vehicle}_labels_test.npy")
    else:
        data = pd.read_csv(
            f"data/in-vehicle_dataset/in-vehicle_{vehicle}.csv",
            dtype={
                "Time": float,
                "ID": str,
                "Length": int,
                "Data": str,
                "Label": float,
                "Type": str,
            },
        )
        data["Label"] = np.where(data["Label"] == label, 1, 0)
        y = data["Label"]
        X = data["ID"].apply(lambda x: int(x, 16))

        X, y = build_frames(X, y)
        if not os.path.exists("data/frames/in-vehicle/"):
            os.makedirs("data/frames/in-vehicle/")
        np.save(f"data/frames/in-vehicle/{vehicle}_{label}_features.npy", X)
        np.save(f"data/frames/in-vehicle/{vehicle}_{label}_labels.npy", y)

    # Need to shuffle the data otherwise the test set will be all 0s
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_percent, shuffle=True
    )
    train_loader = torch.utils.data.DataLoader(
        FrameDataset(X_train, y_train), batch_size=64, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        FrameDataset(X_test, y_test), batch_size=64, shuffle=False
    )
    return train_loader, test_loader


def load_frames_car_hacking(label, labels_dict, test_percent):
    filenames = {
        "DoS": "DoS_dataset.csv",
        "Fuzzing": "Fuzzy_dataset.csv",
        "Gear Spoofing": "gear_dataset.csv",
        "RPM Spoofing": "RPM_dataset.csv",
    }
    if os.path.exists(
        f"data/frames/car-hacking-dataset/{labels_dict[label]}_features.npy"
    ) and os.path.exists(
        f"data/frames/car-hacking-dataset/{labels_dict[label]}_labels.npy"
    ):
        X = np.load(
            f"data/frames/car-hacking-dataset/{labels_dict[label]}_features.npy"
        )
        y = np.load(f"data/frames/car-hacking-dataset/{labels_dict[label]}_labels.npy")
    else:
        data = pd.read_csv(
            f"data/car-hacking-dataset/{filenames[labels_dict[label]]}",
            names=[
                "timestamp",
                "ID",
                "DLC",
                "DATA0",
                "DATA1",
                "DATA2",
                "DATA3",
                "DATA4",
                "DATA5",
                "DATA6",
                "DATA7",
                "label",
            ],
        )
        data["ID"] = data["ID"].apply(lambda x: int(x, 16))
        data["label"] = np.where(data["label"] != "R", 1, 0)
        X, y = build_frames(data["ID"].to_numpy(), data["label"].to_numpy())
        if not os.path.exists("data/frames/car-hacking-dataset/"):
            os.makedirs("data/frames/car-hacking-dataset/")
        np.save(f"data/frames/car-hacking-dataset/{labels_dict[label]}_features.npy", X)
        np.save(f"data/frames/car-hacking-dataset/{labels_dict[label]}_labels.npy", y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_percent, shuffle=True
    )
    train_loader = torch.utils.data.DataLoader(
        FrameDataset(X_train, y_train), batch_size=64, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        FrameDataset(X_test, y_test), batch_size=64, shuffle=False
    )
    return train_loader, test_loader
