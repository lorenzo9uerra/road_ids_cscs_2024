import pandas as pd
import os
import numpy as np
import json

directory_in_str = "attacks/"
directory = os.fsencode(directory_in_str)

if not os.path.exists("csv/"):
    os.mkdir("csv/")

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".log") and not filename.startswith('accelerator_attack_'):
        with open(directory_in_str+filename, "r") as fp:
            csv = open("csv/"+filename[:-4]+".csv", "w")
            for line in fp:
                # (1000000000.000000) can0 20E#4E2003A0C63F8FFF
                tokens = line.split()
                id, data = tokens[2].split('#')
                csv_line = [tokens[0][1:-1],id,data[0:2],data[2:4],data[4:6],data[6:8],data[8:10],data[10:12],data[12:14],data[14:16]]
                csv.write(",".join(csv_line)+"\n")
            csv.close()

directory_in_str = "ambient/"
directory = os.fsencode(directory_in_str)

if not os.path.exists("csv/ambient/"):
    os.mkdir("csv/ambient/")

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".log"):
        with open(directory_in_str+filename, "r") as fp:
            csv = open("csv/ambient/"+filename[:-4]+".csv", "w")
            for line in fp:
                # (1000000000.000000) can0 20E#4E2003A0C63F8FFF
                tokens = line.split()
                id, data = tokens[2].split('#')
                csv_line = [tokens[0][1:-1],id,data[0:2],data[2:4],data[4:6],data[6:8],data[8:10],data[10:12],data[12:14],data[14:16]]
                csv.write(",".join(csv_line)+"\n")
            csv.close()

metadata = json.load(open("attacks/capture_metadata.json"))
# the list of labels contains the categories of attacks in the same order as in the metadata file
labels = [1,2]*3+[3]*3+[4,5]+[6,7]*3+[8,9]*3+[10,11]*3

# correlated_signal_attack = 1
# correlated_signal_attack_masquerade = 2
# fuzzing_attack = 3
# max_engine_coolant_temp_attack = 4
# max_engine_coolant_temp_attack_masquerade = 5
# max_speedometer_attack = 6
# max_speedometer_attack_masquerade = 7
# reverse_light_off_attack = 8
# reverse_light_off_attack_masquerade = 9
# reverse_light_on_attack = 10
# reverse_light_on_attack_masquerade = 11

idx_label = 0
full_df = pd.DataFrame()
for file in list(metadata.keys())[4:]:
    print(f"Parsing and labelling {file}")
    df = pd.read_csv(f"csv/{file}.csv",names=["timestamp","ID","DATA0","DATA1","DATA2","DATA3","DATA4","DATA5","DATA6","DATA7"])
    df["label"] = 0
    start = df["timestamp"].values[0]
    condition = [
        (df["timestamp"] >= start + metadata[file]["injection_interval"][0])
        & (df["timestamp"] <= start + metadata[file]["injection_interval"][1])
    ]
    if "XXX" not in metadata[file]["injection_id"]:
        condition[0] = (df["ID"] == metadata[file]["injection_id"][2:].upper().rjust(3,"0")) & (condition[0])
    for i in range(8):
        if "XX" not in metadata[file]["injection_data_str"][i*2:(i*2)+2]:
            condition[0] = (df[f"DATA{i}"] == metadata[file]["injection_data_str"][i*2:(i*2)+2]) & (condition[0])
    df["label"] = np.select(condition, [labels[idx_label]])
    df['ID'] = df['ID'].apply(int, base=16)
    for i in range(8):
        df[f'DATA{i}'] = df[f'DATA{i}'].apply(int, base=16)
    df.to_csv(f"csv/{file}.csv", index=False, header=None)
    full_df = pd.concat([full_df, df], ignore_index=True)
    idx_label += 1
ambient_directory = "csv/ambient/"
for file in os.listdir(ambient_directory):
    if file.endswith(".csv"):
        print(f"Reading {file} from ambient folder")
        df = pd.read_csv(os.path.join(ambient_directory, file), names=["timestamp","ID","DATA0","DATA1","DATA2","DATA3","DATA4","DATA5","DATA6","DATA7"])
        df['ID'] = df['ID'].apply(int, base=16)
        df["label"] = 0
        for i in range(8):
            df[f'DATA{i}'] = df[f'DATA{i}'].apply(int, base=16)
        full_df = pd.concat([full_df, df], ignore_index=True)
print("Saving full dataset to road_1.csv.gz")
full_df.to_csv("csv/road_1.csv.gz", index=False, compression='gzip')
