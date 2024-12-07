import pandas as pd

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

index = 1.0
full_df = pd.DataFrame()
for attack in ["DoS", "Fuzzy", "gear", "RPM"]:
    df = pd.read_csv(
        f"{attack}_dataset.csv",
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
    df = df.apply(reformat_car_hacking_row, axis=1)
    df["ID"] = df["ID"].apply(lambda x: int(x, 16))
    df['label'] = df['label'].apply(lambda x: index if x == "T" else 0.0)
    full_df = pd.concat([full_df, df], ignore_index=True)
    index += 1.0
full_df.to_csv("car-hacking-dataset.csv", index=False)