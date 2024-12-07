import pandas as pd


for vehicle in ["CHEVROLET_Spark", "HYUNDAI_Sonata", "KIA_Soul"]:
    attack_free = pd.read_csv(
        f"car_track_preliminary_train/Attack_free_{vehicle}_train.csv",
        names=["Time", "ID", "Length", "Data", "Label"],
    )
    flooding = pd.read_csv(
        f"car_track_preliminary_train/Flooding_{vehicle}_train.csv",
        names=["Time", "ID", "Length", "Data", "Label"],
    )
    fuzzy = pd.read_csv(
        f"car_track_preliminary_train/Fuzzy_{vehicle}_train.csv",
        names=["Time", "ID", "Length", "Data", "Label"],
    )
    malfunction = pd.read_csv(
        f"car_track_preliminary_train/Malfunction_{vehicle}_train.csv",
        names=["Time", "ID", "Length", "Data", "Label"],
    )
    attack_free["Label"] = 0.0
    flooding["Label"] = flooding["Label"].apply(lambda x: 1.0 if x == "T" else 0.0)
    fuzzy["Label"] = fuzzy["Label"].apply(lambda x: 2.0 if x == "T" else 0.0)
    malfunction["Label"] = malfunction["Label"].apply(
        lambda x: 3.0 if x == "T" else 0.0
    )
    df = pd.concat([attack_free, flooding, fuzzy, malfunction])
    print("Saving", vehicle)
    df.to_csv(
        f"car_track_preliminary_train/in-vehicle_{vehicle.split('_')[0].lower()}.csv",
        index=False,
    )
