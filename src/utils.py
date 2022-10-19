import pandas

def remove_head(fpath):
    with open(fpath, "r") as fin:
        data = fin.read()
        if (data[0]=="T"):
            return pandas.read_csv(fpath)

        row = ",".join([str(i) for i in range(0, 114)])
        mag = []
        for tx in range(1, 3):
            for rx in range(1, 3):
                for sc in range(1, 115):
                    row += f",T{tx}R{rx}SC{sc}_mag"
        for tx in range(1, 3):
            for rx in range(1, 3):
                for sc in range(1, 115):
                    row += f",T{tx}R{rx}SC{sc}_ang"

    with open(fpath, "w") as fout:
        fout.write(row+"\n"+data)

    fin = pandas.read_csv(fpath)
    for i in range(0, 114):
        fin = fin.drop(str(i), axis=1)

    return fin

def merge_data_and_label(folder_path):
    data = remove_head(f"{folder_path}/data.csv")
    data.insert(0, "label", [pandas.NA for i in range(len(data))])
    with open(f"{folder_path}/label.csv", "r") as label:
        for ln in label.readlines():
            idx, motion = ln.split(",")
            data["label"].loc[int(idx)] = motion.split("\n")[0]
    data = data.dropna()

    fout = f"assets/preproccessed_datasets/{folder_path.split('/')[-2]}_session{folder_path.split('/')[-1]}.csv"
    data.to_csv(fout, index=False)

if __name__=="__main__":
    folder_path = "assets/wifi_csi_har_dataset/room_1/1"
    merge_data_and_label(folder_path)