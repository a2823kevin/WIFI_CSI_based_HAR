import os
import pandas
import torch
from sklearn.decomposition import PCA

from preprocessing import *

def remove_head(fpath):
    print(fpath)
    with open(fpath, "r") as fin:
        data = fin.read()
        if (data[0]=="T"):
            return pandas.read_csv(fpath)

        row = ",".join([str(i) for i in range(0, 114)])
        for tx in range(1, 3):
            for rx in range(1, 3):
                for sc in range(1, 115):
                    row += f",T{tx}R{rx}SC{sc}_amp"
        for tx in range(1, 3):
            for rx in range(1, 3):
                for sc in range(1, 115):
                    row += f",T{tx}R{rx}SC{sc}_ang"

    with open(fpath, "w") as fout:
        fout.write(row+"\n"+data)

    fin = pandas.read_csv(fpath)
    for i in range(0, 114):
        fin = fin.drop(str(i), axis=1)
    fin.to_csv(fpath, index=False)

    return fin

def merge_data_and_label(folder_path):
    data = remove_head(f"{folder_path}/data.csv")
    data.insert(0, "label", [pandas.NA for i in range(len(data))])
    with open(f"{folder_path}/label.csv", "r") as label:
        for ln in label.readlines():
            idx, motion = ln.split(",")
            data["label"].loc[int(idx)] = motion.split("\n")[0]
    data = data.dropna()

    fout = f"assets/preprocessed_datasets/{folder_path.split('/')[-2]}_session{folder_path.split('/')[-1]}.csv"
    data.to_csv(fout, index=False)

def generate_CSI_dataset(fpath, settings, data_length, model=None, amplitude_only=True, threshold=0.75, n_PCA_components=None):
    fin = pandas.read_csv(fpath)
    #preprocess
    labels = fin["label"]
    if (amplitude_only):
        fin = normalize(fin.iloc[:, 1:457], settings["max_amplitude"], settings["min_amplitude"])
    else:
        fin = normalize(fin.iloc[:, 1:], settings["max_amplitude"], settings["min_amplitude"])

    datas = numpy.zeros(fin.shape, numpy.float64)
    for i in range(fin.shape[1]):
        datas[:, i] = reduce_noise(fin.iloc[:, i])
    
    if (n_PCA_components is not None):
        pca = PCA(n_PCA_components)
        new_datas = numpy.zeros((len(datas), n_PCA_components*4), numpy.float64)
        for i in range(0, 4):
            new_datas[:, n_PCA_components*i:n_PCA_components*(i+1)] = pca.fit_transform(datas[:, 114*i:114*(i+1)])
        datas = new_datas

    dataset = []
    action_dict = {
        "standing": 0,
        "walking": 1,
        "get_down": 2,
        "sitting": 3,
        "get_up": 4,
        "lying": 5,
        "no_person": 6
    }

    for i in range(len(datas)-data_length):
        data = datas[i:i+data_length, :]
        action_count = labels.iloc[i:i+data_length].value_counts()
        main_action = action_count.idxmax()
        if (action_count[main_action]/data_length>=threshold):
            data = torch.tensor(data, dtype=torch.float)
            if (model=="tcn"):
                data = torch.transpose(data, 0, 1)
            label = [0 for i in range(0, 7)]
            label[action_dict[main_action]] = 1
            label = torch.tensor(label, dtype=torch.float)
            dataset.append((data, label))

    return dataset

def merge_datasets(dataset_lst):
    for i in range(1, len(dataset_lst)):
        for j in range(len(dataset_lst[i])):
            dataset_lst[0].append(dataset_lst[i][j])
    return dataset_lst[0]

if __name__=="__main__":
    #preprocess datas
    '''
    ds_folder = "assets/wifi_csi_har_dataset"
    for room in os.listdir(ds_folder):
        for session in os.listdir(f"{ds_folder}/{room}"):
            if (session!=".DS_Store"):
                folder_path = f"{ds_folder}/{room}/{session}"
                merge_data_and_label(folder_path)
    '''
    with open("src/settings.json", "r") as fin:
        settings = json.load(fin)

    fpath = "assets/preprocessed_datasets/room_1_session1.csv"
    generate_CSI_dataset(fpath, settings, 25, n_PCA_components=50)