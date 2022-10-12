import os
import scipy.io

def convert_mat2csv(fpath):
    fin = scipy.io.loadmat(fpath)
    with open(f"./assets/CSI_datasets/{os.path.split(fpath)[-1][:-4]}.csv", "w") as fout:
        #head
        row = "label,"
        for i in range(1, 3):
            for j in range(1, 4):
                for k in range(1, 31):
                    row += f"T{i}R{j}SC{k},"
        row = row[:-1] + "\n"
        fout.write(row)

        #content
        for i in range(len(fin['Raw_Cell_Matrix'])):
            row = ""
            label = fin['Raw_Cell_Matrix'][i][0][0][0][-1].squeeze()
            mimo_csi = fin['Raw_Cell_Matrix'][i][0][0][0][-2]
            row = row + str(label) + ","
            for j in range(0, 2):
                for k in range(0, 3):
                    for sc in range(0, 30):
                        row = row + str(mimo_csi[j][k][sc])[1:-1] + ","
            row = row[:-1] + "\n"
            fout.write(row)

if __name__=="__main__":
    convert_mat2csv("S13_S21\I1\S13_S21_I1_T1.mat")