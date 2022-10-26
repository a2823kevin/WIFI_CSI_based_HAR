import os
import json

import numpy
import pandas
import pywt

def update_amp_settings():
    max_lst = []
    min_lst = []
    for data in os.listdir("assets/preprocessed_datasets"):
        if (data.endswith(".csv")):
            fin = pandas.read_csv(f"assets/preprocessed_datasets/{data}")
            amplitudes = fin.iloc[:, 1:457].to_numpy()
            max_lst.append(float(amplitudes.max()))
            min_lst.append(float(amplitudes.min()))

    with open("src/settings.json", "r") as fin:
        settings = json.load(fin)
    with open("src/settings.json", "w") as fout:
        settings["max_amplitude"] = max(max_lst)
        settings["min_amplitude"] = min(min_lst)
        json.dump(settings, fout)

def update_class_weight():
    action_dict = {
        "standing": 0,
        "walking": 1,
        "get_down": 2,
        "sitting": 3,
        "get_up": 4,
        "lying": 5,
        "no_person": 6
    }

    counting_table = {}
    for i in range(0, 7):
        counting_table[i] = 0

    for data in os.listdir("assets/preprocessed_datasets"):
        if (data.endswith(".csv")):
            fin = pandas.read_csv(f"assets/preprocessed_datasets/{data}")
            count = fin["label"].value_counts()
            for key in count.keys():
                counting_table[action_dict[key]] += count[key]
    
    n_sample = sum(counting_table.values())
    for key in counting_table.keys():
        counting_table[key] /= n_sample
    
    with open("src/settings.json", "r") as fin:
        settings = json.load(fin)
    with open("src/settings.json", "w") as fout:
        settings["class_weight"] = list(counting_table.values())
        json.dump(settings, fout)

def normalize(data, max_amp, min_amp):
    return (data.select_dtypes(include=[numpy.float64])-min_amp) / (max_amp-min_amp)

def hampel(data, k=7, t=3):
    L = 1.4826
    rolling_median = data.rolling(k).median()
    difference = numpy.abs(rolling_median-data)
    median_abs_deviation = difference.rolling(k).median()
    threshold = t * L * median_abs_deviation
    outlier_idx = difference > threshold
    data[outlier_idx] = rolling_median

    return data

def reduce_noise(data):
    #outlier removal
    data = hampel(data).to_numpy()
    data_length = len(data)

    #wavelet analysis
    wavelet = pywt.Wavelet("sym5")
    max_level = pywt.dwt_max_level(len(data), wavelet.dec_len)

    #decompose
    coefficients = pywt.wavedec(data, wavelet, level=max_level)

    #filtering
    for i in range(len(coefficients)):
        #see coefficients 5% smaller than max coef as noise
        coefficients[i] = pywt.threshold(coefficients[i], 0.05*max(coefficients[i]))

    #reconstruct
    data = pywt.waverec(coefficients, wavelet)

    return data[:data_length]


if __name__=="__main__":
    update_amp_settings()
    update_class_weight()