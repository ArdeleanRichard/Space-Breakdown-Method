from sbm import Spectrum2D
import numpy as np

def load_atoms_synthetic_data():
    data_folder = "./data/toy/"
    file = "atoms-2.csv"

    f = open(data_folder+file, "r")
    intro = f.readlines()[:5]
    f.close()

    timeValues = []
    for str_time in intro[1].split(","):
        timeValues.append(float(str_time))

    frequencyValues = []
    for str_time in intro[3].split(","):
        frequencyValues.append(float(str_time))

    data = np.loadtxt(data_folder + file, delimiter=",", dtype=float, skiprows=5)

    spectrumData = Spectrum2D(timeValues=np.array(timeValues), frequencyValues=frequencyValues, powerValues=data)

    return data, spectrumData
