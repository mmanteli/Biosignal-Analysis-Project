import pandas as pd
import glob
import os
import numpy as np




# read the data

PATH = os.path.join(os.getcwd(), 'data', 'train')

raw_data = []

for label in glob.glob(PATH+"/*"):
    for file in glob.glob((label+"/*.csv")):
        data = pd.read_csv(file)
        data["label"] = [label.split(os.sep)[-1]]*len(data)
        raw_data.append(data)


# filtteroi jokainen raw_data jonon jäsen



# etsi tarvittavat ominaisuudet ja tee uusi taulukko
# indeksi   feat1   feat2   ... featn   label


# kaksi eri mallia ja tulokset


# graafinen esitys -koodi
    

