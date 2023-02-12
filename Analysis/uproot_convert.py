import numpy as np
import pandas as pd
import uproot
import argparse

parser = argparse.ArgumentParser(
                    prog = 'uproot_covert',
                    description = 'Convert Root TTree to Pandas DataFrame and save as csv file',
                    epilog = '')

parser.add_argument('filename')
args = parser.parse_args()


filein = "stage1/"+args.filename+".root"

data = uproot.open(filein)["events"]

branches = data.keys()
remove_branch = ["n_jet",
            "n_muon",
            "n_electron",

            "jet_e",
            "jet_pt",
            "jet_phi",
            "jet_eta",

            "muon_e",
            "muon_pt",
            "muon_phi",
            "muon_eta",

            "electron_e",
            "electron_pt",
            "electron_phi",
            "electron_eta",]

new_branch = []
for b in branches:
    if b in remove_branch:
        continue
    else:
        new_branch.append(b)

df = pd.DataFrame(columns=new_branch)

for column in df.columns:
    print(column)
    print(data.arrays(column))
    df[column] = data[column].array().to_numpy()


df.to_csv(args.filename+'.csv')


#print(data.arrays(b))
#print(data[b].array().to_numpy())

#dfObj = data.pandas.df()#`.DataFrame.from_dict(data.arrays(branches))

#print(df.columns)
