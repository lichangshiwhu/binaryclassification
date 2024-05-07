import os
import re  

import pandas as pd  
import numpy as np
  
def is_pre_num_post_format(str_to_check, prestr, poststr):  
    pattern = re.compile(f'^{re.escape(prestr)}\\d+{re.escape(poststr)}$')  
    match = pattern.match(str_to_check)  
    return match is not None

def export_all_files(directory, prestr, poststr = ".csv"):
    all_files = []
    for root, dirs, _ in os.walk(directory):  
        for _dir in dirs:
            sub_dirs = os.path.join(root, _dir)
            for _root, _, files in os.walk(sub_dirs):
                for file in files:
                    if is_pre_num_post_format(file, prestr, poststr):
                        file_path = os.path.join(_root, file)
                        all_files.append(file_path)
    return all_files

root_dirs = ['./breastCancer', './earlyStageDiabetesRiskPrediction', './houseVotes', './ionosphere', './musk', './statlog']       
pattern_strs = [['breastcancer_LogisticLoss_', 'breastcancer_sigmoidloss_'],
                ['earlyStageDiabetesRiskPrediction_LogisticLoss_', 'earlyStageDiabetesRiskPrediction_SigmoidLoss_'],
                ['housevotes_LogisticLoss_', 'housevotes_SigmoidLoss_'],
                ['ionosphere_LogisticLoss_', 'ionosphere_SigmoidLoss_'],
                ['musk_LogisticLoss_', 'musk_SigmoidLoss_'],
                ['statlog_LogisticLoss_', 'statlog_SigmoidLoss_'],
                ]

# root_dirs = ['./makeCircle', './makeMoon', './makeLinearClassifiction']
# pattern_strs = [['makecircle_'], ['makemoon_'], ['makeLinearClassifiction_']]

# root_dirs = ['./breastCancer']
# pattern_strs = [['breastcancer_LogisticLoss_', 'breastcancer_sigmoidloss_']]

def convert_value(x):  
    try:  
        return float(x)  
    except ValueError:  
        return np.nan

def export_mean_data(all_files):
    all_data = []
    for file in all_files:
        try:
            df = pd.read_csv(file, header=None)
        except:
            print(f"Find error in {file}")
        # [:, 2:]
        data = df.values
        all_data.append(data)

    sum_data = all_data[0]
    for index in range(1, len(all_data)):
        sum_data += all_data[index]

    mean_data = sum_data / len(all_data)
    return mean_data


with pd.ExcelWriter('output_width35.xlsx') as writer:  
    for i in range(len(root_dirs)):
        for j in range(len(pattern_strs[i])):
            files = export_all_files(root_dirs[i], pattern_strs[i][j])
            print(i, ":", j, files)
            # assert len(files) == 10
            if len(files) == 0:
                print(root_dirs[i], pattern_strs[i][j])
            mean_data = export_mean_data(files)
            df = pd.DataFrame(mean_data)
            df.to_excel(writer, sheet_name=pattern_strs[i][j], index=False)

# save_mean_data(root_dirs[0], pattern_strs[0][1]) 

