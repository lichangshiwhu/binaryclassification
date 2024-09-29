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

# , './ShellsorPebbles'
# root_dirs = ['./CactusAerialPhotos', './CatAndDog', './ShellsorPebbles']
# pattern_strs = [['test_adv_CactusAerialPhotos_epsilon_baseline_'], ['test_adv_catanddog_epsilon_baseline_'], ['test_adv_ShellsorPebbles_epsilon_baseline_']]
# write_files = "output_adv_image_alpha_baseline.xlsx"
# rand_files = 3

root_dirs = ['./CatAndDog', './ShellsorPebbles']
pattern_strs = [['test_adv_catanddog_epsilon_baseline_'], ['test_adv_ShellsorPebbles_epsilon_baseline_']]
# sheet_names = [['catanddog'], ['ShellsorPebbles']]
write_files = "output_adv_image_epsilon.xlsx"
rand_files = 3

# root_dirs = ['./makeCircleX2Y2']
# pattern_strs = [['test_adv_makeCircleX2Y2_margin_']]
# write_files = "makeCircleX2Y2_margin.xlsx"
# rand_files = 5

# root_dirs = ['./makeByYeqX']
# pattern_strs = [['test_adv_makeByYeqX_margin_baseline_']]
# write_files = "makeByYeqX_margin_baseline.xlsx"
# rand_files = 10

def convert_value(x):  
    try:  
        return float(x)  
    except ValueError:  
        return np.nan

def export_statical_data(all_files):
    all_data = []
    for file in all_files:
        try:
            df = pd.read_csv(file, header=None)
        except:
            print(f"Find error in {file}")
        # [:, 2:]
        data = df.values[:, 2:]
        all_data.append(data)

    sum_data = all_data[0]
    for index in range(1, len(all_data)):
        sum_data += all_data[index]
# mean
    mean_data = sum_data / len(all_data)
# std
    sum_std_data = None
    for index in range(1, len(all_data)):
        zero_mean_data = (all_data[index] - mean_data) ** 2
        if sum_std_data is None:
            sum_std_data = zero_mean_data
        else:
            sum_std_data += zero_mean_data
    std_data = (sum_std_data / (len(all_data) - 1))**0.5
    return {'mean' : mean_data, 'std' : std_data}

with pd.ExcelWriter(write_files) as writer:
    for i in range(len(root_dirs)):
        for j in range(len(pattern_strs[i])):
            files = export_all_files(root_dirs[i], pattern_strs[i][j])
            print(files)
            assert len(files) == rand_files
            if len(files) == 0:
                print(root_dirs[i], pattern_strs[i][j])
            statical_data_dict = export_statical_data(files)
            for key in statical_data_dict.keys():
                statical_data = statical_data_dict[key]
                df = pd.DataFrame(statical_data)
                sheet_name = pattern_strs[i][j] + '_' + str(key)
                sheet_name = sheet_name[-30:]
                df.to_excel(writer, sheet_name=sheet_name, index=False)

# save_mean_data(root_dirs[0], pattern_strs[0][1]) 

