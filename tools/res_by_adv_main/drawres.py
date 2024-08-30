import pandas as pd  
import numpy as np

import matplotlib.pyplot as plt

#expand experiment
def generate_expand_wd():
    file_path = 'output_expand.xlsx'
    xls = pd.ExcelFile(file_path)    
    sheet_names = xls.sheet_names
    with pd.ExcelWriter('with_depth_heatmap.xlsx') as writer:  
        for sheet_name in sheet_names:
            arr = pd.read_excel(xls, sheet_name=sheet_name).values[:, 6].reshape(-1, 8)
            df = pd.DataFrame(arr)
            df.to_excel(writer, sheet_name=sheet_name, index=False)


#convergence speed with data number
def draw_data_speed():
    linewidth = 2
    file_path = 'output_adv_makedata_num.xlsx'
    xls = pd.ExcelFile(file_path)
    sheet_names = ['test_adv_makecircle_num_', 'test_adv_makeMoon_num_']

    titles = ['makeCircle', 'makeMoon']
    fig, axs = plt.subplots(1, 2, sharex=True, figsize=(12, 4))
    i = 0
    x = [500,1000,3000,5000,6000,9000, 12000, 15000, 18000, 21000, 23000, 26000]
    adv_alpha = ['0.05', '0.2', '0.4', '0.6', '0.8']
    markers = [ 'x','o','v', 's', 'p', 'P']
    for ax in axs.flatten():
        arr =  pd.read_excel(xls, sheet_name=sheet_names[i]).values[:, 8]
        logistic_arr = arr.reshape(-1, len(x)*len(adv_alpha))
        logistic_max = np.max(logistic_arr, axis=0)
        logistic_max = logistic_max.reshape(len(x), len(adv_alpha))
        for j in range(len(adv_alpha)):
            ax.plot(x, logistic_max[:, j], label=adv_alpha[j], linewidth = linewidth, linestyle='dashdot', marker = markers[j], markersize=8) 
        
        ax.legend()
        if i == 0:
            ax.set_ylabel('Accuracy/%')  
        ax.set_title(titles[i])  
        ax.set_xlabel('Data number')
        ax.grid(ls='--')
        i += 1

    plt.tight_layout()  
    plt.show()

def draw_nat_siglog():
    linewidth = 2
    file_path = 'output.xlsx'
    xls = pd.ExcelFile(file_path)
    sheet_names = ['nat_cactusaerialphotos_sigmoid', 'cactusaerialphotos_log', \
                   'nat_catanddog_sigmoid_', 'catanddog_log', \
                    'nat_ShellsorPebbles_sigmoid_', 'ShellsorPebbles_log']
    titles = ['CactusAerialPhotos', 'CatAndDog', 'ShellsorPebbles']
    adv_alpha = ['0.2', '0.4', '0.6', '0.8']
    #  13 nat
    #  12 log
    markers = [ 'x','o','v', 's', 'p', 'P']
    fig, axs = plt.subplots(1, 3, sharex=True, figsize=(10, 3))
    for i in range(3):
        ax = axs[i]
        sheet_name = sheet_names[2*i]
        arr_sig = pd.read_excel(xls, sheet_name=sheet_name, header=None).values[:, 13]
        sheet_name = sheet_names[2*i+1]
        arr_log = pd.read_excel(xls, sheet_name=sheet_name, header=None).values[:, 12]
        ax.plot(adv_alpha, arr_sig, label='Sigmoid Loss', linewidth = linewidth, linestyle='dashdot', marker = markers[0], markersize=8) 
        ax.plot(adv_alpha, arr_log, label='Logistic Loss', linewidth = linewidth, linestyle='dashdot', marker = markers[1], markersize=8) 
        ax.legend()
        if i == 0:
            ax.set_ylabel('Accuracy/%')  
        ax.set_title(titles[i])  
        ax.set_xlabel('Adversarial Level')
        ax.grid(ls='--')

    plt.tight_layout()  
    plt.show()

# draw_data_speed()
draw_nat_siglog()
