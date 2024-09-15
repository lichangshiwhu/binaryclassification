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
    titles = ['Cactus Aerial Photos', 'Cat and Dog', 'Shells or Pebbles']
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
        ax.set_xlabel('Noise Level')
        ax.grid(ls='--')

    plt.tight_layout()
    plt.show()

def draw_makeYByX_best_margin():
    linewidth = 2
    beg_index = 8
    end_index = -2
    file_path = 'makeByYeqX_margin.xlsx'
    import pandas as pd
    xls = pd.ExcelFile(file_path)
    margins = [0.15, 0.25, 0.35]
    lossscale = [0, 1, 2, 3, 4, 5]
    titles = ['Margin is 0.15', 'Margin is 0.25', 'Margin is 0.35']

    sheet_name = 'test_adv_makeByYeqX_margin_'
    markers = [ 'x','o','v', 's', 'p', 'P']
    fig, axs = plt.subplots(1, 3, sharex=True, figsize=(10, 3))
    pds = pd.read_excel(xls, sheet_name=sheet_name)
    draw_acc_index = 15
    y_label = 'Adversarial Accuracy/%'
    # 'Adversarial Accuracy/%'
    # 'Natural Accuracy/%'
    # 8 -> adv acc
    # 15 -> nat acc
    for i in range(3):
        ax = axs[i]
        pd = pds[pds[2] == margins[i]]
        nat_acc = None
        for j in range(len(lossscale)):
            scale = lossscale[j]
            subpd = pd[pd[1] == scale]
            if j == 0:
                ax.plot(subpd.values[beg_index:end_index, 3], subpd.values[beg_index:end_index, draw_acc_index], label='Logistic Loss', linewidth = linewidth, linestyle='dashdot', marker = markers[0], markersize=8) 
                continue
            if nat_acc is None:
                nat_acc =  subpd.values[beg_index:end_index, draw_acc_index]
            else:
                new_nat_acc = subpd.values[beg_index:end_index, draw_acc_index]
                index_val_loss = new_nat_acc > nat_acc
                nat_acc[index_val_loss] = new_nat_acc[index_val_loss]
        ax.plot(subpd.values[beg_index:end_index, 3], nat_acc, label='Focal Loss', linewidth = linewidth, linestyle='dashdot', marker = markers[1], markersize=8) 
        ax.legend()
        if i == 0:
            ax.set_ylabel(y_label)
        ax.set_title(titles[i])  
        ax.set_xlabel('Noise Level')
        ax.grid(ls='--')

    plt.tight_layout()
    plt.show()

def draw_makeYByX_margin():
    linewidth = 2
    beg_index = 9
    file_path = 'makeByYeqX_margin.xlsx'
    import pandas as pd
    xls = pd.ExcelFile(file_path)
    margins = [0.15, 0.25, 0.35]
    lossscale = [0, 1, 2, 3, 4, 5]
    titles = ['Cactus Aerial Photos', 'Cat and Dog', 'Shells or Pebbles']

    sheet_name = 'test_adv_makeByYeqX_margin_'
    markers = [ 'x','o','v', 's', 'p', 'P']
    fig, axs = plt.subplots(1, 3, sharex=True, figsize=(10, 3))
    pds = pd.read_excel(xls, sheet_name=sheet_name)
    draw_acc_index = 8
    # 8 -> adv acc
    # 15 -> nat acc
    for i in range(3):
        ax = axs[i]
        pd = pds[pds[2] == margins[i]]
        nat_acc = None
        for j in range(len(lossscale)):
            scale = lossscale[j]
            subpd = pd[pd[1] == scale]
            ax.plot(subpd.values[:, 3], subpd.values[:, draw_acc_index], label='Focal Loss, alpha = ' + str(scale), linewidth = linewidth, linestyle='dashdot', marker = markers[0], markersize=8) 
        ax.legend()
        if i == 0:
            ax.set_ylabel('Accuracy/%')  
        ax.set_title(titles[i])  
        ax.set_xlabel('Noise Level')
        ax.grid(ls='--')

    plt.tight_layout()
    plt.show()

def draw_makeYByX():
    size = 3000
    np.random.seed(seed=0)
    data = np.empty((0, 2))
    label = np.ones(size)

    while data.shape[0] < size:
        x = np.random.uniform(0, 1)
        y = np.random.uniform(0, 1)
        
        distance = np.abs(x-y) / np.sqrt(2)
        
        if distance >= 0.2:
            data = np.vstack((data, [x, y]))
            if x < y:
                label[data.shape[0]-1] = -1
    x1 = data[label==1]
    x2 = data[label==-1]
    plt.scatter(x1[:, 0], x1[:, 1])
    plt.scatter(x2[:, 0], x2[:, 1])
    plt.show()

def draw_circlex2y2():
    size = 3000
    np.random.seed(0)
    data = np.empty((0, 2))
    label = np.ones(size)
    margin = 0.1
    r2 = 0.6
    while data.shape[0] < size:
        x = np.random.uniform(0, 1)
        y = np.random.uniform(0, 1)
        p = x * x + y * y

        distance = np.abs(2 * (p - r2)) / (np.sqrt(4 * p + 1) + np.sqrt(4 * r2))

        if distance >= margin:
            data = np.vstack((data, [x, y]))
            if p < r2:
                label[data.shape[0]-1] = -1
    x1 = data[label==1]
    x2 = data[label==-1]

    print(x1.shape, x2.shape)
    plt.scatter(x1[:, 0], x1[:, 1])
    plt.scatter(x2[:, 0], x2[:, 1])
    plt.show()

def draw_makeYByX_res():
    linewidth = 2
    file_path = 'adv_makeByYeqX_margin.xlsx'
    xls = pd.ExcelFile(file_path)
    sheet_names = ['adv_makeByYeqX_margin_test_log', 'adv_makeByYeqX_margin_test_sig']
    margins = [0.15, 0.25, 0.35]
    titles = ['Margin is 0.15', 'Margin is 0.25', 'Margin is 0.35']
    noise_level = ['0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4']
    #  14 log
    #  15 sig
    markers = ['o', 'x', 'v', 's', 'p', 'P']
    fig, axs = plt.subplots(1, 3, sharex=True, figsize=(10, 3))
    for i in range(3):
        ax = axs[i]
        sheet_name = sheet_names[0]
        pd_log = pd.read_excel(xls, sheet_name=sheet_name)
        arr_log = pd_log[pd_log[1] == margins[i]].values[:, 14]
        sheet_name = sheet_names[1]
        pd_sig = pd.read_excel(xls, sheet_name=sheet_name)
        arr_sig = pd_sig[pd_sig[2] == margins[i]].values[:, 15]
        print(noise_level, arr_sig)
        ax.plot(noise_level, arr_log, label='Logistic Loss', linewidth = linewidth, linestyle='dashdot', marker = markers[1], markersize=8) 
        ax.plot(noise_level, arr_sig, label='Sigmoid Loss', linewidth = linewidth, linestyle='dashdot', marker = markers[0], markersize=8) 
        ax.legend()
        if i == 0:
            ax.set_ylabel('Accuracy/%')  
        ax.set_title(titles[i])  
        ax.set_xlabel('Noise Level')
        ax.grid(ls='--')

    plt.tight_layout()
    plt.show()

# draw_data_speed()
# draw_makeYByX()
# draw_nat_siglog()
# draw_makeYByX_res()
# draw_circlex2y2()
draw_makeYByX_best_margin()
