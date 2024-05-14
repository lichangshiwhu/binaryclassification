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


# depth experiment
def draw_depth():
    linewidth = 2
    file_path = 'output_depth.xlsx'
    xls = pd.ExcelFile(file_path)
    sheet_names = ['breastcancer_LogisticLoss_depth', 'breastcancer_sigmoidloss_depth_', \
    'statlog_LogisticLoss_depth_', 'statlog_SigmoidLoss_depth_', \
    'ionosphere_LogisticLoss_depth_', 'ionosphere_SigmoidLoss_depth_', \
    'housevotes_LogisticLoss_depth_', 'housevotes_SigmoidLoss_depth_', \
    'musk_LogisticLoss_depth_', 'musk_SigmoidLoss_depth_', \
    'esdrp_logistic', 'esdrp_sigmoid']

    fig, axs = plt.subplots(2, 3, sharex=True, figsize=(12, 6))
    titles = ['Breast Cancer', 'Statlog', 'Ionosphere', 'Congressional Voting Records', 'Musk', 'Early Stage Diabetes Risk Prediction']
    markers = ['x', 'o', 'v', 's', 'p', 'P']
    i = 0
    x = np.arange(3, 21)
    for ax in axs.flatten():
        y1 = pd.read_excel(xls, sheet_name=sheet_names[i * 2]).values[:, 5]
        ax.plot(x, y1, label='Logistic loss', linewidth = linewidth, linestyle='dashdot', marker = markers[0], markersize=8)
        y2 = pd.read_excel(xls, sheet_name=sheet_names[i * 2 + 1]).values[:, 5]
        ax.plot(x, y2, label='Sigmoid loss', linewidth = linewidth, linestyle='dashdot', marker = markers[1], markersize=8)  

        ax.legend()
        if i % 3 == 0:
            ax.set_ylabel('Accuracy/%', fontsize=15)  
        ax.set_title(titles[i])  
        if i >= 2:
            ax.set_xlabel('depth', fontsize=15)
        ax.grid(ls='--')
        i += 1

    plt.tight_layout()
    plt.show()

#width experiment
def draw_width():
    linewidth = 2
    file_path = 'output_width.xlsx'
    xls = pd.ExcelFile(file_path)
    sheet_names = ['breastcancer', 'statlog', \
    'ionosphere', 'housevotes_', \
    'musk', 'esdrp']

    fig, axs = plt.subplots(2, 3, sharex=True, figsize=(12, 6))
    titles = ['Breast Cancer', 'Statlog', 'Ionosphere', 'Congressional Voting Records', 'Musk', 'Early Stage Diabetes Risk Prediction']
    markers = ['x', 'o', 'v', 's', 'p', 'P']
    i = 0
    x = np.arange(3, 36)
    print(x)
    for ax in axs.flatten():
        y1 = pd.read_excel(xls, sheet_name=sheet_names[i], header=None).values[:, 0]
        print(y1)
        ax.plot(x, y1, label='Logistic loss', linewidth = linewidth, linestyle='dashdot', marker = markers[0], markersize=8)
        y2 = pd.read_excel(xls, sheet_name=sheet_names[i], header=None).values[:, 1]
        ax.plot(x, y2, label='Sigmoid loss', linewidth = linewidth, linestyle='dashdot', marker = markers[1], markersize=8)  

        ax.legend()
        if i % 3 == 0:
            ax.set_ylabel('Accuracy', fontsize=15)  
        ax.set_title(titles[i])  
        if i >= 2:
            ax.set_xlabel('Width', fontsize=15)
        ax.grid(ls='--')
        i += 1

    plt.tight_layout()
    plt.show()

#swap training strategy
def draw_noise_width35_depth10():
    linewidth = 2
    file_path = 'output_noise_width35_depth10.xlsx'
    xls = pd.ExcelFile(file_path)
    sheet_names = ['breastcancer_noise_', 'statlog_noise_', \
    'ionosphere_noise_', 'housevotes_noise_', \
    'musk_noise_', 'earlyStageDiabetesRiskPredictio']

    titles = ['Breast Cancer', 'Statlog', 'Ionosphere', 'Congressional Voting Records', 'Musk', 'Early Stage Diabetes Risk Prediction']
    nn_numbers = [3*3, 4*2, 4*5, 6*3, 3*2, 4*5]
    loss_names = ['LogisticLoss', 'Logistic then sigmoid', 'LogisticAndWelschLoss', 'LogisticAndSavageLoss', 'Sigmoid only', 'Sigmoid then logistic']
    keep_loss = [0, 1, 0, 0, 1, 1]
    loss_numbers = len(loss_names)
    fig, axs = plt.subplots(2, 3, sharex=True, figsize=(12, 6))
    i = 0
    x = [0, 0.08, 0.16, 0.24, 0.32, 0.40]
    markers = ['x', 'o', 'v', 's', 'p', 'P']

    for ax in axs.flatten():
        values =  pd.read_excel(xls, sheet_name=sheet_names[i]).values
        width = values[:, 1]
        depth = values[:, 2]
        all_acc = values[:, 6]
        loss_acc = all_acc.reshape(-1, len(x), nn_numbers[i])
        width = width.reshape(-1, len(x), nn_numbers[i])
        depth = depth.reshape(-1, len(x), nn_numbers[i])
        # print(loss_acc)
        assert loss_acc.shape[0] == len(loss_names)
        y = np.max(loss_acc, axis=2)
        index = np.argmax(loss_acc, axis=2)
        print(index.shape)
        print(width.shape)
        i_indices, j_indices = np.indices(index.shape)
        print(f"width:{width[i_indices, j_indices, index]}")
        print(f"depth:{depth[i_indices, j_indices, index]}")
        for j in range(len(loss_names)):
            if keep_loss[j]:
                ax.plot(x, y[j], label=loss_names[j], linewidth = linewidth, linestyle='dashdot', marker = markers[j], markersize=8)  

        ax.legend()
        ax.set_ylabel('Accuracy')  
        ax.set_title(titles[i])  
        if i >= 3:  
            ax.set_xlabel('Depth')
        i += 1

    plt.tight_layout()  
    plt.show()

#swap training strategy
def draw_noise_width6_depth6():
    linewidth = 2
    file_path = 'output_noise_W6_D6.xlsx'
    xls = pd.ExcelFile(file_path)
    sheet_names = ['breastcancer_noise_', 'statlog_noise_', \
    'ionosphere_noise_', 'housevotes_noise_', \
    'musk_noise_', 'earlyStageDiabetesRiskPredictio']

    titles = ['Breast Cancer', 'Statlog', 'Ionosphere', 'Congressional Voting Records', 'Musk', 'Early Stage Diabetes Risk Prediction']
    nn_numbers = [16, 16, 16, 16, 16, 16]
    loss_names = ['Logistic then sigmoid', 'Sigmoid only', 'Sigmoid then logistic']
    keep_loss = [1, 1, 1]
    loss_numbers = len(loss_names)
    fig, axs = plt.subplots(2, 3, sharex=True, figsize=(12, 6))
    i = 0
    x = ["0", "8", "16", "24", "32", "40"]
    markers = ['o', 'v', 'x', 's', 'p', 'P']

    for ax in axs.flatten():
        values =  pd.read_excel(xls, sheet_name=sheet_names[i]).values
        width = values[:, 1]
        depth = values[:, 2]
        all_acc = values[:, 6]
        loss_acc = all_acc.reshape(-1, len(x), nn_numbers[i])
        width = width.reshape(-1, len(x), nn_numbers[i])
        depth = depth.reshape(-1, len(x), nn_numbers[i])
        # print(loss_acc)
        assert loss_acc.shape[0] == len(loss_names)
        y = np.max(loss_acc, axis=2)
        index = np.argmax(loss_acc, axis=2)
        print(index.shape)
        print(width.shape)
        i_indices, j_indices = np.indices(index.shape)
        print(f"width:{width[i_indices, j_indices, index]}")
        print(f"depth:{depth[i_indices, j_indices, index]}")
        for j in range(len(loss_names)):
            if keep_loss[j]:
                ax.plot(x, y[j], label=loss_names[j], linewidth = linewidth, linestyle='dashdot', marker = markers[j], markersize=8)  

        ax.legend()
        if i % 3 == 0:
            ax.set_ylabel('Accuracy/%')  
        ax.set_title(titles[i])  
        if i >= 3:  
            ax.set_xlabel('The percentage of mislabeled sample/%')
        ax.grid(ls='--')
        i += 1

    plt.tight_layout()  
    plt.show()

#convergence speed with data number
def draw_data_speed():
    linewidth = 2
    file_path = 'output_data_speed.xlsx'
    xls = pd.ExcelFile(file_path)
    sheet_names = ['makeCircle_', 'makeMoon_']

    titles = ['Makecircle', 'Makemoon']
    fig, axs = plt.subplots(1, 2, sharex=True, figsize=(12, 4))
    i = 0
    x = [500,2000,4000,6000,8000,10000,12000,14000,16000,18000,20000,]
    markers = [ 'x','o','v', 's', 'p', 'P']
    for ax in axs.flatten():
        arr =  pd.read_excel(xls, sheet_name=sheet_names[i]).values[:, 6]
        logistic_arr = arr[:176].reshape(-1, len(x))
        sigmoid_arr = arr[176:].reshape(-1, len(x))
        logistic_max = np.max(logistic_arr, axis=0)
        sigmoid_max = np.max(sigmoid_arr, axis=0)
        ax.plot(x, logistic_max, label='Logistic loss', linewidth = linewidth, linestyle='dashdot', marker = markers[0], markersize=8) 
        ax.plot(x, sigmoid_max, label='Sigmoid loss', linewidth = linewidth, linestyle='dashdot', marker = markers[1], markersize=8)
        
        ax.legend()
        if i == 0:
            ax.set_ylabel('Accuracy/%')  
        ax.set_title(titles[i])  
        ax.set_xlabel('Data number')
        ax.grid(ls='--')
        i += 1

    plt.tight_layout()  
    plt.show()

# generate_expand_wd()
# draw_depth()
# draw_width()
# draw_noise_width35_depth10()
draw_data_speed()
# draw_noise_width6_depth6()
