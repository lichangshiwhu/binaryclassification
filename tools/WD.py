import matplotlib.pyplot as plt
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd

total_file_num = 11
data = np.loadtxt(open("test0.csv", "rb"), delimiter=",", usecols=(0, 1, 2, 3, 4))

for i in range(total_file_num - 1):
    testi = "test" + str(i + 1) + ".csv"
    datai = np.loadtxt(open(testi, "rb"), delimiter=",", usecols=(0, 1, 2, 3, 4))
    data = data + datai

data = data/total_file_num

x = [2, 3, 4, 5, 6, 7] 
y = [100, 5000, 10000, 15000, 20000, 25000]
z = data[:, 2].reshape(6, 6)

pd_z = pd.DataFrame(z, index=y, columns=x)

sns.set()


# fig = plt.figure(figsize=(12,8))
# ax = Axes3D(fig)
# fig.add_axes(ax)

# X, Y = np.meshgrid(x, y)

# print(X)
# print(Y)


# surf = ax.plot_surface(X, Y, z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))

# plt.title("WD")
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()


f, ax = plt.subplots(figsize=(9, 6))

sns.heatmap(pd_z, ax=ax, annot=True, cmap='YlOrRd')

ax.set_xlabel('Deep')
ax.set_ylabel('Width')

plt.show()
