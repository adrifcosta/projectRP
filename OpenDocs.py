from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition
dataset = np.loadtxt('X_train.txt')
labels=np.loadtxt('y_train.txt')

print(dataset)



#Redução de dimensão-PCA
pca=decomposition.PCA(n_components=3)
pca.fit(dataset)
datasetPCA=pca.transform(dataset)



#Scatter dos pontos
colors = ['red','green','blue','purple','yellow','pink']
fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
ax.scatter(datasetPCA[:, 0], datasetPCA[:, 1], datasetPCA[:, 2], c=labels, cmap=matplotlib.colors.ListedColormap(colors))
plt.show()

