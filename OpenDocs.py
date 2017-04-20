from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition
import pandas as pd
#dataset = np.loadtxt('X_train.txt')
#linhas são samples
#colunas são features
#labels=np.loadtxt('y_train.txt')


dataset=pd.read_csv('X_train.txt',delim_whitespace=True,  header=None);
dataset=dataset.as_matrix();
print(dataset)
print("Colunas", len(dataset[0,:]))
print("Linhas", len(dataset[:,0]))
#Matriz de correlação
matrix_corr=pd.DataFrame.corr(dataset);
plt.matshow(matrix_corr);

#Redução de dimensão-PCA
pca=decomposition.PCA(n_components=3)
pca.fit(dataset)
datasetPCA=pca.transform(dataset)

#Scatter dos pontos obtidos pelo PCA
colors = ['red','green','blue','purple','yellow','pink']
fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
ax.scatter(datasetPCA[:, 0], datasetPCA[:, 1], datasetPCA[:, 2], c=labels, cmap=matplotlib.colors.ListedColormap(colors))
plt.show()

