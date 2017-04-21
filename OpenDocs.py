from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition
import pandas as pd
from sklearn import preprocessing
import scipy.stats
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

features_names=pd.read_csv('features.txt', delim_whitespace=True,header=None)
labels_multi=pd.read_csv('y_train.txt', header=None)
dataset=pd.read_csv('X_train.txt',delim_whitespace=True,  header=None)
dataset=dataset.as_matrix()

labels_multi=labels_multi.as_matrix()
labels_bin=np.zeros(len(labels_multi))

labels_bin = np.squeeze(labels_bin)
labels_multi = np.squeeze(labels_multi)

for i in range(0, len(labels_multi)):
    if labels_multi[i]<4:
        labels_bin[i]=1
    else:
        labels_bin[i] = 0

#LABELS_MULTI
#1-Walking
#2-Walking-upstairs
#3-Walking- downstairs
#4-Sitting
#5-Standing
#6-Laying

#LABELS_BIN
#0-Not Walking
#1-Walking

#Nomes de todas as features
features_names=features_names.as_matrix()[:,1]


#Normalização dos dados
#dataset_scaled tem média nula e desvio padrão unitário
dataset_scaled=preprocessing.scale(dataset)

#SELEÇAO DE FEATURES
#-----------------------------------------------------------------
#Matriz de correlação
#corrcoef() usa estrutura de dados em que as features estão por linhas
matrix_corr=np.corrcoef(dataset_scaled.transpose())
mean_corr=labels_bin=np.zeros(len(matrix_corr[:,0]))

for i in range (0, len(matrix_corr[:,0])):
    mean_corr[i]=np.mean(matrix_corr[i,:])
mean_corr=np.absolute(mean_corr)


mean_corr_cm=[]
features_names_cm=[]
all_indexs=[]

for j in range (0, len(mean_corr)):
    if (mean_corr[j]<0.03):
        mean_corr_cm.append(mean_corr[j])
        features_names_cm.append(features_names[j])
        all_indexs.append(j)

reduced_data_cm = np.zeros((len(dataset_scaled[:,1]),len(mean_corr_cm)))
i = 0
for k in all_indexs:
    reduced_data_cm[:,i]=dataset_scaled[:,k]
    i=i+1
#reduced_data_cm dados que resultaram da redução por interpretação dos coeficientes de correlação
#-----------------------------------------------------------------
#Kruskall Wallis
'''
scores_kw_multi=[];
pValue_kw_multi=[];

scores_kw_bin=[];
pValue_kw_bin=[];

for i in range(0,(len(dataset_scaled[1,:]))):
    data=dataset_scaled[:,i]

    hstat,pval = scipy.stats.mstats.kruskalwallis(data,labels_bin)
    #hstat1,pval1 = scipy.stats.mstats.kruskalwallis(data,labels_multi)
    scores_kw_multi.append(hstat)
    pValue_kw_multi.append(pval)
    #scores_kw_bin.append(hstat1)
    #pValue_kw_bin.append(pval1)

print(pValue_kw_multi)
#
#
# print(pValue_kw_bin)
'''
#-----------------------------------------------------------------
#Recursive Feature Elimination
'''
model = LogisticRegression()
rfe = RFE(model, 3)
labels_multi=np.squeeze(labels_multi)
rfe = rfe.fit(dataset_scaled, labels_multi)
print(rfe.support_)
print(rfe.ranking_)
'''
#-----------------------------------------------------------------
#Feature Importance using Extra Trees Classifier
'''
labels_multi=np.squeeze(labels_multi)
model = ExtraTreesClassifier()
model.fit(dataset_scaled, labels_multi)
print(model.feature_importances_)
'''
#REDUÇÃO DE DIMENSÃO
def scatter3d(data,labels, title, colors):
    plt.figure()
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels,
               cmap=matplotlib.colors.ListedColormap(colors))
    plt.title(title)
    plt.show()

#-----------------------------------------------------------------
#PCA
'''
pca=decomposition.PCA(n_components=3)
pca.fit(dataset)
datasetPCA=pca.transform(dataset_scaled)
print(len(datasetPCA[1,:]))
print(len(datasetPCA[:,1]))

pca1=decomposition.PCA(n_components=30)
pca1.fit(dataset_scaled)
datasetPCA1=pca.transform(dataset_scaled)

#Scatter dos pontos obtidos pelo PCA
 colors = ['red','green','blue','purple','yellow','pink']
 fig = plt.figure(1, figsize=(4, 3))
 plt.clf()
 ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
 ax.scatter(datasetPCA[:, 0], datasetPCA[:, 1], datasetPCA[:, 2], c=labels_multi, cmap=matplotlib.colors.ListedColormap(colors))
 plt.title('LDA')
 plt.show()

'''
#-----------------------------------------------------------------
#LDA

lda = LinearDiscriminantAnalysis(n_components=3)
lda_components = lda.fit(dataset_scaled, labels_multi).transform(dataset_scaled)

lda_bin = LinearDiscriminantAnalysis(n_components=3)
lda_components_bin = lda_bin.fit(dataset_scaled, labels_bin).transform(dataset_scaled)

colors = ['red','green','blue','purple','yellow','pink']
colorsBin = ['yellow','pink']

scatter3d(lda_components,labels_multi, 'Multiclass Problem-LDA', colors);
scatter3d(lda_components_bin,labels_bin, 'Binary Problem-LDA', colors);

# plt.figure()
# fig = plt.figure(1, figsize=(4, 3))
# plt.clf()
# ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
# ax.scatter(lda_components[:, 0], lda_components[:, 1], lda_components[:, 2], c=labels_multi,
#            cmap=matplotlib.colors.ListedColormap(colors))
# plt.title('Multiclass Problem-LDA')
# plt.show()
#
# plt.figure()
# fig = plt.figure(1, figsize=(4, 3))
# plt.clf()
# ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
# ax.scatter(lda_components_bin[:, 0], lda_components_bin[:, 1], lda_components_bin[:, 2], c=labels_bin,
#            cmap=matplotlib.colorsBin.ListedColormap(colorsBin))
# plt.title('Binary Problem-LDA')
# plt.show()
#
#
