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
from collections import Counter

features_names=pd.read_csv('features.txt', delim_whitespace=True,header=None);
labels_multi=pd.read_csv('y_train.txt', header=None);
dataset=pd.read_csv('X_train.txt',delim_whitespace=True,  header=None);
dataset=dataset.as_matrix();




labels_multi=labels_multi.as_matrix();

#print(labels_multi)
unique, counts = np.unique(labels_multi, return_counts=True)
print(dict(zip(unique, counts)))

labels_bin=np.zeros(len(labels_multi));
for i in range(0, len(labels_multi)):
    if labels_multi[i]<4:
        labels_bin[i]=1;
    else:
        labels_bin[i] = 0;

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
features_names=features_names.as_matrix()[:,1];


#Normalização dos dados
#dataset_scaled tem média nula e desvio padrão unitário
dataset_scaled=preprocessing.scale(dataset);


#SELEÇAO DE FEATURES
#-----------------------------------------------------------------
#Matriz de correlação
#corrcoef() usa estrutura de dados em que as features estão por linhas
matrix_corr=np.corrcoef(dataset_scaled.transpose());
mean_corr=labels_bin=np.zeros(len(matrix_corr[:,0]));

for i in range (0, len(matrix_corr[:,0])):
    mean_corr[i]=np.mean(matrix_corr[i,:]);
mean_corr=np.absolute(mean_corr);


mean_corr_cm=[];
features_names_cm=[];
all_indexs=[];

for j in range (0, len(mean_corr)):
    if (mean_corr[j]<0.03):
        mean_corr_cm.append(mean_corr[j])
        features_names_cm.append(features_names[j])
        all_indexs.append(j)

reduced_data_cm = np.zeros((len(dataset_scaled[:,1]),len(mean_corr_cm)));
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
    labels_bin = np.squeeze(labels_bin)
    labels_multi=np.squeeze(labels_multi)
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
#-----------------------------------------------------------------
#PCA
#fazer transposto

pca=decomposition.PCA(n_components=3)
pca.fit(dataset)
datasetPCA=pca.transform(dataset_scaled)
#print(len(datasetPCA[1,:]))
#print(len(datasetPCA[:,1]))

#Scatter dos pontos obtidos pelo PCA
colors = ['red','green','blue','purple','yellow','pink']
fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
ax.scatter(datasetPCA[:, 0], datasetPCA[:, 1], datasetPCA[:, 2], c=labels_multi, cmap=matplotlib.colors.ListedColormap(colors))
plt.show()

def fit(self):
    # Function estimates the LDA parameters
    def estimate_params(data):
        # group data by label column
        grouped = data.groupby(self.data.ix[:,self.labelcol])

        # calculate means for each class
        means = {}
        for c in self.classes:
            means[c] = np.array(self.drop_col(self.classwise[c], self.labelcol).mean(axis = 0))

        # calculate the overall mean of all the data
        overall_mean = np.array(self.drop_col(data, self.labelcol).mean(axis = 0))

        # calculate between class covariance matrix
        # S_B = \sigma{N_i (m_i - m) (m_i - m).T}
        S_B = np.zeros((data.shape[1] - 1, data.shape[1] - 1))
        for c in means.keys():
            S_B += np.multiply(len(self.classwise[c]),
                               np.outer((means[c] - overall_mean),
                                        (means[c] - overall_mean)))

        # calculate within class covariance matrix
        # S_W = \sigma{S_i}
        # S_i = \sigma{(x - m_i) (x - m_i).T}
        S_W = np.zeros(S_B.shape)
        for c in self.classes:
            tmp = np.subtract(self.drop_col(self.classwise[c], self.labelcol).T, np.expand_dims(means[c], axis=1))
            S_W = np.add(np.dot(tmp, tmp.T), S_W)

        # objective : find eigenvalue, eigenvector pairs for inv(S_W).S_B
        mat = np.dot(np.linalg.pinv(S_W), S_B)
        eigvals, eigvecs = np.linalg.eig(mat)
        eiglist = [(eigvals[i], eigvecs[:, i]) for i in range(len(eigvals))]

        # sort the eigvals in decreasing order
        eiglist = sorted(eiglist, key = lambda x : x[0], reverse = True)

        # take the first num_dims eigvectors
        w = np.array([eiglist[i][1] for i in range(self.num_dims)])

        self.w = w
        self.means = means
        return

    # estimate the LDA parameters
    estimate_params(dataset)

