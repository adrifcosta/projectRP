from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
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

labels_bin_names=np.array(['Not Walking', 'Walking'])
labels_bin_names=labels_bin_names.transpose()

labels_multi_names=np.array(['Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Standing', 'Laying'])
labels_multi_names=labels_multi_names.transpose()


# SABER QUANTOS LABELS É QUE HÁ DE CADA TIPO
unique, counts = np.unique(labels_multi, return_counts=True)
print(dict(zip(unique, counts)))

labels_bin=np.zeros(len(labels_multi));

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
mean_corr=np.zeros(len(matrix_corr[:,0]))

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
print(len(mean_corr_cm))
print(mean_corr_cm)
print(features_names_cm)
reduced_data_cm = np.zeros((len(dataset_scaled[:,1]),len(mean_corr_cm)))
i = 0
for k in all_indexs:
    reduced_data_cm[:,i]=dataset_scaled[:,k]
    i=i+1




#reduced_data_cm dados que resultaram da redução por interpretação dos coeficientes de correlação
#-----------------------------------------------------------------
#Kruskall Wallis
#Este método não está a apresentar os resultados que pretendemos
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
#decidimos então correr o método em matlab obtendo um ficheiro com o rank de features

features_kw_multi=pd.read_csv('rank_bin_kw.txt', delim_whitespace=True,header=None)
features_kw_bin=pd.read_csv('rank_multi_kw.txt', delim_whitespace=True,header=None)

#print(features_kw_multi)
#print(features_kw_bin)
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
#---------------------------------------------
#SVM
'''
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(dataset_scaled, labels_multi)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(dataset_scaled)
print(X_new.shape)
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

colors = ['red','green','blue','purple','yellow','pink']
colorsBin = ['yellow','pink']

def scatter3d(data,labels, title, colors,legend):
    plt.figure()
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels,
               cmap=matplotlib.colors.ListedColormap(colors))
    plt.title(title)
    plt.show()

def plot2d(data,labels, title, colors):
    plt.figure()
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.scatter(data[:, 0], data[:, 1], c=labels,
               cmap=matplotlib.colors.ListedColormap(colors))
    plt.title(title)
    plt.show()


#-----------------------------------------------------------------
#PCA
#'''
pca=decomposition.PCA(n_components=3)
pca.fit(dataset_scaled)
datasetPCA=pca.transform(dataset_scaled)

pca2d=decomposition.PCA(n_components=2)
pca2d.fit(dataset_scaled)
datasetPCA2d=pca2d.transform(dataset_scaled)

pca1=decomposition.PCA(n_components=30)
pca1.fit(dataset_scaled)
datasetPCA1=pca.transform(dataset_scaled)

#Scatter dos pontos obtidos pelo PCA
#scatter3d(datasetPCA,labels_multi, 'Multiclass Problem - PCA', colors, labels_multi_names)
scatter3d(datasetPCA,labels_bin, 'Binary Problem - PCA', colorsBin, labels_bin_names)

plot2d(datasetPCA2d,labels_multi, 'Multiclass Problem - PCA', colors)
plot2d(datasetPCA2d,labels_bin, 'Binary Problem - PCA', colorsBin)

#'''
#-----------------------------------------------------------------
#LDA
#'''
lda = LinearDiscriminantAnalysis(n_components=3)
lda_components = lda.fit(dataset_scaled, labels_multi).transform(dataset_scaled)


lda1 = LinearDiscriminantAnalysis(n_components=3)
lda_components_bin = lda1.fit(dataset_scaled, labels_bin).transform(dataset_scaled)

print(lda_components_bin.shape)

plot2d(lda_components,labels_multi, 'Multiclass Problem - LDA', colors)
plot2d(lda_components,labels_bin, 'Binary Problem - LDA', colorsBin)

scatter3d(lda_components,labels_multi, 'Multiclass Problem - LDA', colors);

#'''

#------------------------------------------
# FISHER LDA

def fisher(class1, class2):
    #class1, class2 = read_data()
    mean1=np.mean(class1, axis=0)
    mean2=np.mean(class2, axis=0)

    #calculate variance within class
    Sw=np.dot((class1-mean1).T, (class1-mean1))+np.dot((class2-mean2).T, (class2-mean2))
    print(Sw.shape)
    #w, eig_vecs = np.linalg.eig(np.linalg.inv(Sw).dot(S))
    #calculate weights which maximize linear separation
    w=np.dot(np.linalg.inv(Sw), (mean2-mean1))
    print(w)
    print("vector of max weights", w)
    #projection of classes on 1D space
    plt.plot(np.dot(class1, w), [0]*class1.shape[0], "bo", label="0")
    plt.plot(np.dot(class2, w), [0]*class2.shape[0], "go", label="1")
    plt.legend()
    plt.show()

