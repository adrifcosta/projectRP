from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
import scipy.stats
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
import math
from sklearn.decomposition import PCA
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression


features_names=pd.read_csv('features.txt', delim_whitespace=True,header=None)
labels_multi=pd.read_csv('y_train.txt', header=None)
dataset=pd.read_csv('X_train.txt',delim_whitespace=True,  header=None) # o data set é 7352 linhas por 561 colunas
dataset=dataset.as_matrix()
dataset_test=pd.read_csv('X_test.txt',delim_whitespace=True,  header=None)
labels_test=pd.read_csv('y_test.txt', header=None)
dataset_test = dataset_test.as_matrix() # o data set teste tem 2947 linhas por 561 colunas

labels_multi=labels_multi.as_matrix()
labels_test= labels_test.as_matrix()
labels_bin=np.zeros(len(labels_multi))

labels_bin = np.squeeze(labels_bin)
labels_multi = np.squeeze(labels_multi)
labels_test= np.squeeze(labels_test)

labels_bin_names=np.array(['Not Walking', 'Walking'])
labels_bin_names=labels_bin_names.transpose()

labels_multi_names=np.array(['Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Standing', 'Laying'])
labels_multi_names=labels_multi_names.transpose()


# SABER QUANTOS LABELS É QUE HÁ DE CADA TIPO
unique, counts = np.unique(labels_multi, return_counts=True)
#print(dict(zip(unique, counts)))

labels_bin=np.zeros(len(labels_multi));

for i in range(0, len(labels_multi)):
    if labels_multi[i]<4:
        labels_bin[i]=1
    else:
        labels_bin[i] = 0

labels_bin_test = np.zeros(len(labels_test));

for i in range(0, len(labels_test)):
    if labels_test[i] < 4:
        labels_bin_test[i] = 1
    else:
        labels_bin_test[i] = 0


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

def distance_min_classification(dataset_scaled,labels_bin, dataset_test):

    indices_0=np.empty(0)
    indices_1=np.empty(0)
    for i in range(0, len(labels_bin)):
        if(labels_bin[i]==0):
           indices_0 =np.append(indices_0,i)
        else:
            indices_1=np.append(indices_1,i)

    indices_0=indices_0.astype(int)
    indices_1 = indices_1.astype(int)

    data_0= dataset_scaled[indices_0,:]
    data_1=dataset_scaled[indices_1,:]

    mean_labels_1=np.empty(0)
    mean_labels_0=np.empty(0)

    for i in range(0, len(dataset_scaled[0])):
        mean_labels_0 = np.append(mean_labels_0,np.mean((data_0[:,i])))
        mean_labels_1 = np.append(mean_labels_1, np.mean((data_1[:,i])))

    labels_vector=np.empty(0)
    for i in range(0,len(dataset_test[:,0])):
        dist0=np.sqrt(sum(np.power(np.subtract(dataset_test[i,:],mean_labels_0),2)))
        dist1=np.sqrt(sum(np.power(np.subtract(dataset_test[i,:], mean_labels_1), 2)))
        #print(dist0,dist1)
        if (dist0<dist1):
            labels_vector = np.append(labels_vector,[0],axis=0)
        else:
            labels_vector = np.append(labels_vector,[1],axis=0)
    return labels_vector

labels_classification=distance_min_classification(dataset_scaled,labels_bin,dataset_test)
#print(confusion_matrix(labels_bin_test,labels_classification))

#SELEÇAO DE FEATURES
#-----------------------------------------------------------------
def features_selection(dataset_scaled,features_names, labels, option):
    if (option==1):
        # Matriz de correlação
        # corrcoef() usa estrutura de dados em que as features estão por linhas
        matrix_corr=np.corrcoef(dataset_scaled.transpose())
        mean_corr=np.zeros(len(matrix_corr[:,0]))

        for i in range (0, len(matrix_corr[:,0])):
            mean_corr[i]=np.mean(matrix_corr[i,:])
        mean_corr=np.absolute(mean_corr)

        mean_corr_cm=np.empty(0)
        features_names_cm=np.empty(0)
        all_indexs=np.empty(0)

        for j in range (0, len(mean_corr)):
            if (mean_corr[j]<0.03):
                mean_corr_cm=np.append(mean_corr_cm,mean_corr[j])
                features_names_cm= np.append(features_names_cm,features_names[j])
                all_indexs =np.append(all_indexs,j)
        #print(len(mean_corr_cm))
        #print(mean_corr_cm.shape)
        #print(features_names_cm.shape)
        reduced_data_cm = np.zeros((len(dataset_scaled[:,1]),len(mean_corr_cm)))
        i = 0
        for k in all_indexs:
            reduced_data_cm[:,i]=dataset_scaled[:,k]
            i=i+1
        # reduced_data_cm dados que resultaram da redução por interpretação dos coeficientes de correlação
        features_reduce=reduced_data_cm
        features_name_reduce=features_names_cm
    elif (option==2):
        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(dataset_scaled, labels)
        model = SelectFromModel(lsvc, prefit=True)
        features_reduce = model.transform(dataset_scaled)
        #print(X_new.shape)
        #print(model.get_support())  # ESTE METODO DIZ-NOS QUAIS FEATURES FORAM ESCOLHIDA E QUAIS NAO (TRUE SE FORAM, FALSE SE NAO FORAM)
        features_name_reduce=features_names[model.get_support()] #FICAMOS SO COM OS NOMES DAS FEATURES SELECIONADAS
    elif(option==3):
        # Feature Importance using Extra Trees Classifier
        labels_multi = np.squeeze(labels)
        model = ExtraTreesClassifier()
        model.fit(dataset_scaled, labels)
        print(model.feature_importances_.shape)
        features_importances = model.feature_importances_
        ind = np.argsort(features_importances)[::-1]  # RETORNA OS INDICES DAS FEATURES PELA ORDEM DE IMPORTANCIA DECRESCENTE
        features_reduce = features_importances[ind[0:39]]  # ESCOLHI POR EXEMPLO AS 40 MELHORES FEATURES
        features_name_reduce = features_names[ind[0:39]]  # NOMES DESSAS 40 FEATURES ESCOLHIDAS
    elif(option==4):
        ##LASOV
        clf = LassoCV()
        sfm = SelectFromModel(clf)
        sfm.fit(dataset_scaled, labels)
        features_reduce=sfm.transform(dataset_scaled)
        #n_features = features_reduce.shape[1]
        features_name_reduce=features_names[sfm.get_support()]
    return features_name_reduce,features_reduce


#features_selection(dataset_scaled,features_names,labels_bin,2)

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
def ourPCA(data,dataY, comp):
    pca = decomposition.PCA(n_components=comp)
    pca.fit(data)
    #Variance (% cumulative) explained by the principal components
    print(np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4) * 100))
    #pca.components_ The components are sorted by explained_variance_
    #a 1a componente é aquela que tem mais variance_ratio
    dataReduced = pca.transform(data)

    n = len(dataReduced)
    kf_10 = cross_validation.KFold(n, n_folds=10, shuffle=True, random_state=2)
    regr = LinearRegression()
    mse = []

    score = -1 * cross_validation.cross_val_score(regr, np.ones((n, 1)), dataY.ravel(), cv=kf_10,
                                                  scoring='mean_squared_error').mean()
    mse.append(score)

    for i in np.arange(1, comp+1):
        score = -1 * cross_validation.cross_val_score(regr, dataReduced[:, :i], dataY.ravel(), cv=kf_10,
                                                      scoring='mean_squared_error').mean()
        mse.append(score)
    plt.figure()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(mse, '-v')
    ax2.plot(np.arange(1, comp+1), mse[1:comp+1], '-v')
    ax2.set_title('Intercept excluded from plot')

    for ax in fig.axes:
        ax.set_xlabel('Number of principal components in regression')
        ax.set_ylabel('MSE')
        ax.set_xlim((-0.2, comp+0.2))
    return dataReduced

datasetPCA=ourPCA(dataset_scaled,labels_bin,3)

datasetPCA2d=ourPCA(dataset_scaled,labels_bin,2)

datasetPCA30d=ourPCA(dataset_scaled,labels_bin,30)


#Scatter dos pontos obtidos pelo PCA
#scatter3d(datasetPCA,labels_multi, 'Multiclass Problem - PCA', colors, labels_multi_names)
scatter3d(datasetPCA,labels_bin, 'Binary Problem - PCA', colorsBin, labels_bin_names)

plot2d(datasetPCA2d,labels_multi, 'Multiclass Problem - PCA', colors)
plot2d(datasetPCA2d,labels_bin, 'Binary Problem - PCA', colorsBin)



#'''
#-----------------------------------------------------------------
#LDA
'''
lda = LinearDiscriminantAnalysis(n_components=3)
lda_components = lda.fit(dataset_scaled, labels_multi).transform(dataset_scaled)


lda1 = LinearDiscriminantAnalysis(n_components=3)
lda_components_bin = lda1.fit(dataset_scaled, labels_bin).transform(dataset_scaled)

print(lda_components_bin.shape)

plot2d(lda_components,labels_multi, 'Multiclass Problem - LDA', colors)
plot2d(lda_components,labels_bin, 'Binary Problem - LDA', colorsBin)

scatter3d(lda_components,labels_multi, 'Multiclass Problem - LDA', colors);

'''

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

