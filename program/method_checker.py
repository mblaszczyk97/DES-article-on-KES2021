import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import imblearn
import itertools
import random
import lazypredict
import os
import glob
import statistics
import heapq
from m_plots import plots as mplot
from irregular_set import irregular_set as irrs
from csv import writer
from sklearn.datasets import fetch_openml

from collections import Counter
from matplotlib import pyplot
from numpy import where
from my_method import my_method

from lazypredict.Supervised import LazyClassifier


from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import AllKNN
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.metrics import geometric_mean_score
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
#from sklearn import cross_validation

from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture
from imblearn.under_sampling import ClusterCentroids
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import Perceptron
from deslib.dcs import Rank

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import BaggingClassifier

#importing DCS techniques from DESlib
from deslib.dcs.ola import OLA
from deslib.dcs.mla import MLA
from deslib.dcs.a_priori import APriori
from deslib.dcs.mcb import MCB
from deslib.dcs.a_priori import APriori
#import DES techniques from DESlib
from deslib.des.des_p import DESP
from deslib.des.knora_u import KNORAU
from deslib.des.knora_e import KNORAE
from deslib.des.meta_des import METADES

from deslib.static import StackedClassifier

#------------
#EN(English)-
#------------
#INSTRUCTION:
#
#To classificate by the simplest way possible with printing g-mean score to terminal use:
#set=irrs("DATA_SET.csv")   - IMPORTANT!!! it has to be: ~\input\bases
#X=set.X
#y=set.y
#k_fold(X, y, "USED BALANCING TECHNIC", K NEAREST EXAMPLES ON VALIDATION SET)
#
#if you want to display figures while computing please uncomment line: "gmeans_plot.show()" in k_fold definition
#
#ADVANCED OPTIONS (NOT RECOMENDED):
#read_all_bases(1)              - Read and test all data sets in: ~\input\bases
#write_to_file()                - Write g-mean scores to .csv file
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#-----------
#PL(POLISH)-
#-----------
#INSTRUCKJA:
#
#Aby przeprowadzić klasyfikacje w najprostszy sposób z wypisaniem dokładności do terminalu należy użyć:
#set=irrs("ZBIÓR_DANYCH.csv")   - WAŻNE!!! musi on być zawarty w folderze: ~\input\bases
#X=set.X
#y=set.y
#k_fold(X, y, "UŻYWANA TECHNIKA BALANSUJĄCA", ILOSC K NAJBLIŻSZYCH PRÓBEK NA ZBIORZE WALIDACYJNYM)
#
#jeżeli chcemy dodatkowy wyświetlić dane na wykresach to wystarczy odkomentować linię "gmeans_plot.show()" w definicji metody k_fold
#
#OPCJE ZAAWANSOWANE (NIEZALECANE):
#read_all_bases(1)              - Wczytanie i przetestowanie wszystkich zbiorów danych w folderze: ~\input\bases
#write_to_file()                - Nadpisanie wyników do pliku .csv


def auc_score_predict(model, X_train, y_train):
    """Predict AUC for model and X, y"""
    return roc_auc_score(y_train.values.ravel(), model.predict_proba(X_train)[:, 1])

def auc_score_predict_multiclass(model, X_train, y_train):
    """Predict AUC in multiclassification for model and X, y"""
    return roc_auc_score(y_train.values.ravel(), model.predict_proba(X_train)[:, 1], multi_class='ovr')

def methods_modeling(X_train, y_train, X_test, y_test, random_seed, balancing_scheme, number_of_k=10):
    """
     Parameters
        ----------
        X_train : dataframe
            database without class attribute of the training values

        y_train : dataframe
            database with only class attribute of the training values

        X_test : dataframe
            database without class attribute of the testing values

        y_test : dataframe
            database with only class attribute of the testing values

        random_seed : int
            random int neede for some classification methods

        balancing_scheme : str, optional
            used only in previous version to show on the plot
    """

    rng=random_seed
    X_train, X_dsel, y_train, y_dsel = train_test_split(X_train, y_train, test_size=0.33, random_state=rng)
    #X_test, X_dsel, y_test, y_dsel = train_test_split(X_test, y_test, test_size=0.33, random_state=rng)
    model_svc = SVC(probability=True, gamma='auto', random_state=rng).fit(X_train, y_train.values.ravel())
    model_bayes = GaussianNB().fit(X_train, y_train.values.ravel())
    model_tree = DecisionTreeClassifier(random_state=rng).fit(X_train, y_train)
    model_knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train.values.ravel())
    rf = RandomForestClassifier(n_estimators=100, random_state=rng, n_jobs=-1).fit(X_train, y_train.values.ravel())
    model_quadric = QuadraticDiscriminantAnalysis().fit(X_train, y_train.values.ravel())
    model_gaussian = GaussianNB().fit(X_train, y_train.values.ravel())
    model_mlp = MLPClassifier(alpha=1, max_iter=1000,random_state=rng).fit(X_train, y_train.values.ravel())

    #Here we prepare the classifiers we want to use
    pool_classifiers = [
                        rf,
                        model_knn,
                        model_svc,
                        model_mlp,
                        model_bayes,
                        model_tree
                        ]
    #------------------------------------------------


    knorau = KNORAU(pool_classifiers, k=number_of_k, knn_classifier=None)
    kne = KNORAE(pool_classifiers, k=number_of_k, knn_classifier=None)
    apriori = MLA(pool_classifiers, k=number_of_k, knn_classifier=None)
    ola = OLA(pool_classifiers, k=number_of_k, knn_classifier=None)
    my_method_1 = my_method(pool_classifiers, k=number_of_k, knne=False, knn_classifier=None)

    #print("DCS")
    knorau.fit(X_dsel, y_dsel.values.ravel())
    kne.fit(X_dsel, y_dsel.values.ravel())

    #print("DES")
    ola.fit(X_dsel, y_dsel.values.ravel())
    apriori.fit(X_dsel, y_dsel.values.ravel())
    my_method_1.fit(X_dsel, y_dsel.values.ravel())

   
    knora       =   knorau.score(X_test, y_test.values.ravel())
    knorae      =   kne.score(X_test, y_test.values.ravel())
    aprioris    =   apriori.score(X_test,  y_test.values.ravel())
    olas        =   ola.score(X_test,  y_test.values.ravel())
    knn         =   model_knn.score(X_test,  y_test.values.ravel())
    svc         =   model_svc.score(X_test,  y_test.values.ravel())
    bayes       =   model_bayes.score(X_test,  y_test.values.ravel())
    decisionTree=   model_tree.score(X_test,  y_test.values.ravel())
    my_method_1_score = my_method_1.score(X_test,  y_test.values.ravel())
    rf_score    =   rf.score(X_test,  y_test.values.ravel())
    model_quadric_score = model_quadric.score(X_test,  y_test.values.ravel())
    model_gaussian_score = model_gaussian.score(X_test,  y_test.values.ravel())
    model_mlp_score = model_mlp.score(X_test,  y_test.values.ravel())

    list_of_models = []
    list_of_models.append((knn,model_knn))
    list_of_models.append((svc,model_svc))
    list_of_models.append((bayes,model_bayes))
    list_of_models.append((decisionTree,model_tree))
    list_of_models.append((rf_score,rf))
    list_of_models.append((model_mlp_score,model_mlp))
    #print(heapq.nlargest(3,list_of_models))


    #brf_score   =   brf.score(X_test,  y_test.values.ravel())
    #lr_score    =   lr.score(X_test,  y_test.values.ravel())
    
    knora_gmean       =   geometric_mean_score(y_test.values.ravel(), knorau.predict(X_test))
    knorae_gmean      =   geometric_mean_score(y_test.values.ravel(), kne.predict(X_test))
    aprioris_gmean    =   geometric_mean_score(y_test.values.ravel(), apriori.predict(X_test))
    olas_gmean        =   geometric_mean_score(y_test.values.ravel(), ola.predict(X_test))
    knn_gmean         =   geometric_mean_score(y_test.values.ravel(), model_knn.predict(X_test))
    svc_gmean         =   geometric_mean_score(y_test.values.ravel(), model_svc.predict(X_test))
    bayes_gmean       =   geometric_mean_score(y_test.values.ravel(), model_bayes.predict(X_test))
    decisionTree_gmean=   geometric_mean_score(y_test.values.ravel(), model_tree.predict(X_test))
    my_method_1_gmean =   geometric_mean_score(y_test.values.ravel(), my_method_1.predict(X_test), )
    rf_gmean    =   geometric_mean_score(y_test.values.ravel(), rf.predict(X_test))
    #print(rf.predict(X_test))
    model_quadric_gmean = geometric_mean_score(y_test.values.ravel(), model_quadric.predict(X_test))
    model_gaussian_gmean = geometric_mean_score(y_test.values.ravel(), model_gaussian.predict(X_test))
    model_mlp_gmean = geometric_mean_score(y_test.values.ravel(), model_mlp.predict(X_test))
    #brf_gmean   =   geometric_mean_score(y_test.values.ravel(), brf.predict(X_test), average='micro')
    #lr_gmean    =   geometric_mean_score(y_test.values.ravel(), lr.predict(X_test), average='micro')
    #knora pass
    methods= {'KNORA-U':[0, 0, knora_gmean], 'KNORA-E':[0, 0, knorae_gmean], 'KNORA-P':[0, 0, my_method_1_gmean],
              'Decision Tree':[0, 0, decisionTree_gmean], 'MLA':[0, 0, aprioris_gmean], 
              'OLA':[0, 0, olas_gmean], 
              'KNN':[0, 0, knn_gmean], 'SVC':[0, 0, svc_gmean], 'Bayes':[0, 0, bayes_gmean],
              'rf':[rf_gmean,rf_gmean,rf_gmean], 'quadric':[model_quadric_gmean,model_quadric_gmean,model_quadric_gmean], 
              'gausianNB':[model_gaussian_gmean,model_gaussian_gmean,model_gaussian_gmean], 'MLP':[model_mlp_gmean,model_mlp_gmean,model_mlp_gmean],
              }
    import operator
    bestScore = max(methods.items(),key = lambda x:x[1]) #max function will return a (key,value) tuple of the maximum value from the dictionary
    bestScoreList =[i[0] for i in methods.items() if i[1]==bestScore[1]] #my_tuple[1] indicates maximum dictionary items value
    bestAUCScoresList = []
    print("----------------------------------------")
    for item in methods:
        for bestItem in bestScoreList:
            if item.capitalize==bestItem.capitalize:
                bestAUCScoresList.append(methods.get(item))

    maximum=0
    for item in bestAUCScoresList:
        if item[1] >= maximum:
            maximum=item[1]
            maxAUC=item
    
    bestMethod=list(methods.keys())[list(methods.values()).index(maxAUC)]

    return methods

    objects = ('KNORA-U','KNORA-E', 'Apriori', 'OLA', 'Perceptron','KNN', 'SVC', 'BAYES', 'Decision Tree')
    y_pos = np.arange(len(objects))
    performance = [knora,knorae,aprioris,olas,perceptron,knn,svc,bayes,decisionTree]
    bars = plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Procent dokładności')
    plt.title(balancing_scheme)
    for bar in bars:
        yval = bar.get_height()
        string = "{:.5f}".format(bar.get_height())
        plt.text(bar.get_x(), yval + .005, string)
    plt.show()

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


def plotter(X, y, balancing_technic):
    g1, g2, g3, plot1=k_fold(X, y, balancing_technic, use="methods_checking")
    return plot1


def k_fold(X, y, balancing_technic, use="methods_checking", number_of_k=10):
    """
     Parameters
        ----------
        X : dataframe
            database without class attribute

        y : dataframe
            database with only class attribute

        balancing_technic : str
            balancing method we want to use example:
            - "SMOTE" - SMOTE method
            - "ROS" - Random Over Sampling
            - "ADASYN" - ADASYN method
            - "ALLKNN" - All K Nearest Neighbours method
            - "RUS" - Random Under Sampling Method

        use : str, optional
            default - "method_checking"
            Other uses will be implemented in the future if needed
    """
    methods_means = {}
    methods_means_sm = {}
    methods_means_without = {}
    methods_means_ada = {}
    methods_means_ros = {}
    performance = []
    array_of_methods = []
    cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=False)
    for train_index, test_index in cv.split(X, y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if balancing_technic=="WITHOUT":
            X_train_kfold, y_train_kfold = X_train, y_train
            balancing_label=6
        elif balancing_technic=="SMOTE":
            ros = SMOTE(random_state=42, k_neighbors=4)
            X_train_kfold, y_train_kfold = ros.fit_sample(X_train, y_train)
            balancing_label=1
        elif balancing_technic=="ROS":
            ros = RandomOverSampler(random_state=42)
            X_train_kfold, y_train_kfold = ros.fit_sample(X_train, y_train)
            balancing_label=2
        elif balancing_technic=="ADASYN":
            ros = ADASYN(random_state=42, n_neighbors=4)
            X_train_kfold, y_train_kfold = ros.fit_sample(X_train, y_train)
            balancing_label=3
        elif balancing_technic=="RUS":
            ros = RandomUnderSampler(random_state=42)
            X_train_kfold, y_train_kfold = ros.fit_sample(X_train, y_train)
            balancing_label=5

        if use=="methods_checking":
            methods=methods_modeling(X_train_kfold, y_train_kfold, X_test, y_test, 7, balancing_technic, number_of_k)
            for key, value in methods.items():
                methods_means.setdefault(key, [])
                methods_means[key].append(value[2])

    if use=="methods_checking":
        for key, value in methods_means.items():
                #print(key, '->', statistics.mean(value))
                methods_means[key]=statistics.mean(value)
                performance.append(statistics.mean(value))
        v=list(methods_means.values())
        k=list(methods_means.keys())
        bestScore = max(v)
        best_method=k[v.index(bestScore)]
        print(methods_means)    
        print(best_method)
        objects = methods_means.keys()
        methods_means.pop('Decision Tree', None)
        methods_means.pop('KNN', None)
        methods_means.pop('SVC', None)
        methods_means.pop('Bayes', None)
        methods_means.pop('rf', None)
        methods_means.pop('quadric', None)
        methods_means.pop('gausianNB', None)
        methods_means.pop('MLP', None)
        gmeans_plot=mplot(objects, methods_means.values(), balancing_technic)
        gmeans_plot.save("{}{}".format(balancing_technic, "master-MammographyALL.png"))
        methods_means.pop('OLA', None)
        methods_means.pop('MLA', None)
        gmeans_plot=mplot(objects, methods_means.values(), balancing_technic)
        gmeans_plot.save("{}{}".format(balancing_technic, "article-MammographyALL.png"))
        #gmeans_plot.show()
        return best_method, bestScore, balancing_label


def classificate_from(file, balancing_technic, is_append=0):
    """
     Parameters
        ----------
        file : str
            name of the file used ex. "irregularData1.csv"

        balancing_technic : str
            balancing method we want to use example:
            - "SMOTE" - SMOTE method
            - "ROS" - Random Over Sampling
            - "ADASYN" - ADASYN method
            - "ALLKNN" - All K Nearest Neighbours method
            - "RUS" - Random Under Sampling Method

        is_append : boolean, optional
            value which represents writing to .csv file change to is_append=1 if you want to write to file.
    """
    print("Classification of: {}".format(file))
    set=irrs(file)
    X=set.X
    y=set.y
    print("read file success")
    imbRatio=set.get_imbalanced_ratio()
    attributes_number=set.get_attributes_number()
    records_number=set.get_records_number()
    balancing_label=0
    bestMethods, bestScore, balancing_label=k_fold(X, y, balancing_technic)
   
    #bestMethods, bestScore=heterogeneous(X_train_res_SMOTE, y_train_res_SMOTE, tst_X, tst_y, rng, balancing_technic)
    testing_x = np.array([imbRatio,attributes_number,records_number,bestScore,balancing_label])
    row_contents = [imbRatio,attributes_number,records_number,bestScore,balancing_label,bestMethods]
    if is_append==1:
        append_list_as_row('../input/bases/accuracy/accuracy.csv', row_contents)

    print("Att0 (Imbalance ratio): {}".format(imbRatio))
    print("Att1 (Number of attributes): {}".format(attributes_number))
    print("Att2 (Number of records): {}".format(records_number))
    print("Att3 (Balance Technic): {}".format(balancing_technic))
    print("Att4 (best): {}".format(bestScore))
    print("Class: {}".format(bestMethods))
    print("--------------------------------------------------")
    print("predicted class: ")
    #print(multiclassificate("accuracy.csv").predict(testing_x.reshape(1, -1)))
    print("--------------------------------------------------")
    return testing_x.reshape(1, -1) 

def read_all_bases(is_append):
    """For each .csv file in folder "bases" classificate using all balancing methods"""
    path = '../input/bases'
    for filename in glob.glob(os.path.join(path, '*.csv')):
        #print(filename.replace('../input/bases\\', ''))
        classificate_from(filename.replace('../input/bases\\', ''), "SMOTE",is_append)
        classificate_from(filename.replace('../input/bases\\', ''), "ROS",is_append)
        classificate_from(filename.replace('../input/bases\\', ''), "ADASYN",is_append)
        classificate_from(filename.replace('../input/bases\\', ''), "RUS",is_append)
        classificate_from(filename.replace('../input/bases\\', ''), "WITHOUT",is_append)

rng = np.random.RandomState(42)


def write_to_file():
    """If file exists write in the new row"""
    if os.stat("../input/bases/accuracy/accuracy.csv").st_size == 0:
        row_contents = ["Imbalance_ratio","Numbers_attributes","Records_attributes","Accuracy","AUC","Balancing_scheme","Class"]
        append_list_as_row('../input/bases/accuracy/accuracy.csv', row_contents)
        read_all_bases()
    else:
        os.remove("../input/bases/accuracy/accuracy.csv")
        row_contents = ["Imbalance_ratio","Numbers_attributes","Records_attributes","Accuracy","AUC","Balancing_scheme","Class"]
        append_list_as_row('../input/bases/accuracy/accuracy.csv', row_contents)
        read_all_bases()

#---------------------------------------------------------------------------------------------
#example printing information about dataset:
#set=irrs("yeast5.csv")
#print(set.get_attributes_number())
#print(set.get_imbalanced_ratio())
#print(set.get_records_number())

#---------------------------------------------------------------------------------------------
#example testing with k-fold method:
#
set=irrs("yeast5.csv")
X=set.X
y=set.y
k_fold(X, y, "ADASYN", number_of_k=15)

#---------------------------------------------------------------------------------------------
#example testing without k-fold method:
#
#X_train, X_dsel, y_train, y_dsel = train_test_split(X, y, test_size=0.5, random_state=111)
#methods=methods_modeling(X_train, y_train, X_dsel, y_dsel, 7, "SMOTE", 15)
#print(methods)

#---------------------------------------------------------------------------------------------
#example using only g-mean score:
#
#y_true = [0, 1, 0, 0, 1, 1]
#y_pred = [0, 1, 0, 0, 0, 0]
#print(geometric_mean_score(y_true, y_pred))

