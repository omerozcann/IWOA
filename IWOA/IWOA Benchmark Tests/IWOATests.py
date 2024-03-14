from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
import copy
import math
import numpy as np
import random
import pandas as pd
import sys
import functools
import numpy as np
import sklearn.metrics
import sklearn.datasets
import sklearn.model_selection
import scipy.special
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import pyswarms as ps
from mealpy.swarm_based.WOA import OriginalWOA
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from scipy.io import arff
from matplotlib import pyplot
from numpy import arange
from numpy import meshgrid
from sklearn.decomposition import PCA
from numpy.linalg import norm
from pycm import ConfusionMatrix
import warnings
import statistics
import os
warnings.filterwarnings('ignore')
###################################### load the dataset ###############


# from google.colab import drive
# drive.mount('/content/drive/')
# base='/content/drive/MyDrive/ColabNotebooks/Data'
# dataset = pd.read_csv(base+'/diabetes.csv')



#######################################################################
########################Global variables###############################    
global y_pred,y_true,bestacc,bestpre,bestrec,bestf1,bestloss,bestcm,besttpr,besttnr,bestfpr,bestfnr,bestaccval
def clearbests():
    global bestacc,bestpre,bestrec,bestf1,bestloss,bestcm,besttpr,besttnr,bestfpr,bestfnr,bestaccval
    bestloss=sys.maxsize
    bestacc=0
    bestpre=0
    bestrec=0
    bestf1=0
    besttpr=0
    besttnr=0
    bestfpr=0
    bestfnr=0
    bestaccval=0
############################### Initialize ANN ########################
def initialize_ann(shape):
    model = Sequential()
    for i in range(len(shape)-1):   
        if i==0:
            model.add(Dense(shape[0], input_shape=(shape[0],),activation='relu'))
        else:
            model.add(Dense(shape[i], activation='relu'))
    model.add(Dense(shape[len(shape)-1], activation='softmax'))
    model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
    model.summary()
    return model
#######################################################################
################ weightstovector ######################################
def get_weightbias_vector(model):
    weights=[]
    bias=[]
    wshape=[]
    bshape=[]
    for layer in model.layers:        
        weights+=list(layer.get_weights()[0].flatten())
        bias+=list(layer.get_weights()[1].flatten())        
        wshape.append(layer.get_weights()[0].shape)
        bshape.append(layer.get_weights()[1].shape)
    vector=[]
    vector+=weights
    vector+=bias
    return vector,wshape,bshape
#######################################################################
################ vectortoweights ######################################
def vector_to_weights(vector,wshape,bshape):
    wcount=0
    bcount=0
    for shape in wshape:
        wcount+=shape[0]*shape[1]
    for shape in bshape:
        bcount+=shape[0]
    ws=vector[0:wcount]
    bs=vector[wcount:wcount+bcount]
    wcount=0
    bcount=0
    weights=[]
    bias=[]
    for shape in wshape:
        w=np.reshape(ws[wcount:wcount+shape[0]*shape[1]],shape)
        wcount+=shape[0]*shape[1]
        weights.append(w)
    for shape in bshape:
        b=np.reshape(bs[bcount:bcount+shape[0]],shape)      
        bcount+=shape[0]
        bias.append(b)        
    return weights,bias
#######################################################################
################ objective function ###################################
def objective_function(params): 
    global y_pred,y_true,bestacc,bestpre,bestrec,bestf1,bestloss,bestcm,besttpr,besttnr,bestfpr,bestfnr,bestaccval
    acc=0
    pre=0
    rec=0
    f1=0
    TPR=0
    TNR=0
    FPR=0
    FNR=0
    global model
    
    weights,bias=vector_to_weights(params, wshape, bshape)       
    set_weightsbias(weights,bias) 
    loss,acc=model.evaluate(X, yb, verbose=0)    
    y_pred=model.predict(X_test, verbose=0)
    y_pred=np.argmax(y_pred, axis=1)
    y_true=np.argmax(ytb, axis=1)
    accval=accuracy_score(y_true, y_pred) 
    pre=precision_score(y_true, y_pred,average='macro') 
    rec=recall_score(y_true, y_pred,average='macro') 
    f1=f1_score(y_true, y_pred,average='macro') 
    cm=ConfusionMatrix(actual_vector=y_true, predict_vector=y_pred)
    TPR=cm.TPR_Macro
    TNR=cm.TNR_Macro
    FPR=cm.FPR_Macro
    FNR=cm.FNR_Macro
    
    if acc>bestacc:       
        bestacc=acc 
        bestcm=cm
    if loss<bestloss:
        bestloss=loss
    if TPR>besttpr:
        besttpr=TPR
    if TNR>besttnr:
        besttnr=TNR
    if FPR<bestfpr:
        bestfpr=FPR
    if FNR<bestfnr:
        bestfnr=FNR 
    if pre>bestpre:
        bestpre=pre
    if rec>bestrec:
        bestrec=rec
    if f1>bestf1:
        bestf1=f1   
    if accval>bestaccval:
        bestaccval=accval
    return loss
######################################################################
####################### set weights and bias #########################
def set_weightsbias(weights,bias):
    wid=0
    for layer in model.layers:       
        layer.set_weights([weights[wid],bias[wid]])       
        wid+=1        
######################################################################
####################### whale #########################
# whale class
class whale:
    def __init__(self, fitness, dim, minx, maxx, seed):
        self.rnd = random.Random(seed)
        self.position = [0.0 for i in range(dim)]
 
        for i in range(dim):
            self.position[i] = ((maxx - minx) * self.rnd.random() + minx)
 
        self.fitness = fitness(self.position)  # curr fitness
 
 
# whale optimization algorithm(WOA)
def woa(fitness, max_iter, n, dim, minx, maxx):
    rnd = random.Random(0)
 
    # create n random whales
    whalePopulation = [whale(fitness, dim, minx, maxx, i) for i in range(n)]
 
    # compute the value of best_position and best_fitness in the whale Population
    Xbest = [0.0 for i in range(dim)]
    Fbest = sys.float_info.max
 
    for i in range(n):  # check each whale
        if whalePopulation[i].fitness < Fbest:
            Fbest = whalePopulation[i].fitness
            Xbest = copy.copy(whalePopulation[i].position)
 
    # main loop of woa
    Iter = 0
    while Iter < max_iter:
 
        # after every 10 iterations
        # print iteration number and best fitness value so far
        # if Iter % 10 == 0 and Iter > 1:
        print("Epoch = " + str(Iter) + " MSE = %.4f" % Fbest)
        print("Acc= %f" %(bestacc))
        # linearly decreased from 2 to 0
        a = 2 * (1 - Iter / max_iter)
        a2=-1+Iter*((-1)/max_iter)
 
        for i in range(n):
            A = 2 * a * rnd.random() - a
            C = 2 * rnd.random()
            b = 1
            l = (a2-1)*rnd.random()+1;
            p = rnd.random()
 
            D = [0.0 for i in range(dim)]
            D1 = [0.0 for i in range(dim)]
            Xnew = [0.0 for i in range(dim)]
            Xrand = [0.0 for i in range(dim)]
            if p < 0.5:
                if abs(A) > 1:
                    for j in range(dim):
                        D[j] = abs(C * Xbest[j] - whalePopulation[i].position[j])
                        Xnew[j] = Xbest[j] - A * D[j]
                else:
                    p = random.randint(0, n - 1)
                    while (p == i):
                        p = random.randint(0, n - 1)
 
                    Xrand = whalePopulation[p].position
 
                    for j in range(dim):
                        D[j] = abs(C * Xrand[j] - whalePopulation[i].position[j])
                        Xnew[j] = Xrand[j] - A * D[j]
            else:                
                for j in range(dim):
                    D1[j] = abs(Xbest[j] - whalePopulation[i].position[j])
                    Xnew[j] = D1[j] * math.exp(b * l) * math.cos(2 * math.pi * l) + Xbest[j]
           #Eğer yeni balina Xbest e %90 dan daha fazla benziyorsa dairesel sinusoidal aramaya geç         
                if cousine(Xbest,Xnew)>0.9:                
                    for j in range(dim):
                        D1[j] = abs(Xbest[j] - whalePopulation[i].position[j])
                        Xnew[j] = D1[j] * math.exp(b * l) * math.cos(2 * math.pi * l)*(np.sin(j * a)+A) + Xbest[j]                    
                    
            for j in range(dim):
                whalePopulation[i].position[j] = Xnew[j]
 
        for i in range(n):
            # if Xnew < minx OR Xnew > maxx
            # then clip it
            for j in range(dim):
                whalePopulation[i].position[j] = max(whalePopulation[i].position[j], minx)
                whalePopulation[i].position[j] = min(whalePopulation[i].position[j], maxx)
 
            whalePopulation[i].fitness = fitness(whalePopulation[i].position)
            # cousine(whalePopulation[i].position,Xbest)
            if (whalePopulation[i].fitness < Fbest):                
                Xbest = copy.copy(whalePopulation[i].position)
                Fbest = whalePopulation[i].fitness            
     
        Iter += 1
    # end-while
 
    # returning the best solution
    return Xbest
       
######################################################################
###########################Cousine Similarity#########################
def cousine(A,B):
    # compute cosine similarity
    cosine = np.dot(A,B)/(norm(A)*norm(B))
    return cosine
#####################################################################
######################### start #######################################





fitness = objective_function
pop_size =20
epoch = 250
opt="IWOA"
base='.\\Benchmark Datasets'
datasets=os.listdir(base)
for ds in datasets:
    clearbests()
    acclist=[]
    accvallist=[]
    datasetname=ds
    dataset = pd.read_csv(base+'\\'+datasetname)
    print(datasetname)    
    feature_count=dataset.shape[1]-1    
    data=np.array(dataset.iloc[:,:feature_count])
    target=np.array(dataset.iloc[:,[feature_count]])    
    num_classes = len(np.unique(target))
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    X, X_test, y, y_test = train_test_split(data, target, test_size = 0.34, random_state = 25)
    
    if len(np.unique(target))<3:
        yb=[]
        ytb=[]
        for i in range(len(y)):
            b=[]
            if y[i]==1:
                b.append(0)
                b.append(1)
            if y[i]==2:
                b.append(1)
                b.append(0)
            yb.append(b)
        yb=np.array(yb)
        for i in range(len(y_test)):
            b=[]
            if y_test[i]==1:
                b.append(0)
                b.append(1)
            if y_test[i]==2:
                b.append(1)
                b.append(0)
            ytb.append(b)
        ytb=np.array(ytb)
    else:
        binarizer1 = LabelBinarizer(neg_label=0, pos_label=1)
        binarizer1.fit(y)
        yb=binarizer1.transform(y)
        binarizer2 = LabelBinarizer(neg_label=0, pos_label=1)
        binarizer2.fit(y_test)
        ytb=binarizer2.transform(y_test)
        
    shape=[X.shape[1],2*feature_count+1,num_classes]
    model=initialize_ann(shape)
    dim,wshape,bshape=get_weightbias_vector(model)
    
    for i in range(10): 
        del model    
        model=initialize_ann(shape)
        f = open(opt+str(datasetname)+".txt", "a")
        best_position = woa(fitness, epoch, pop_size, len(dim), -1, 1)
        print(opt)    
        print(datasetname)
        print(i)
        print("Loss %f" %(bestloss))
        print("ACC %f" %(bestacc))   
        print("ACC val %f" %(bestaccval))
        # print("Precision %f" %(bestpre)) 
        # print("Recall %f" %(bestrec)) 
        # print("F1 %f" %(bestf1))
        # print("TPR %f" %(besttpr))
        # print("TNR %f" %(besttnr))
        # print("FPR %f" %(bestfpr))
        # print("FNR %f" %(bestfnr))
        # print("All Metrics")
        # print(bestcm)
        acclist.append(bestacc)
        accvallist.append(bestaccval)
        f.write("\n###############################################################")
        f.write(opt)
        f.write("\nLoss:"+str(bestloss)+"\n")
        f.write("ACC:"+str(bestacc)+"\n")
        f.write("ACC val:"+str(bestaccval)+"\n")
        f.write("Pre:"+str(bestpre)+"\n")
        f.write("Rec:"+str(bestrec)+"\n")
        f.write("F1:"+str(bestf1)+"\n")
        f.write("Tpr:"+str(besttpr)+"\n")
        f.write("Tnr:"+str(besttnr)+"\n")
        f.write("Fpr:"+str(bestfpr)+"\n")
        f.write("Fnr:"+str(bestfnr)+"\n")
        f.write("All Metrics")
        f.write(str(bestcm)+"\n")
        f.write("###############################################################\n")
        f.close()
    print("Std:%f" %(statistics.stdev(acclist)))
    f = open(opt+str(datasetname)+".txt", "a")
    f.write("\nMean:%f" %(statistics.mean(acclist)))
    f.write("\nStd:%f" %(statistics.stdev(acclist)))
    f.write("\nBest:%f" %(np.max(acclist)))
    f.write("\nMeanv:%f" %(statistics.mean(accvallist)))
    f.write("\nStdv:%f" %(statistics.stdev(accvallist)))
    f.write("\nBestv:%f" %(np.max(accvallist)))
    f.close()
print("Tamamlandı")