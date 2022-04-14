import warnings
warnings.filterwarnings('ignore')
import os
import numpy
import cv2
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix,graycoprops
import scipy
import skimage
from skimage.morphology import disk
from sklearn.decomposition import PCA
from collections import Counter
from imblearn.over_sampling import SMOTE
import pickle
path = r'/home/aswinkumar/Desktop/project/data/ph2/PH2_dataset.txt'
#print(path)
file1 = open(path,'r')
contents = file1.readlines()
name_list = []
cd_list = []
feature = []
f3 = []
path1 = r'/home/aswinkumar/Desktop/project/data/ph2/PH2 Dataset images'  
a = 1
b = 201
F = []
for line in range(1,201):
   name = contents[line].split('||')[1]
   name_list.append(name)
   cd   = contents[line].split('||')[3]
   cd_list.append(int(cd))
   #print(name,cd)
   folder_path = path1+'/'+name.strip()
   #print(folder_path) 
   sub_folder = os.listdir(folder_path)
   
   for i in sub_folder:
      if i.endswith('Dermoscopic_Image'):
         print(i)
         subfolderpath = folder_path+'/'+i
         #print(subfolderpath)
         ssfolder = os.listdir(subfolderpath)[0]
         #print(ssfolder)
         newfolder = subfolderpath+'/'+ssfolder
         #print(newfolder)
         img1 = cv2.imread(newfolder)
         img1 = cv2.resize(img1,(767,1022))
         B = img1[:,:,0]
         #cv2.imshow('1',B)
         #cv2.waitKey(100)
         G = img1[:,:,1]
         #cv2.imshow('2',G)
         #cv2.waitKey(100)
         R = img1[:,:,2]
         #cv2.imshow('3',R)
         #cv2.waitKey(100)
         sk = B+G+R
         meanr = numpy.mean(R.flatten())
         meang = numpy.mean(G.flatten())
         meanb = numpy.mean(B.flatten())
         #print(meanr)
         #print(meang)
         #print(meanb)
         kurtr = scipy.stats.kurtosis(R.flatten())
         kurtg = scipy.stats.kurtosis(G.flatten())
         kurtb = scipy.stats.kurtosis(B.flatten())
         #print('kurt of r',kurtr) 
         #print('kurt of g',kurtg) 
         #print('kurt of b',kurtb)
         skewr = scipy.stats.skew(R.flatten())
         skewg = scipy.stats.skew(G.flatten())
         skewb = scipy.stats.skew(B.flatten())
         #print('skew of r',skewr) 
         #print('skew of g',skewg) 
         #print('skew of b',skewb)
         varr = numpy.var(R.flatten())
         varg = numpy.var(G.flatten())
         varb = numpy.var(B.flatten())
         #print('varience of r',varr)
         #print('varience of g',varg)
         #print('varience of b',varb)
         m = numpy.mean(img1)
         #print('mean of colour image',m)
         k =scipy.stats.kurtosis(sk.flatten())
         #print('kurtosis of colour image',k)
         s = scipy.stats.skew(sk.flatten())
         #print('skew of colour img',s)
         v = numpy.var(img1)
         #print('varience of colour img',v)
         hsv = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)    
         #cv2.imshow('hsv',hsv)
         #cv2.waitKey(100)
         H = hsv[:,:,0]
         S = hsv[:,:,1]
         V = hsv[:,:,2]
         #cv2.imshow('h',H)
         #cv2.waitKey(100)
         #cv2.imshow('s',S)
         #cv2.waitKey(100)
         #cv2.imshow('V',V)
         #cv2.waitKey(100)
         meanh = numpy.mean(H.flatten())
         means = numpy.mean(S.flatten())
         meanv = numpy.mean(V.flatten())
         #print('mean of h',meanh)
         #print('mean of s',means)
         #print('mean of v',meanv)
         kurth = scipy.stats.kurtosis(H.flatten())
         kurts = scipy.stats.kurtosis(S.flatten())
         kurtv = scipy.stats.kurtosis(V.flatten())
         #print('kurt of h',kurth) 
         #print('kurt of s',kurts) 
         #print('kurt of v',kurtv)
         skewh = scipy.stats.skew(H.flatten())
         skews = scipy.stats.skew(S.flatten())
         skewv = scipy.stats.skew(V.flatten())
         #print('skew of h',skewh) 
         #print('skew of s',skews) 
         #print('skew of v',skewv)
         varh = numpy.var(H.flatten())
         varss = numpy.var(S.flatten())
         varv = numpy.var(V.flatten())
         #print('varience of h',varh)
         #print('varience of s',varss)
         #print('varience of v',varv)
         lab = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)  
         #cv2.imshow('lab',lab)
         #cv2.waitKey(100)
         l = lab[:,:,0]
         a = lab[:,:,1]
         b = lab[:,:,2]
         #cv2.imshow('l',l)
         #cv2.waitKey(100)
         #cv2.imshow('a',a)
         #cv2.waitKey(100)
         #cv2.imshow('b',b)
         #cv2.waitKey(100)
         meanl = numpy.mean(l.flatten())
         meana = numpy.mean(a.flatten())
         meanb = numpy.mean(b.flatten())
         #print('mean of l',meanl)
         #print('mean of a',meana)
         #print('mean of b',meanb)
         kurtl = scipy.stats.kurtosis(l.flatten())
         kurta = scipy.stats.kurtosis(a.flatten())
         kurtb = scipy.stats.kurtosis(b.flatten())
         #print('kurt of l',kurtl) 
         #print('kurt of a',kurta) 
         #print('kurt of b',kurtb)
         skewl = scipy.stats.skew(l.flatten())
         skewa = scipy.stats.skew(a.flatten())
         skewb = scipy.stats.skew(b.flatten())
         #print('skew of l',skewl) 
         #print('skew of a',skewa) 
         #print('skew of b',skewb)
         varl = numpy.var(l.flatten())
         vara = numpy.var(a.flatten())
         varb = numpy.var(b.flatten())
         #print('varience of l',varl)
         #print('varience of a',vara)
         #print('varience of b',varb) 
         F=[meanr,meang,meanb,varr,varg,varb,kurtr,kurtg,k,skewr,skewg,skewb,meanh,means,m,varh,v,varv,kurth,kurts,kurtv,skewh,skews,skewv,meanl,meana,meanb,varl,vara,
           varb,kurtl,kurta,kurtb,skewl,skewa,skewb] 
         gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
         #cv2.imshow('gray',gray)
         #cv2.waitKey(100)
         g = graycomatrix(gray,distances=[5],angles=[0],levels=256,symmetric=True,normed=True)
         #print ('value of g',g.shape)
         corelation = graycoprops(g,'correlation')[0,0]
         #print('corelation',corelation)
         energy = graycoprops(g,'energy')[0,0]
         #print('energy',energy)
         contrast = graycoprops(g,'contrast')[0,0]
         #print('contrast',contrast)
         homogenity = graycoprops(g,'homogeneity')[0,0]
         #print('homogeneity',homogenity)
         #standard = graycoprops(g,'standard deviation')[0,0]
         #print('standard',standard)
         #rms = graycoprops(g,'root mean square')[0,0]
         #print('rms',rms)
         #smoothness = graycoprops(g,'smoothness')[0,0]
         #print('smoothnes',smoothness)
         #entropy = graycoprops(g,'entropy')[0,0]
         #print('entropy',entropy)
         varience = numpy.var(g.flatten())
         #print('varience',varience)
         kurtosis = scipy.stats.kurtosis(g.flatten())
         #print('kurtosis',kurtosis)
         mean = numpy.mean(g)
         #print('mean',mean)
         skewnes = scipy.stats.skew(g.flatten())
         #print('skewness',skewnes)
         standard = numpy.std(g)
         #print('standard',standard)
         r = 0
         c = 0
         for j in g:
            #print(j[0][0][0])
            r = r+(j[0][0][0]*j[0][0][0])
            c = c+1
         rms = (r/c)
         #print('rms',rms)  
         entropy = skimage.filters.rank.entropy(gray,disk(5))
         entropy = numpy.sum(entropy)
         print('entropy',entropy) 
         idm = (numpy.sum(g))/(1+homogenity)
         #print('idm',idm)
         smoothnes = numpy.sum(g)/idm
         #print('smoothnes',smoothnes)
         f1 = [skewnes,mean,contrast,energy,homogenity,standard,rms,varience,smoothnes,kurtosis,corelation,entropy,idm]
         F.extend(f1)
         feature.append(F)
         #print(F,f1)
         fd,hog_image=hog(img1,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),visualize=True,multichannel=True) 
         #print(type(fd))
         fd = list(fd)
         #print(fd,len(fd),type(fd))
         f3.append(fd)
         
#pickle.dump(feature,open('feature1.pkl','wb'))
#pickle.dump(f3,open('f3.pkl','wb')) 
     
f3 = numpy.array(f3) [:,0:100]
#pca = PCA(n_components=100) 
#f3 = pca.fit_transform(f3)
#print(f3)
#print(f3.shape) 
feature = numpy.array(feature)
#print(feature.shape) 
feature = numpy.concatenate((feature,f3),axis=1) 
#print(feature.shape)
print(Counter(cd_list))
oversample=SMOTE()
feature,cd_list=oversample.fit_resample(feature,cd_list)
print(Counter(cd_list))
pickle.dump(feature,open('feature.pkl','wb'))
pickle.dump(cd_list,open('cd_list.pkl','wb'))
         
         
         
          
         
         
