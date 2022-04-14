import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report
import matplotlib.pyplot as plt
from collections import Counter
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
feature=pickle.load(open('feature.pkl','rb'))
cd_list=pickle.load(open('cd_list.pkl','rb'))
print(Counter(cd_list))
xtrain,xtext,ytrain,ytest=train_test_split(feature,cd_list,test_size=0.30,random_state=42)
svm_model=SVC()
svm_model.fit(xtrain,ytrain)
prediction=svm_model.predict(xtext)
print(ytest)
print(prediction)



count=0
for i in range(0,len(ytest)):
    if ytest[i]==prediction[i]:
        count=count+1
        

print(count)
accuracy=(count/len(ytest))*100
print('accuracy',accuracy)
cm=confusion_matrix(ytest,prediction)
print('confusion matrix',cm)
dis=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['Common Nevus','Atypical Nevus','Melanoma'])
dis.plot()
plt.title('SVM')
plt.show()
print(classification_report(ytest,prediction))
pickle.dump(svm_model,open('svm_model.pkl','wb'))


rf_model=RandomForestClassifier()
rf_model.fit(xtrain,ytrain)
prediction=rf_model.predict(xtext)
print(ytest)
print(prediction)
count=0
for i in range(0,len(ytest)):
    if ytest[i]==prediction[i]:
        count=count+1
        

print(count)
accuracy=(count/len(ytest))*100
print('accuracy',accuracy)
cm=confusion_matrix(ytest,prediction)
print('confusion matrix',cm)
dis=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['Common Nevus','Atypical Nevus','Melanoma'])
dis.plot()
plt.title('Random Forest')
plt.show()
print(classification_report(ytest,prediction))
pickle.dump(rf_model,open('rf_model.pkl','wb'))

xtrain=xtrain.reshape(xtrain.shape[0],xtrain.shape[1],1)
xtext=xtext.reshape(xtext.shape[0],xtext.shape[1],1)
ytrain=np_utils.to_categorical(ytrain)
ytest=np_utils.to_categorical(ytest)
model=Sequential()
model.add(Conv1D(filters=64,kernel_size=5,activation='relu',input_shape=(149,1)))
model.add(MaxPooling1D(pool_size=2,strides=1))
model.add(Conv1D(filters=128,kernel_size=3,activation='relu'))
model.add(MaxPooling1D(pool_size=2,strides=1))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(150,activation='relu'))
model.add(Dense(3,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
history=model.fit(xtrain,ytrain,validation_data=(xtext,ytest),epochs=100,verbose=2)
loss,acc=model.evaluate(xtext,ytest)
print(loss)
print(accuracy)
plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='test')
plt.title('training progress')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
plt.plot(history.history['accuracy'],label='train')
plt.plot(history.history['val_accuracy'],label='test')
plt.title('training progress')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
model_json = model.to_json()
with open('cnn_model.json','w') as json_file:
   json_file.write(model_json)
model.save_weights('cnn_model.h5')   
   
