#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2


# In[2]:


with_mask=np.load('with_mask.npy')
without_mask=np.load('without_mask.npy')


# In[3]:


with_mask.shape


# In[4]:


without_mask.shape


# In[5]:


with_mask=with_mask.reshape(200,50*50*3)
without_mask=without_mask.reshape(200,50*50*3)
with_mask.shape


# In[6]:


without_mask.shape


# In[7]:


X=np.r_[with_mask,without_mask]
X.shape


# In[8]:


labels=np.zeros(X.shape[0])
labels[200:]=1.0
names={0:'Mask',1:'No Mask'}


# In[9]:


#svm=support vector machine
#svc-support vector classifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


x_train,x_test,y_train,y_test=train_test_split(X,labels, test_size=0.20)


# In[12]:


x_train.shape


# In[13]:


#PCA-principl component Analysis
from sklearn.decomposition import PCA


# In[14]:


pca= PCA(n_components=3)
x_train= pca.fit_transform(x_train)


# In[15]:


x_train[0]


# In[16]:


x_train.shape


# In[17]:


svm = SVC()
svm.fit(x_train, y_train)


# In[18]:


x_test = pca.transform(x_test)
y_pred=svm.predict(x_test)


# In[19]:


accuracy_score(y_test,y_pred)


# In[20]:


haar_data=cv2.CascadeClassifier('data.xml')
capture=cv2.VideoCapture(0)
data_withoutmask=[]
data=[]
while True:
    flag,img=capture.read()
    if flag:
        faces=haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face,(50,50))
            face = face.reshape(1,-1)
            face = pca.transform(face)
            pred=svm.predict(face)
            n = names[int(pred)]
            print(n)
        cv2.imshow('result',img)
        if cv2.waitKey(2)==27:
            break
            
        
capture.release()
cv2.destroyAllWindows()


# In[ ]:




