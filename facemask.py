#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np
img=cv2.imread('aish.jpeg')




# In[3]:


img.shape


# In[4]:


img[0]


# In[5]:


import matplotlib.pyplot as plt


# In[6]:


plt.imshow(img)


# In[ ]:





# In[ ]:





# In[7]:


#algo used is voila-jones object detection algorithm
#it has four parts-
#Haar feature selection
#Creating an integral image
#Adaboost Training
#Cascading Classifiers

#Haar feature selection-common features found betweeen all human beings faces
#mid section of your nose is always lighter than the left an right section of nose
#value=sum(pixels in black area)-sum(pixels in white area)
#if value is close to 1 then there is possibility of haar feature
#if value is not close to 0 then there is no possibilty of haar feature


# In[8]:


haar_data=cv2.CascadeClassifier('data.xml')


# In[9]:


haar_data.detectMultiScale(img)


# In[10]:


#we use cv2.rectangle to draw a rectangle on face
#cv2.rectangle(img,(x,y),(w,h),(b,g,r),border_thickness)


# In[11]:


while True:
    faces=haar_data.detectMultiScale(img)
    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
    cv2.imshow('result',img)
    if cv2.waitKey(2)==27:
        break
cv2.destroyAllWindows()


# In[14]:


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
            print(len(data))
            if len(data)<400:
                data.append(face)  
        cv2.imshow('result',img)
        if cv2.waitKey(2)==27 or len(data) >=200:
            break
            
        
capture.release()

cv2.destroyAllWindows()


# In[13]:


import numpy as np
np.save('without_mask.npy',data)


# In[15]:


np.save('with_mask.npy',data)


# In[16]:


plt.imshow(data[0])


# In[ ]:




