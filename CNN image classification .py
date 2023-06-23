#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten


# In[2]:


X_train = np.loadtxt('C:/Users/DIPAYAN\OneDrive/Desktop/New folder (2)/Image Classification CNN Keras Dataset/input.csv', delimiter  = ',')
Y_train = np.loadtxt('C:/Users/DIPAYAN\OneDrive/Desktop/New folder (2)/Image Classification CNN Keras Dataset/labels.csv', delimiter = ',')
X_test = np.loadtxt('C:/Users/DIPAYAN\OneDrive/Desktop/New folder (2)/Image Classification CNN Keras Dataset/input_test.csv', delimiter = ',')
Y_test = np.loadtxt('C:/Users/DIPAYAN\OneDrive/Desktop/New folder (2)/Image Classification CNN Keras Dataset/labels_test.csv', delimiter =',')


# In[4]:


X_train = X_train.reshape(len(X_train), 100, 100, 3)
Y_train = Y_train.reshape(len(Y_train), 1)
 
X_test = X_test.reshape(len(X_test), 100, 100, 3)
Y_test = Y_test.reshape(len(Y_test), 1)
X_train = X_train/255.0
X_test = X_test/255.0


# In[5]:


print('shape of x_train: ',X_train.shape)
print("shape of y_train:",Y_train.shape)
print('shape of x_test: ',X_test.shape)
print('shape of y_test: ',Y_test.shape)


# In[6]:


idx = random.randint(0,len(X_train))
plt.imshow(X_train[idx, :])
plt.show()


# In[7]:


model=Sequential([
    Conv2D(32, (3,3), activation='relu',input_shape=(100, 100, 3)),
    MaxPooling2D((2,2)),
    
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])


# In[23]:


model=Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape = (100, 100, 3))),
    model.add(MaxPooling2D((2,2))),
    
    model.add(Conv2D(32, (3,3), activation='relu')),
    model.add(MaxPooling2D((2,2))),
    
    model.add(Flatten())
    model.add(Dense(64, activation='relu')),
    model.add(Dense(1, activation='sigmoid')),


# In[22]:



model.compile(loss= 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[10]:


model.fit(X_train, Y_train, epochs = 5, batch_size = 64)


# In[11]:


model.evaluate(X_test, Y_test)


# In[12]:


idx2 = random.randint(0, len(Y_test))
plt.imshow(X_test[idx2, :])
plt.show()

Y_pred = model.predict(X_test[idx2, :].reshape(1, 100, 100, 3))


# In[20]:


idx2 = random.randint(0, len(Y_test))
plt.imshow(X_test[idx2, :])
plt.show()
Y_pred = Y_pred > 0.5
if(Y_pred == 0):
    pred = 'dog'
else:
    pred ='cat'
    print("our model says it is a :", pred)


# In[ ]:





# In[ ]:




