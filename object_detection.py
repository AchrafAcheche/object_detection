#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


# In[13]:


(X_train, y_train), (X_test,y_test) = datasets.cifar10.load_data()
X_train.shape


# In[3]:


X_test.shape


# In[5]:


y_train.shape


# In[7]:


y_test.shape


# In[14]:


y_train[:5]


# In[15]:


y_train = y_train.reshape(-1,)
y_train[:5]


# In[16]:


y_test[:5]


# In[17]:


y_test = y_test.reshape(-1,)
y_test[:5]


# In[18]:


classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


# In[24]:


def plot_sample(X, y, index):
    plt.figure(figsize = (15,2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])


# plot_sample(X_train, y_train, 1000)

# RGB valeur entre 0 --> 255 on divise par 255 on obtient un numéro entre 0 et 1

# In[26]:


X_train = X_train / 255.0
X_test = X_test / 255.0


# simple artificial neural network

# In[27]:


ann = models.Sequential([
        layers.Flatten(input_shape=(32,32,3)),
        layers.Dense(3000, activation='relu'),
        layers.Dense(1000, activation='relu'),
        layers.Dense(10, activation='softmax')    
    ])

ann.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

ann.fit(X_train, y_train, epochs=5)


# In[28]:


from sklearn.metrics import confusion_matrix , classification_report
import numpy as np
y_pred = ann.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

print("Classification Report: \n", classification_report(y_test, y_pred_classes))


# convolutional neural network

# In[29]:


cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# In[30]:


from sklearn.metrics import confusion_matrix , classification_report
import numpy as np
y_pred = ann.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

print("Classification Report: \n", classification_report(y_test, y_pred_classes))


# In[31]:


cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[32]:


cnn.fit(X_train, y_train, epochs=10)


# In[33]:


cnn.evaluate(X_test,y_test)


# In[34]:


y_pred = cnn.predict(X_test)
y_pred[:5]


# In[35]:


y_classes = [np.argmax(element) for element in y_pred]
y_classes[:5]


# In[36]:


y_test[:5]


# In[37]:


plot_sample(X_test, y_test,3)


# In[38]:


classes[y_classes[3]]


# In[ ]:





# In[ ]:




