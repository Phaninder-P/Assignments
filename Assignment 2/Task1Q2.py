#!/usr/bin/env python
# coding: utf-8

# In[8]:


word = "ACADGILD"
list = [ i for i in word ]
print (str(list))


# In[9]:


input_list = ['x','y','z']
result = [ i*num for i in input_list for num in range(1,5)  ]
print(str(result))


# In[10]:


input_list = ['x','y','z']
result = [ i*num for num in range(1,5) for i in input_list  ]
print(str(result))


# In[11]:


input_list = [2,3,4]
result = [ [i+num] for i in input_list for num in range(0,3)]
print(str(result))


# In[12]:


input_list = [2,3,4,5]
result = [ [i+num for i in input_list] for num in range(0,4)  ]
print(str(result))


# In[13]:


input_list=[1,2,3]
result = [ (b,a) for a in input_list for b in input_list]
print(str(result))


# In[ ]:




