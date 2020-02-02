#!/usr/bin/env python
# coding: utf-8

# In[1]:


def myreduce(func,list):
    result = list[0]
    for item in list[1:]:
        result = func(result,item)
        return result


# In[2]:


def sub(x,y): return x - y
print ("subtraction on list [2,4,6] using custom reduce function ="   + str(myreduce(sub, [2,4,6])) )


# In[ ]:




