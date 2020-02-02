#!/usr/bin/env python
# coding: utf-8

# In[1]:


def find_longest_word(words_list):
    word_len = []
    for n in words_list:
        word_len.append((len(n), n))
    word_len.sort()
    return word_len[-1][1]
print(find_longest_word(["image", "processing", "detection"]))


# In[ ]:




