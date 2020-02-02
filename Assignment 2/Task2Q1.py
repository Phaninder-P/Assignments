#!/usr/bin/env python
# coding: utf-8

# In[2]:


class Polygon:

    def __init__(self, n):
        self.number_of_sides = n

    def print_num_sides(self):
        print('There are ' + str(self.number_of_sides) + ' sides.')
        


# In[3]:


class Triangle(Polygon):

    def __init__(self, lengths_of_sides):
        Polygon.__init__(self, 3)
        self.lengths_of_sides = lengths_of_sides  # list of three numbers

    def get_area(self):
        '''return the area of the triangle using the semi-perimeter method'''
        a, b, c = self.lengths_of_sides

        # calculate the semi-perimeter
        s = (a + b + c) / 2
        return (s*(s-a)*(s-b)*(s-c)) ** 0.5


# In[4]:


tri = Triangle([3, 4, 5])
print(tri.get_area())


# In[5]:


tri.print_num_sides()


# In[ ]:




