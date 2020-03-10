#!/usr/bin/env python
# coding: utf-8

# In[264]:


#question 1
def change_char(str1):
    char = str1[0]
    str1 = str1.replace(char, '$')
    str1 = char + str1[1:]
    return str1
print(change_char('prospect'))


# In[34]:


#question 2
def chars_mix_up(a, b):
    new_a = b[:2] + a[2:]
    new_b = a[:2] + b[2:]
    return new_a + ' ' + new_b

print(chars_mix_up('abc', 'xyz'))


# In[41]:


#question 3
def sample(str1):
    k=str1
    a=len(str1)
    if a>=3:
        if k[3:]=='ing':
            return k + 'ly'
        else:
            return k +'ing'
    else:
        return 'Length is less than 3'

print(sample('a'))


# In[50]:


#question 4
def not_poor(str1):
    snot = str1.find('not')
    spoor = str1.find('poor')
    print(snot,spoor)
    if spoor > snot and snot>0 and spoor>0:
        str1 = str1.replace(str1[snot:(spoor+4)], 'good')
        return str1
    else:
        return str1
    
print(not_poor('The lyrics is not that poor!'))
#print(not_poor('The lyrics is poor!'))


# In[67]:


def oddindex(str1):
    str1=str1[::-1]
    str1=str1[::2]
    str1=str1[::-1]
    return str1

print(oddindex('string'))


# In[69]:


#question 5
def odd_values_string(str):
    result = ""
    for i in range(len(str)):
        if i % 2 == 0:
            result = result + str[i]
            return result
        
print(odd_values_string('abcdef'))
print(odd_values_string('python'))


# In[71]:


#question 6
def insert_end(str):
    sub_str = str[-2:]
    return sub_str*4
print(insert_end('Python'))
print(insert_end('Exercises'))


# In[74]:


#question 8
def start(str):
    k=len(str)
    if k>=3:
        sub_str = str[0:3]
        return sub_str 
    else:
        return 'Inavlid'
print(start('Py'))


# In[77]:


#question 9
x = 3.1415926
y = 12.9999
print("\nOriginal Number: ", x)
print("Formatted Number: "+"{:.2f}".format(x));
print("Original Number: ", y)
print("Formatted Number: "+"{:.2f}".format(y));
print()


# In[123]:



#question 10
def substring(k):
    str="Machine Learning is undeniably one of the most influential and powerful technologies in today’s world. More importantly, we are far from seeing its full potential. There’s no doubt, it will continue to be making headlines for the foreseeable future. This article is designed as an introduction to the Machine Learning concepts, covering all the fundamental ideas without being too high level."
    l=str.count(k)
    return l
    
print(substring("the"))


# In[268]:


#question 11
import collections
str1 = 'thequickbrownfoxjumpsoverthelazydog'
d = collections.defaultdict(int)
for c in str1: 
    d[c] += 1
for c in sorted(d, key=d.get, reverse=True):
    if d[c] > 1:
        print('%s %d' % (c, d[c]))


# In[107]:


#question 12
area = 1256.66
volume = 1254.725
decimals = 2
print("The area of the rectangle is {0:.{1}f}cm\u00b2".format(area, decimals))
decimals = 3
print("The volume of the cylinder is {0:.{1}f}cm\u00b3".format(volume,
decimals))


# In[137]:


#question 13
import string
alphabet = set(string.ascii_lowercase)
print(alphabet)

input_string = 'The quick brown fox jumps over the lazy dog'
l=set(input_string.lower())
print(l)
print(set(input_string.lower()) >= alphabet)

input_string = 'The quick brown fox jumps over the lazy cat'
print(set(input_string.lower()) >= alphabet)
print(len(input_string)-8)


# In[5]:


#question 15
def word_count(str):
    counts = dict()
    words = str.split()
    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
            counts_x = sorted(counts.items(), key=lambda kv: kv[1])
            #print(counts_x)
    return counts_x[-1]

print(word_count("Both of these issues are fixed by postponing the evaluation of annotations. Instead of compiling code which executes expressions in annotations at their definition time, the compiler stores the annotation in a string form equivalent to the AST of the expression in question. If needed, annotations can be resolved at runtime using typing.get_type_hints(). In the common case where this is not required, the annotations are cheaper to store (since short strings are interned by the interpreter) and make startup time faster."))


# In[33]:


#question 17
from itertools import combinations
def lowerstr(str,l):
    k=set(str)
    m=list(combinations(k,l))
    return len(m)
    

print(lowerstr("wolf",3))


# In[ ]:


#question 18
def number_of_substrings(str):
    str_len = len(str);
    return int(str_len * (str_len + 1) / 2);

str1 = input("Input a string: ")
print("Number of substrings:")
print(number_of_substrings(str1))


# In[6]:


#question 20
def match_words(words):
    ctr = 0
    for word in words:
        if len(word) > 1 and word[0] == word[-1]:
            ctr += 1
    return ctr
print(match_words(['abc', 'xyz', 'aba', '1221']))


# In[27]:


#question 21
def last(n): 
    return n[-1]


def incrlast(list1):
    return sorted(list1,key=last)

    
    
    
print(incrlast([(3,5),(6,1),(2,2),(3,3)]))


# In[31]:


#question 22
def remove_duplicate(list1):
    list2=set(list1)
    l=list(list2)
    return l

print(remove_duplicate([2,2,3,4,5,6,5,3,5,7]))


# In[37]:


#question 23
def long(n,list1):
    word=[]
    l=list1.split('')
    for i in range(len(l)):
        if(l[i]>n):
            word.append(l[i])
    return word

print(long_words(4, "The quick brown fox jumps over the lazy dog"))


# In[53]:


#question 24

def removelist(list1):
    k=[x for (i,x) in enumerate(list1) if i not in (0,4,5)]
    return k

print(removelist(['Red', 'Green', 'White', 'Black', 'Pink', 'Yellow']))


# In[55]:


#question 25
import itertools
print(list(itertools.permutations([5,3,8])))


# In[101]:


#question 26
def sorteduniq(tuple1):
    m=[]
    k=tuple1[0]
    l=list(k)
    return l
    
    

print(sorteduniq( [ (1, 2), (3, 4), (1, 2), (5, 6), (7, 8), (1, 2), (3, 4), (3, 4), (7, 8), (9, 10) ]))


# In[ ]:


#question 27
class py_solution:
    def int_to_Roman(self, num):


        val = [
1000, 900, 500, 400,
100, 90, 50, 40,
10, 9, 5, 4,
1]
        syb = [
"M", "CM", "D", "CD",
"C", "XC", "L", "XL",
"X", "IX", "V", "IV",
"I"
]
        roman_num = ''
        i = 0
        while num > 0:
            for _ in range(num // val[i]):
                roman_num += syb[i]
                num -= val[i]
                i += 1
        return roman_num

print(py_solution().int_to_Roman(1))
print(py_solution().int_to_Roman(4000))


# In[1]:


#problem 29

class py_solution:
    def is_valid_parenthese(self, str1):
        stack, pchar = [], {"(": ")", "{": "}", "[": "]"}
        for parenthese in str1:
            if parenthese in pchar:
                stack.append(parenthese)
            elif len(stack) == 0 or pchar[stack.pop()] != parenthese:
                return False
        return len(stack) == 0

print(py_solution().is_valid_parenthese("(){}[]"))
print(py_solution().is_valid_parenthese("()[{)}"))
print(py_solution().is_valid_parenthese("()"))


# In[13]:


#problem 31
def fucn(list1,n):
    l=[]
    l=list1
    for i in range(len(l)):
        for j in range(len(l)):
            if(l[i]+l[j]==n):
                return i,j
                
print(fucn([10,20,10,40,50,60,70],70))
    


# In[24]:


#problem 32
def func(list1):
    l=[]
    l=list1
    k=0
    for i in range(len(list1)):
        for j in range(len(list1)):
            for k in range(len(list1)):
                if(l[i]+l[j]+l[k]==0):
                    print((l[i],l[j],l[k]))

print(func( [-25, -10, -7, -3, 2, 4, 8, 10]))


# In[28]:


#problem 33
class py_solution:
    def pow(self, x, n):
        if x==0 or x==1 or n==1:
            return x 
        if x==-1:
            if n%2 ==0:
                return 1
            else:
                return -1
            if n==0:
                return 1
            if n<0:
                return 1/self.pow(x,-n)
            val = self.pow(x,n//2)
            if n%2 ==0:
                return val*val
            return val*val*x
        
print(py_solution().pow(2, -3));
print(py_solution().pow(3, 5));
print(py_solution().pow(100, 0));


# In[30]:


class solution:
    def __init__(self):
        self.str1 = ""
    def get_String(self):
        self.str1 = input()
    def print_String(self):
        print(self.str1.upper())
            
str1 = solution()
str1.get_String()
str1.print_String()      


# In[6]:


#problem 35
class rectangle():
    
    def __init__(self,l,b):
        self.length=l
        self.breadth=b
        
    def area(self):
        return self.breadth*self.length
    
print(rectangle(20,30).area())


# In[14]:


#problem 36
class circle():
    def __init__(self, r):
        self.radius = r
        
    def area(self):
        print("Area is:")
        return 3.14*self.radius*self.radius
    
    def perimeter(self):
        print("Perimeter is:")
        return 2*3.14*self.radius
    
    
print(circle(8).area())
print(circle(8).perimeter())


# In[15]:


#question 37
import itertools
x = itertools.cycle('ABCD')
print(type(x).__name__)


# In[19]:


from collections import Counter
classes = (
('V', 1),
('VI', 1),
('V', 2),
('VI', 2),
('VI', 3),
('VII', 1),
)
students = Counter(class_name for class_name, no_students in classes)
print(students)


# In[26]:


#question 39
from collections import OrderedDict
dict = {'Afghanistan': 93, 'Albania': 355, 'Algeria': 213, 'Andorra':
376, 'Angola': 244}
new_dict = OrderedDict(dict.items())
for key in reversed(new_dict):
    print (key, new_dict[key])


# In[27]:


#question 40
from collections import Counter
def compare_lists(x, y):
    return Counter(x) == Counter(y)
n1 = [20, 10, 30, 10, 20, 30]
n2 = [30, 20, 10, 30, 20, 50]
print(compare_lists(n1, n2))


# In[28]:


#question 41
from array import array
a = array("I", (12,25))
print("Array buffer start address in memory and number of elements.")
print(a.buffer_info())


# In[29]:


#problem 42
import array
import binascii
a = array.array('i', [1,2,3,4,5,6])
print("Original array:")
print('A1:', a)
bytes_array = a.tobytes()
print('Array of bytes:', binascii.hexlify(bytes_array))


# In[30]:


#problem 43
import array
import binascii
a = array.array('i', [1,2,3,4,5,6])
print("Original array:")
print('A1:', a)
bytes_array = a.tobytes()
print('Array of bytes:', binascii.hexlify(bytes_array))


# In[31]:


#problem 44
from array import array
import binascii
array1 = array('i', [7, 8, 9, 10])
print('array1:', array1)
as_bytes = array1.tobytes()
print('Bytes:', binascii.hexlify(as_bytes))
array2 = array('i')
array2.frombytes(as_bytes)
print('array2:', array2)


# In[33]:


#question 45
import heapq
heap = []
heapq.heappush(heap, ('V', 3))
heapq.heappush(heap, ('V', 2))
heapq.heappush(heap, ('V', 1))
print("Items in the heap:")
for a in heap:
    print(a)
    print("The smallest item in the heap:")
    print(heap[0])
    print("Pop the smallest item in the heap:")
    heapq.heappop(heap)
for a in heap:
    print(a)


# In[269]:


#question 47
def harmonic_sum(n):
    if n < 2:
        return 1
    else:
        return 1 / n + (harmonic_sum(n - 1))
print(harmonic_sum(7))
print(harmonic_sum(4))


# In[270]:


#question 48
import numpy as np
x = np.ones((5,5))
print("Original array:")
print(x)
print("1 on the border and 0 inside in the array")
x[1:-1,1:-1] = 0
print(x)


# In[271]:


#question 49
import numpy as np
x = np.ones((3,3))
print("Checkerboard pattern:")
x = np.zeros((8,8),dtype=int)
x[1::2,::2] = 1
x[::2,1::2] = 1
print(x)


# In[272]:


#question 50
import numpy as np
# Create an empty array
x = np.empty((3,4))
print(x)
# Create a full array
y = np.full((3,3),6)
print(y)
51. 
import numpy as np
fvalues = [0, 12, 45.21, 34, 99.91]
F = np.array(fvalues)
print("Values in Fahrenheit degrees:")
print(F)
print("Values in Centigrade degrees:")
print(5*F/9 - 5*32/9)


# In[273]:


#question 52
import numpy as np
x = np.sqrt([1+0j])
y = np.sqrt([0+1j])
print("Original array:x ",x)
print("Original array:y ",y)
print("Real part of the array:")
print(x.real)
print(y.real)
print("Imaginary part of the array:")
print(x.imag)
print(y.imag)


# In[274]:


#question 53
import numpy as np
array1 = np.array([0, 10, 20, 40, 60])
print("Array1: ",array1)
array2 = [0, 40]
print("Array2: ",array2)
print("Compare each element of array1 and array2")
print(np.in1d(array1, array2))


# In[275]:


#question 54
import numpy as np
array1 = np.array([0, 10, 20, 40, 60])
print("Array1: ",array1)
array2 = [10, 30, 40]
print("Array2: ",array2)
print("Common values between two arrays:")
print(np.intersect1d(array1, array2))


# In[276]:


#question 55
import numpy as np
x = np.array([10, 10, 20, 20, 30, 30])
print("Original array:")
print(x)
print("Unique elements of the above array:")
print(np.unique(x))
x = np.array([[1, 1], [2, 3]])
print("Original array:")
print(x)
print("Unique elements of the above array:")
print(np.unique(x))


# In[277]:


#question 56
import numpy as np
array1 = np.array([0, 10, 20, 40, 60, 80])
print("Array1: ",array1)
array2 = [10, 30, 40, 50, 70]
print("Array2: ",array2)
print("Unique values that are in only one (not both) of the input arrays:")
print(np.setxor1d(array1, array2))


# In[278]:


#question 57
import numpy as np
print(np.all([[True,False],[True,True]]))
print(np.all([[True,True],[True,True]]))
print(np.all([10, 20, 0, -50]))
print(np.all([10, 20, -50]))


# In[282]:


#question 58
import numpy as np
print(np.any([[False,False],[False,False]]))
print(np.any([[True,True],[True,True]]))
print(np.any([10, 20, 0, -50]))
print(np.any([10, 20, -50]))


# In[283]:


#question 59
import numpy as np
a = [1, 2, 3, 4]
print("Original array")
print(a)
print("Repeating 2 times")
x = np.tile(a, 2)
print(x)
print("Repeating 3 times")
x = np.tile(a, 3)
print(x)


# In[284]:


#question 60
import numpy as np
x = np.array([1, 2, 3, 4, 5, 6])
print("Original array: ",x)
print("Maximum Values: ",np.argmax(x))
print("Minimum Values: ",np.argmin(x))


# In[285]:


#question 61
import numpy as np
a = np.array([1, 2])
b = np.array([4, 5])
print("Array a: ",a)
print("Array b: ",b)
print("a > b")
print(np.greater(a, b))
print("a >= b")
print(np.greater_equal(a, b))
print("a < b")
print(np.less(a, b))
print("a <= b")
print(np.less_equal(a, b))


# In[286]:


#question 62
import numpy as np
a = np.array([[4, 6],[2, 1]])
print("Original array: ")
print(a)
print("Sort along the first axis: ")
x = np.sort(a, axis=0)
print(x)
print("Sort along the last axis: ")
y = np.sort(x, axis=1)
print(y)


# In[287]:


#question 63
import numpy as np
first_names = ('Margery', 'Betsey', 'Shelley', 'Lanell', 'Genesis')
last_names = ('Woolum', 'Battle', 'Plotner', 'Brien', 'Stahl')
x = np.lexsort((first_names, last_names))
print(x)


# In[288]:


#question 64
import numpy as np
x = np.array([[0, 10, 20], [20, 30, 40]])
print("Original array: ")
print(x)
print("Values bigger than 10 =", x[x>10])
print("Their indices are ", np.nonzero(x > 10))


# In[289]:


#question 65
import numpy as np
n = np.zeros((4,4))
print("%d bytes" % (n.size * n.itemsize))


# In[290]:


#question 66
import numpy as np
print("Create an array of zeros")
x = np.zeros((1,2))
print("Default type is float")
print(x)
print("Type changes to int")
x = np.zeros((1,2), dtype = np.int)
print(x)
print("Create an array of ones")
y= np.ones((1,2))
print("Default type is float")
print(y)
print("Type changes to int")
y = np.ones((1,2), dtype = np.int)
print(y)


# In[291]:


#question 67
import numpy as np
x = np.array([1, 2, 3, 4, 5, 6])
print("6 rows and 0 columns")
print(x.shape)
y = np.array([[1, 2, 3],[4, 5, 6],[7,8,9]])
print("(3, 3) -> 3 rows and 3 columns ")
print(y)
x = np.array([1,2,3,4,5,6,7,8,9])
print("Change array shape to (3, 3) -> 3 rows and 3 columns ")
x.shape = (3, 3)
print(x)


# In[292]:


#question 68
import numpy as np
x = np.array([1, 2, 3, 4, 5, 6])
y = np.reshape(x,(3,2))
print("Reshape 3x2:")
print(y)
z = np.reshape(x,(2,3))
print("Reshape 2x3:")
print(z)


# In[293]:


#problem 101
import pandas as pd
d1=pd.Series([21,22,23,56,89])
d2=pd.Series([40,32,56,78,90])
print("Add two series")
k=d1+d2
print(k)
print("Subtact two series")
k=d1-d2
print(k)
print("Multipy two series")
k=d1*d2
print(k)
print("Divide two series")
k=d1/d2
print(k)


# In[11]:


#question 102
import pandas as pd
k={'a': 100, 'b': 200, 'c': 300, 'd': 400, 'e': 800}
l=pd.Series(k)
print(l)


# In[23]:


#question 103
import pandas as pd
s1 = pd.Series(['100', '200', 'python', '300.12', '400'])
print(s1)
print("Change the said data type to numeric:")
s2 = pd.to_numeric(s1,errors='coerce',downcast='float')
print(s2)


# In[36]:


#question 104
import pandas as pd
d= {'col1': [1, 2, 3, 4, 7, 11], 'col2': [4, 5, 6, 9, 5, 0], 'col3': [7, 5, 8, 12, 1,11]}
d1=pd.DataFrame(d)
print(d1)
print('First column:')
s1=d1.ix[:,0]
print(s1)


# In[42]:


#question 105
import pandas as pd
d=[1,2,3,4,5,6,7,8,9,5,3]
d1=pd.DataFrame(d)
print(d1)
mean=d1.mean()
print("Mean is")
print(mean)
Std=d1.std()
print("Standard deviation is")
print(Std)


# In[43]:


#question 106
import pandas as pd
d={'X ':[78,85,96,80,86], ' Y ':[84,94,89,83,86],'Z':[86,97,96,72,83]}
d1=pd.DataFrame(d)
d1


# In[54]:


#question 107
import numpy as np
import pandas as pd
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
             'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
             'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
             'qualify': [ 'yes', 'no', 'yes' , 'no', ' no ', ' yes ', 'yes', 'no', 'no', 'yes' ] }
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j' ]
d1=pd.DataFrame(exam_data,index=labels)
print(d1)
print("The first 3 rows are:")
k=d1.iloc[:3]
print(k)


# In[63]:


#question 108
import pandas as pd
import numpy as np

exam_data = {'name ': [ 'Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael','Matthew', 'Laura', 'Kevin', 'Jonas'],
             'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
             'attempts' : [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
             'qualify' : ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes'] }
labels = ['a ', ' b ', ' c ', ' d ', ' e ', ' f ', ' g ', 'h', 'i', 'j']
d1=pd.DataFrame(exam_data,index=labels)
k=d1.iloc[[1, 3, 5, 6],[1,3]]
print(k)


# In[76]:


#question 109
import pandas as pd
import numpy as np
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 
                      'Matthew', 'Laura', 'Kevin', 'Jonas'],
             'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
             'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
             'qualify': ['yes', 'no', ' yes ', ' no ', ' no ', ' yes ', ' yes ', ' no ', ' no ', ' yes '] }
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
d1=pd.DataFrame(exam_data,index=labels)
d=d1.ix[:,1]
print(d)
k=d.mean()
print("The mean is:")
print(k)


# In[85]:


#question 110
import pandas as pd
d = {'col1': [1, 2, 3], 'col2': [4, 5, 6], 'col3': [7, 8, 9]}
d1=pd.DataFrame(d)
print("original Data")
print(d1)
print("Renamed Data")
k=d1.rename(columns={'col1':'Column 1','col2':'Column 2','col3':'Column 3'})
print(k)


# In[94]:


#question 111
import pandas as pd
d1 = pd.DataFrame({'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily','Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
                   'city': ['California', 'Los Angeles', 'California', 'California', 'California', 'Los Angeles', 'Los Angeles', 'Georgia', 'Georgia', 'Los Angeles']})
k= d1.groupby(["city"]).size().reset_index(name='Number of people')
print(k)


# In[104]:


#question 112
import pandas as pd
import numpy as np
d = {'col1': [1, 4, 3, 4, 5], 'col2': [4, 5, 6, 7, 8], 'col3': [7, 8, 9, 0, 1]}
df = pd.DataFrame(data=d)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
print("Original DataFrame")
print(df)


# In[110]:


#question 113
import pandas as pd
d1=pd.Series(['3/11/2000', '3/12/2000', '3/13/2000'])
print("Original series")
print(d1)
d = pd.to_datetime(pd.Series(['3/11/2000', '3/12/2000', '3/13/2000']))
print(d)


# In[112]:


#question 114
import pandas as pd
d = pd.DataFrame()
d1 = pd.DataFrame({"col1": range(3),"col2": range(3)})
d = d.append(d1)
print(d)


# In[117]:


#question 115
import pandas as pd
d = {'col1': [1, 2, 3, 4, 7], 'col2': [4, 5, 6, 9, 5], 'col3': [7, 8, 12, 1, 11]}
d1=pd.DataFrame(d)
print(d1)
k=len(d1.columns)
print("The number of columns are:")
print(k)


# In[121]:


#question 116
import pandas as pd
d = {'col1': [1, 2, 3, 4, 7, 11], 'col2': [4, 5, 6, 9, 5, 0], 'col3': [7, 5, 8, 12, 1,11]}
df = pd.DataFrame(data=d)
k=df.drop([3,4,5])
print(k)
l=df.iloc[:3]
print(l)


# In[136]:


#question 117
import pandas as pd
df=pd.read_excel('C:\\Users\\omkar\\Downloads\\coalpublic2013.xlsx')
print(df)


# In[161]:


#question 118
import pandas as pd
df=pd.read_excel('C:\\Users\\omkar\\Downloads\\coalpublic2013.xlsx')
print(k)
df[df['Mine_Name'].map(lambda x:x.startswith("P"))].head()


# In[170]:


#question 119
import pandas as pd
d=pd.read_excel('C:\\Users\\omkar\\Downloads\\py sub and obj\\py sub and obj\\data for subjective questions\\employee.xlsx')
print(d)

d[d['hire_date'] >='20070101']


# In[174]:


#question 120
import pandas as pd
df=pd.read_excel('C:\\Users\\omkar\\Downloads\\py sub and obj\\py sub and obj\\data for subjective questions\\employee.xlsx')
print(d)
df2 = df.set_index(['hire_date'])
result = df2["2005"]
result


# In[186]:


#question 121
import pandas as pd
df0=pd.read_excel("C:\\Users\\omkar\\Downloads\\py sub and obj\\py sub and obj\\data for subjective questions\\employee.xlsx",sheet_name=0)
df1=pd.read_excel("C:\\Users\\omkar\\Downloads\\py sub and obj\\py sub and obj\\data for subjective questions\\employee.xlsx",sheet_name=1)
df2=pd.read_excel("C:\\Users\\omkar\\Downloads\\py sub and obj\\py sub and obj\\data for subjective questions\\employee.xlsx",sheet_name=2)
df=pd.concat([df0,df1,df2])
df


# In[187]:


#question 122
import pandas as pd
df0=pd.read_excel("C:\\Users\\omkar\\Downloads\\py sub and obj\\py sub and obj\\data for subjective questions\\employee.xlsx",sheet_name=0)
df1=pd.read_excel("C:\\Users\\omkar\\Downloads\\py sub and obj\\py sub and obj\\data for subjective questions\\employee.xlsx",sheet_name=1)
df2=pd.read_excel("C:\\Users\\omkar\\Downloads\\py sub and obj\\py sub and obj\\data for subjective questions\\employee.xlsx",sheet_name=2)
df=pd.concat([df0,df1,df2])
df.to_excel("New excel sheet.xlsx")


# In[199]:


#question 123
import pandas as pd
import numpy as np
df=pd.read_csv("C:\\Users\\omkar\\Downloads\\py sub and obj\\py sub and obj\\data for subjective questions\\titanic.csv")
k=pd.pivot_table(df,index=['sex','age'])
l=pd.pivot_table(df,index=['sex','age'],aggfunc=np.sum)
print(l)


# In[204]:


#question 124
import numpy as np
df=pd.read_csv("C:\\Users\\omkar\\Downloads\\py sub and obj\\py sub and obj\\data for subjective questions\\titanic.csv")
l=pd.cut
print(l)


# In[210]:


#question 125
import numpy as np
import panadas as
df=pd.read_csv("C:\\Users\\omkar\\Downloads\\py sub and obj\\py sub and obj\\data for subjective questions\\titanic.csv")
l = pd.cut(df['age'],[0, 10, 30, 60, 80])
print(l)


# In[211]:


#question 126
import pandas as pd
import numpy as np
df = pd.read_csv('C:\\Users\\omkar\\Downloads\\py sub and obj\\py sub and obj\\data for subjective questions\\titanic.csv')
age = pd.cut(df['age'], [0, 20, 55])
result = df.pivot_table('survived', index=['sex', age], columns='class')
print(result)


# In[217]:


#question 127
import pandas as pd
import numpy as np
df = pd.read_csv('C:\\Users\\omkar\\Downloads\\py sub and obj\\py sub and obj\\data for subjective questions\\titanic.csv')
result = df.pivot_table(index=['sex'], columns=['pclass'], aggfunc='count')
print(result)


# In[221]:


#question 128
import pandas as pd
import numpy as np
df = pd.read_csv('C:\\Users\\omkar\\Downloads\\py sub and obj\\py sub and obj\\data for subjective questions\\titanic.csv')
df.pivot_table( 'survived' , [ 'sex' , 'alone' ] , 'class')


# In[225]:


#question 129
import pandas as pd
import numpy as np
df = pd.read_csv('C:\\Users\\omkar\\Downloads\\py sub and obj\\py sub and obj\\data for subjective questions\\titanic.csv')
df
df.pivot_table( 'survived' , ['class', 'sex', 'alone','embark_town'])


# In[246]:


#question 130
import pandas as pd
df=pd.read_csv("C:\\Users\\omkar\\Downloads\\py sub and obj\\py sub and obj\\data for subjective questions\\ufo_sighting_data.csv")
df


# In[260]:


#question 131
import pandas as pd
df=pd.read_csv("C:\\Users\\omkar\\Downloads\\py sub and obj\\py sub and obj\\data for subjective questions\\ufo_sighting_data.csv")

selected_period = df[(df['Date_time'] >= '1950-01-01 00:00:00') & (df['Date_time']<= '1960-12-31 23:59:59')]
print(selected_period)


# In[ ]:




