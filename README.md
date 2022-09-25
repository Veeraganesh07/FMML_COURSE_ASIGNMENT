# FMML_COURSE_ASIGNMENT

Getting Started
FMML Module 1, Lab 1
Module Coordinator: Amit Pandey ( amit.pandey@research.iiit.ac.in )
Release date: Aug 2022

In this notebook we will be covering the very basics of Python and some basic libraries such as Numpy, Matplotlib and Nltk.
It is suggested that you go through each line and try some examples.
Section 1 - Python : Basic data types and indexing.
## Strings
'''
A string is a collection of one or more characters put in a single quote,
 double-quote or triple quote. In python there is no character data type,
 a character is a string of length one. It is represented by str class.

String can have special characters. String can be indexed

'''


name = 'First Lab'
name_extended = name + 'Module 1'
last_element_string = name[-1] # -1 in python is index of the last element. 
## indexing is important for preprocessing of the raw data.
print(name ,"\n", name_extended, "\n", last_element_string)
First Lab 
 First LabModule 1 
 b
## List

'''
Lists are ordered collection of data, and are very similar to arrays, 
It is very flexible as the items in a list do not need to be of the same type.
'''

name_list = ['First Lab', 3 , '1.1' , 'Lab 1'] ## notice elements are of different data type.
name_list.extend(['Module 1']) ## adding elements to list (Read about append method as well).
element_2 = name_list[1] ## Just like other languages, the index starts from 0.
two_dimesional_list = [[1,2],[3,4]] ## practice with multi-dimensional lists and arrays
## you would soon be required to handle 4 dimensional data :p :)
name_list[2] = '1.111' ##list elements can be changed
print(name_list)
print(element_2)
print(two_dimesional_list)
## list can have list, dictionary, string etc.
['First Lab', 3, '1.111', 'Lab 1', 'Module 1']
3
[[1, 2], [3, 4]]
## Tuples

name_tuple = ('First Lab', 1, (2,3),[1,1,'list having string']) ## A tuple can have a tuple.
 
print(name_tuple[2])
print("first indexing the last element of the tuple, which is a list and \n then last element of the list (a string) and then second last element of the string:")
print(name_tuple[-1][-1][-2])
(2, 3)
first indexing the last element of the tuple, which is a list and 
 then last element of the list (a string) and then second last element of the string:
n
## tuples are immutable, read the error !
 #usued when passing parameters etc. and dont want them to be changed
name_tuple = list(name_tuple)
name_tuple[1] = 2
name_tuple
['First Lab', 2, (2, 3), [1, 1, 'list having string']]
## Sets
'''a Set is an unordered collection of data types that is iterable, mutable and has no duplicate elements. 
The order of elements in a set is undefined though it may consist of various elements.
The major advantage of using a set, as opposed to a list,
 is that it has a highly optimized method for checking whether a specific element is contained in the set.
'''
set_unique = set([1,1,2,3,5,6,'Lab1'])
print(set_unique) ##notice it is unordered
last_el = set_unique.pop()
set_unique.add((1,2))


print(last_el)
print(set_unique)
{1, 2, 3, 'Lab1', 5, 6}
1
{(1, 2), 2, 3, 'Lab1', 5, 6}
set_unique = list(set_unique)
set_unique[1] ##it is not indexable
2
## Dictionary
'''
Dictionary in Python is an unordered collection of data values, used to store data values like a map,
 which, unlike other data types which hold only a single value as an element.
'''

dic = {'1': 'A','2':'B', 'C':3 } ##Observe how key and values can be anything
dic['4'] ='New'
print(dic)
{'1': 'A', '2': 'B', 'C': 3, '4': 'New'}
Question 0:
write down 3-5 methods applicable to each data type. (Hint: extend, reverse, etc.
#string method
string= "chandini adari"
print(string.title()) #title() is used to change the string into titlt format
print(string.capitalize()) #capitalize() is used to convert the first letter into
print(string.isalnum()) #isalnum() is used to return true if all the elements ar
print(string.upper()) #upper() is used to convert all the letters into uppercase
print(string.swapcase()) #swapcase() is used to swap the case in the string
Chandini Adari
Chandini adari
False
CHANDINI ADARI
CHANDINI ADARI
#list methods
lst=[21,2,1231.867,3,453,54,3,56.7,75,3,3,6]
print(lst.count(3))  #count() is used to count no.of times repeated in the list
lst.sort() #sort() is used arrange the values in accending order
print(lst)
print(lst.pop()) #pop() is used to remove and return the last element in the list
lst.reverse() #reverse() is used to reverse all elements in the list
print(lst)
4
[2, 3, 3, 3, 3, 6, 21, 54, 56.7, 75, 453, 1231.867]
1231.867
[453, 75, 56.7, 54, 21, 6, 3, 3, 3, 3, 2]
#tuple methods
tup=(2,34,45,24,6,4,43,56,46,6)
print(tup.count(6)) #count() it returns the no.of times it is repeated
print(tup.index(43)) #index() it is used to find the position of the element
print(tuple(sorted(tup))) #sorted is to arrange the elements in assending order
print(max(tup)) #max() is used to find the maximum element in the tuple
2
6
(2, 4, 6, 6, 24, 34, 43, 45, 46, 56)
56
#set methods
set1={12.3,2,4,234,33,54.76,345}
set2={35,56,798,13,456,56,56}
set1.add(5476) #add() it is used to add the element to the set
print(set1)
print(set2.union(set1)) 
set1.remove(12.3) #remove() is used to remove a paticular element in the set
print(set1)
print(set1.difference(set2)) #difference() is used to show the unique in both the sets
print(set2.pop()) #pop() is used to remove the last element in the set
{33, 2, 4, 5476, 234, 12.3, 54.76, 345}
{33, 2, 35, 4, 5476, 456, 234, 12.3, 13, 54.76, 56, 345, 798}
{33, 2, 4, 5476, 234, 54.76, 345}
{33, 2, 5476, 4, 234, 54.76, 345}
35
#dictionary methods
dic={"one":1,"two":2,"three":3,"four":4,"five":5}
print(dic.keys()) #keys() is used to see all the keys in the dict
print(dic.values()) #values() is used to see all the values of keys in the dict
print(dic.items()) #items() is used to return the keys and values in dict
print(dic.get("two")) #get() is used to get the value of a particular key
print(dic.pop("five")) #pop is used to remove the element
dict_keys(['one', 'two', 'three', 'four', 'five'])
dict_values([1, 2, 3, 4, 5])
dict_items([('one', 1), ('two', 2), ('three', 3), ('four', 4), ('five', 5)])
2
5
Section 2 - Functions
a group of related statements that performs a specific task.
def add_new(a:str, b): ## a and b are the arguments that are passed. to provide data type hint
                              # def add_new(x: float, y: float) -> float: 
  sum = a + b
  return sum

ans = add_new(1,2) ## intentionally written str, and passed int, to show it doesn't matter. It is just hint
print(ans)
3
asn = add_new(34,54)
asn
88
def check_even_list(num_list):
    
    even_numbers = []
    
    # Go through each number
    for number in num_list:
        # Once we get a "hit" on an even number, we append the even number
        if number % 2 == 0:
            even_numbers.append(number)
        # Don't do anything if its not even
        else:
            pass
    # Notice the indentation! This ensures we run through the entire for loop    
    return even_numbers
Question 1 :
Define a function, which takes in two strings A and B. Reverses the first string A, adds it to B, and returns the final string.
Question 2 :
Given a list having Names, work_hours, and gender, Write a function to print name of the female worker that worked the most hours. Also how much do should she be paid if the pay is $ 20 per hour.
work_hours = [('Abby',100 , 'F'),('Billy',400, 'M'),('Cassie',800,'F'), ('Maggi',600,'F'),('Alex',500,'M'),('Raj',225,'M'),('Penny',920,'F'),('Ben',300,'M')]
Answer : the female worker that worked the most hours is Penny and she should be paid 18400
#Question 1 :
def st(a,b):
    return b+a[::-1]
st("hello","hi")
'hiolleh'
#Question 2 :
data=[('Abby',100 , 'F'),('Billy',400, 'M'),('Cassie',800,'F'), ('Maggi',600,'F'),('Alex',500,'M'),('Raj',225,'M'),('Penny',920,'F'),('Ben',300,'M')]
def work_hours(data):
  more=0
  for i in range(len(data)):
    if data[i][1]>more and data[i][2]=='F':
      more=data[i][1]
  high_paid=more*20
  worker=''
  for i in range(len(data)):
    if data[i][1]==more and data[i][2]=='F':
      worker+=data[i][0]
      break
  print(f"the female worker that worked the most hours is {worker} and she should be paid {high_paid}")
work_hours(data)
the female worker that worked the most hours is Penny and she should be paid 18400
Section 3 - Libraries and Reading data.
Numpy - One of the most used libraries - supports for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
import numpy as np
a = np.array([1,1,2,3,4,5,5,6,1]) ## np.array converts given list to array

b = a>1 ## important comparison operation, where frequently used in manipulation and image processing.

print(b)
print(a[b]) ## [printing only those values in a which are greater than 1]
[False False  True  True  True  True  True  True False]
[2 3 4 5 5 6]
a_range = np.arange(10,19).reshape(3,3) ## create a 3x3 array with values in range 10-19
a_range
array([[10, 11, 12],
       [13, 14, 15],
       [16, 17, 18]])
## Indexing in arrays works same as that of list

a_range[0] # printing all the columns of first row
array([10, 11, 12])
a_range[:,2] #printing all the rows of second column
array([12, 15, 18])
iden = np.eye(3) #idnetity matrix of given size
iden
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
## adding two matrices
summed = a_range + iden
summed
array([[11., 11., 12.],
       [13., 15., 15.],
       [16., 17., 19.]])
### arrays support normal matrix multiplication that you are used to, point-wise multiplication
### and dot product as well.

mul = a_range@iden ## normal multiplication
mul
array([[10., 11., 12.],
       [13., 14., 15.],
       [16., 17., 18.]])
## point wise multiplication
p_mul = a_range * iden
p_mul
array([[10.,  0.,  0.],
       [ 0., 14.,  0.],
       [ 0.,  0., 18.]])
## Transpose of a matrix.

mtx_t = mul.T
mtx_t
array([[10., 13., 16.],
       [11., 14., 17.],
       [12., 15., 18.]])
### Here we are changing the values of last row of the transposed matrix.
### basically point wise multiplying the values of last row with 1,2 and 3

mtx_t[2] = mtx_t[2]*[1,2,3] ## indexing, point wise multiplication and mutation of values
mtx_t
array([[10., 13., 16.],
       [11., 14., 17.],
       [12., 30., 54.]])
## Just like the greater than 1 (a>1) example we saw earlier.
## here we are checking if the elements are divisible by 2 (%), and if they are, then replace by 0.

mtx_t[(mtx_t % 2 == 0)] = 0 ## convert even elements of the matrix to zero.
mtx_t
array([[ 0., 13.,  0.],
       [11.,  0., 17.],
       [ 0.,  0.,  0.]])
Question 3 :
a)Create a 5x5 matrix of the following form,
[[1,1]
[2,2]]
i.e. each row is increasing and has repetive elements.
Hint : you can use hstack, vstack etc.
b) find dot product of the matrix with any matrix. (Figure out the size/ shape of the matrix)
#Question 3 :a)
import numpy as np
mtx1 = np.array([1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5]).reshape(5,5)
mtx1
array([[1, 1, 1, 1, 1],
       [2, 2, 2, 2, 2],
       [3, 3, 3, 3, 3],
       [4, 4, 4, 4, 4],
       [5, 5, 5, 5, 5]])
#Question 3 :b)
mtx2=np.arange(26,51).reshape(5,5)
mtx3 =np.dot(mtx1,mtx2)
mtx3
array([[ 180,  185,  190,  195,  200],
       [ 360,  370,  380,  390,  400],
       [ 540,  555,  570,  585,  600],
       [ 720,  740,  760,  780,  800],
       [ 900,  925,  950,  975, 1000]])
Reading Files
with open ('/content/sample_data/README.md', 'r') as f:
  a = f.readlines()

a ## here a is list of elements/strings each splitted at \n, \n is also part of the list element.
['This directory includes a few sample datasets to get you started.\n',
 '\n',
 '*   `california_housing_data*.csv` is California housing data from the 1990 US\n',
 '    Census; more information is available at:\n',
 '    https://developers.google.com/machine-learning/crash-course/california-housing-data-description\n',
 '\n',
 '*   `mnist_*.csv` is a small sample of the\n',
 '    [MNIST database](https://en.wikipedia.org/wiki/MNIST_database), which is\n',
 '    described at: http://yann.lecun.com/exdb/mnist/\n',
 '\n',
 '*   `anscombe.json` contains a copy of\n',
 "    [Anscombe's quartet](https://en.wikipedia.org/wiki/Anscombe%27s_quartet); it\n",
 '    was originally described in\n',
 '\n',
 "    Anscombe, F. J. (1973). 'Graphs in Statistical Analysis'. American\n",
 '    Statistician. 27 (1): 17-21. JSTOR 2682899.\n',
 '\n',
 '    and our copy was prepared by the\n',
 '    [vega_datasets library](https://github.com/altair-viz/vega_datasets/blob/4f67bdaad10f45e3549984e17e1b3088c731503d/vega_datasets/_data/anscombe.json).\n']
import pandas as pd

df = pd.read_csv('/content/sample_data/california_housing_test.csv','r')
df.head(10) ## pass as argument number of top elements you wish to print. Head is used to have a quick glance and understand the data.
/usr/local/lib/python3.7/dist-packages/IPython/core/interactiveshell.py:3326: FutureWarning: In a future version of pandas all arguments of read_csv except for the argument 'filepath_or_buffer' will be keyword-only
  exec(code_obj, self.user_global_ns, self.user_ns)
longitude,"latitude","housing_median_age","total_	ooms","total_bed	ooms","population","households","median_income","median_house_value"
0	-122.050000,37.370000,27.000000,3885.000000,66...	NaN	NaN
1	-118.300000,34.260000,43.000000,1510.000000,31...	NaN	NaN
2	-117.810000,33.780000,27.000000,3589.000000,50...	NaN	NaN
3	-118.360000,33.820000,28.000000,67.000000,15.0...	NaN	NaN
4	-119.670000,36.330000,19.000000,1241.000000,24...	NaN	NaN
5	-119.560000,36.510000,37.000000,1018.000000,21...	NaN	NaN
6	-121.430000,38.630000,43.000000,1009.000000,22...	NaN	NaN
7	-120.650000,35.480000,19.000000,2310.000000,47...	NaN	NaN
8	-122.840000,38.400000,15.000000,3080.000000,61...	NaN	NaN
9	-118.020000,34.080000,31.000000,2402.000000,63...	NaN	NaN
len(df.columns), df.columns
(3,
 Index(['longitude,"latitude","housing_median_age","total_', 'ooms","total_bed',
        'ooms","population","households","median_income","median_house_value"'],
       dtype='object'))
df.columns[0]
'longitude,"latitude","housing_median_age","total_'
df['longitude,"latitude","housing_median_age","total_'][:5]
0    -122.050000,37.370000,27.000000,3885.000000,66...
1    -118.300000,34.260000,43.000000,1510.000000,31...
2    -117.810000,33.780000,27.000000,3589.000000,50...
3    -118.360000,33.820000,28.000000,67.000000,15.0...
4    -119.670000,36.330000,19.000000,1241.000000,24...
Name: longitude,"latitude","housing_median_age","total_, dtype: object
df = df.rename(columns = {'longitude,"latitude","housing_median_age","total_':'Detail1'}) ##rename column names as at times it makes it easier for us
df.head(3)
Detail1	ooms","total_bed	ooms","population","households","median_income","median_house_value"
0	-122.050000,37.370000,27.000000,3885.000000,66...	NaN	NaN
1	-118.300000,34.260000,43.000000,1510.000000,31...	NaN	NaN
2	-117.810000,33.780000,27.000000,3589.000000,50...	NaN	NaN
df.iloc[:5, 0]  ##iloc - index - 0 to 4 rows and first column only.
0    -122.050000,37.370000,27.000000,3885.000000,66...
1    -118.300000,34.260000,43.000000,1510.000000,31...
2    -117.810000,33.780000,27.000000,3589.000000,50...
3    -118.360000,33.820000,28.000000,67.000000,15.0...
4    -119.670000,36.330000,19.000000,1241.000000,24...
Name: Detail1, dtype: object
import matplotlib
from matplotlib import pyplot as plt
xpoints = np.array([1, 8])
ypoints = np.array([3, 10])

plt.plot(xpoints, ypoints)
plt.show()

xpoints = np.array([1, 2, 6, 8])
ypoints = np.array([3, 8, 1, 10])

plt.plot(xpoints, ypoints)
plt.show()


Creating a dataframe.
Task: Study about other methods of creating dataframe (for example: using Pandas Series, Lists etc.)
import pandas as pd
import numpy as np
values = np.arange(16).reshape(4,4)
values
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]])
dataframe_from_array = pd.DataFrame(values, index = ['a','b','c','d'], columns=['w','x','y','z'] )
dataframe_from_array
w	x	y	z
a	0	1	2	3
b	4	5	6	7
c	8	9	10	11
d	12	13	14	15
dataframe_from_array.loc[['a','b'],['w','x']]
w	x
a	0	1
b	4	5
dataframe_from_array.iloc[:2,:2] ## it needs position as integer
w	x
a	0	1
b	4	5
dataframe_from_array.iloc[1,3] #second row and last column
7
dataframe_from_array.iloc[::2,::2]
w	y
a	0	2
c	8	10
import numpy as np
from matplotlib import pyplot as plt

ys = 200 + np.random.randn(100)
x = [x for x in range(len(ys))]

plt.plot(x, ys, '-')
plt.fill_between(x, ys, 195, where=(ys > 200), facecolor='g', alpha=0.6)

plt.title("Sample Visualization")
plt.show()

## Question 3 : Upload an image to your google drive, Use plt.imread to read image from the google drive and then print that image using plt.imshow


## Answer 3 : 

## 1) make sure drive is loaded and then upload a test image onto your drive
plt.imread('/content/drive/MyDrive/sky image.jpeg')
array([[[  0,  87, 164],
        [  0,  87, 164],
        [  0,  87, 164],
        ...,
        [165, 194, 208],
        [165, 194, 208],
        [164, 193, 207]],

       [[  0,  87, 164],
        [  0,  87, 164],
        [  0,  87, 164],
        ...,
        [165, 194, 210],
        [164, 193, 209],
        [164, 193, 209]],

       [[  0,  87, 164],
        [  0,  87, 164],
        [  0,  87, 164],
        ...,
        [165, 194, 210],
        [164, 193, 209],
        [164, 193, 209]],

       ...,

       [[ 71,  69,  54],
        [ 72,  70,  55],
        [ 70,  71,  55],
        ...,
        [ 10,  20,  21],
        [ 14,  24,  25],
        [ 17,  27,  28]],

       [[ 76,  74,  59],
        [ 73,  74,  58],
        [ 74,  75,  59],
        ...,
        [ 12,  23,  25],
        [ 19,  30,  32],
        [ 25,  35,  37]],

       [[ 76,  77,  61],
        [ 74,  75,  59],
        [ 75,  76,  60],
        ...,
        [ 15,  26,  28],
        [ 24,  35,  37],
        [ 31,  42,  44]]], dtype=uint8)
#print the image 
plt.imshow(plt.imread('/content/drive/MyDrive/sky image.jpeg'))
<matplotlib.image.AxesImage at 0x7fe01f78da90> 
