import sys
import matplotlib
import numpy
from numpy import random
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
numpy.random.seed(2)

from scipy import stats

x = [1,2,3,4,5,6,7,8,9,10]
y = [10,15,20,25,30,25,30,35,40,50]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

myline = numpy.linspace(1, 10, 50)

plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()

'''
#training and testing data, train 80%, test 20%
train_x = x[:80]
train_y = y[:80]
test_x = x[80:]
test_y = y[80:]
'''


'''
slope, intercept, r, p, std_err = stats.linregress(x, y)

#0 = no relationship, 1/-1 = full relationship
print("Relationship: ", r)

def linearregression(x):
    return slope * x + intercept
mymodel = list(map(linearregression, x))
'''


'''
#x, y axis labels and x label rotation
plt.ylabel('Y')
plt.xlabel('X')
plt.xticks(rotation=45)

plt.scatter(train_x, train_y)
plt.plot(x, mymodel)
plt.show()

plt.savefig(sys.stdout.buffer)
sys.stdout.flush()
'''