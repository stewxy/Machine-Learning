import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from scipy import stats

x=[1,2,3,4,5,6,7,8,9,10]
y=[1,2,3,4,5,6,7,8,9,15]

slope, intercept, r, p, std_err = stats.linregress(x, y)

#0 = no relationship, 1/-1 = full relationship
print("Relationship: ", r)

def linearregression(x):
    return slope * x + intercept

mymodel = list(map(linearregression, x))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()

plt.savefig(sys.stdout.buffer)
sys.stdout.flush()
