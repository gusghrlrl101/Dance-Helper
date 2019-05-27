import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

x=[0.0,0.25,-0.25,0.5,-0.5]
y=[2.50,5.68,9.00,12.2,15.0]


xp=np.arange(-0.1,0.5,0.05)
 
a11=sum([xx**2 for xx in x])
a12=sum([xx for xx in x])
a21=sum([xx for xx in x])
a22=sum([1 for xx in x])
b2=sum([yy for yy in y])
b1=0.0
for ii in range(len(x)):
	b1=b1+x[ii]*y[ii]
 
A=np.array(((a11,a12),(a21,a22)))
b=np.reshape(np.array((b1,b2)),(2,1))
 
X=np.linalg.solve(A,b)
 
print X[0], X[1]
 
 #a=31.54
 #b=2.57
 
yp=[X[0]*xx+X[1] for xx in xp]
fig=plt.figure(0)
ax=fig.add_subplot(111)
plt.plot(x,y,'o',xp,yp)
ax.axis([-0.1,0.45,-0.5,15.5])
plt.show()


