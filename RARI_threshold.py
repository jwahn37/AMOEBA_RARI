
import numpy as np
import tensorflow as tf

#check ln(x)
print(np.log(2.718))
RARI = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.01, 0.99])
print(RARI)
RARI = 1.0/(RARI) -1
print(RARI)
RARI = np.log(RARI)
print(RARI)
RARI=-RARI
print("result : ", RARI)


z=np.array([0.0,1.0,2.0,3.0,4.0,5.0])
z=np.array([-1.0, -2.0, -3.0, -4.0, -5.0])
#z=np.array([-11.0, -12.0, -13.0, -14.0, -15.0])
print(z)
gz = 1.0/(1.0+np.exp(-z))
print("result : ", gz)
