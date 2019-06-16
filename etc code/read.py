import pickle
import sys

# use -> python read.py 0000

with open('./'+str(sys.argv[1])+'.pkl','rb') as f:
    data = pickle.load(f)

print(data)


f2 = open('./'+str(sys.argv[1])+'.txt', 'w')
f2.write(str(data))
f2.close() 
