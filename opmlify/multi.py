from multiprocessing import Process
from functools import partial

def singleCount(cnt, name):
	for i in range(100000000):
		cnt += 1
		if i % 2500000 == 0:
			print name, ':', i



cnt = 0
name = ['hyunho1', 'hyunho2']
p1 = Process(target=singleCount, args=(cnt, name[0]))
p2 = Process(target=singleCount, args=(cnt, name[1]))
p1.start()
p2.start()

p1.join()
p2.join()
