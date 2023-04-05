from multiprocessing import Process
import os
import math
import time

def calc():
    for i in range(0, 10000000):
        math.sqrt(i)

processes = []

for i in range(os.cpu_count()):
    print("registering process %d" % i)
    processes.append(Process(target=calc))

if __name__ == '__main__':
    start = time.perf_counter()

    for process in processes:
        process.start()

    for process in processes:
        process.join() # join waits until processes are finshed
    
    finish = time.perf_counter()

    print(f" Total time: {finish - start}sec")