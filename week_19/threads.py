from threading import Thread
import os
import math
import time

def calc():
    for i in range(0, 10000000):
        math.sqrt(i)

# def calc():
#     print("Starting execution")
#     time.sleep(1)
#     print("End Execution")

threads = []

for i in range(os.cpu_count()):
    print("registering thread %d" % i)
    threads.append(Thread(target=calc))

if __name__ == '__main__':
    start = time.perf_counter()
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join() # join waits until threads are finished

    finish = time.perf_counter()

    print(f" Total time: {finish - start}sec")