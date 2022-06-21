from threading import Thread
import os
import math
import time

def calc():
    print("Starting execution")
    time.sleep(1)
    print("End Execution")

# threads = []

# for i in range(os.cpu_count()):
#     print("registering thread %d" % i)
#     threads.append(Thread(target=calc))

if __name__ == '__main__':
    start = time.perf_counter()

    for i in range(os.cpu_count()):
        calc()

    finish = time.perf_counter()

    print(f" Total time: {finish - start}sec")