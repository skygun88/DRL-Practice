import torch
from collections import deque
import numpy as np

# a = [1,2,3,4]
# a_tensor = torch.tensor([a,a])
# print(a_tensor)

# b = [6,7,8,9]
# b_tensor = torch.tensor(b)
# a_tensor[0,:] = b_tensor
# print(a_tensor)

# c_tensor = torch.empty_like(a_tensor)
# c_tensor[:] = a_tensor[:]

# print(a_tensor, c_tensor)

# a_tensor[0, 3] = 99
# c_tensor[1, 2] = -99
# b_tensor[0] = -50
# print(a_tensor)
# print(b_tensor)
# print(c_tensor)
# # for i in range(4): b[i] = a[i]+1
# # print(b)
# d = list(range(10))
# print(d[:10])

# e = []
# e.append((a_tensor, c_tensor))

# print(e)

# a_tensor[0, 0] = -44

# print(a_tensor)
# print(e)

# f_tensor = torch.tensor([False, True, True, False])
# print(torch.sum(f_tensor))
# if 0.0001%10000:
#     print('hello')

# import time
# from tkinter import X
# import tracemalloc

# tracemalloc.start()
# my_snapshot = None

# a = [[1,2,3,4]]*10000000
# print(a[:10])

# time1 = tracemalloc.take_snapshot()

# start = time.time()
# b = list(map(lambda x: x[0], a))
# print(time.time()-start)

# start = time.time()
# c = [x[0] for x in a]
# print(time.time()-start)

# time2 = tracemalloc.take_snapshot()
# stats = time2.compare_to(time1, 'lineno')
# for stat in stats[:10]:
#     print(stat)

k = deque([], maxlen=10)
for i in range(5):
    k.append(i)
array = np.array(k)

print(k, array, type(array), array.shape, len(k))

k.clear()
print(k)