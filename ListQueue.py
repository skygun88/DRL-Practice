import random

class listqueue:
    def __init__(self, capcacity):
        self.l = [None]*capcacity
        self.capacity = capcacity
        self.anchor = 0
        self.used = 0
    
    def put(self, x):
        if self.used < self.capacity:
            self.used += 1
        self.l[self.anchor] = x
        self.anchor = (self.anchor + 1)%self.capacity
            
    def sample(self, n):
        return random.sample(self.l[:self.used])

    def to_list(self):
        return self.l[self.anchor:self.used]+self.l[0:self.anchor]

    def __str__(self):
        return f"{self.l[self.anchor:self.used]+self.l[0:self.anchor]}"

if __name__ == '__main__':
    lq = listqueue(capcacity=100)
    for i in range(50):
        lq.put(i)
    # print(lq.l)
    print(lq)

    l = lq.to_list()
    print(l)