import random
import numpy as np
import torch

test = ["1","2","3"]
real_message = "hello"
random_item = random.choice("hello")
print(random_item)
fake_message = real_message[1:len(real_message)] + random_item

temp = test.pop()
print(temp)


test2 = np.array([[0,1,1], [0,1,1], [0,1,1]])
test3 = test2.tolist()
print(type(test3[0]))

count_ones=torch.count_nonzero(torch.tensor([1,0,0]))

if(count_ones == 1):
    print("hereeee")



