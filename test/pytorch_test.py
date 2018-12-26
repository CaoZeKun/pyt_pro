import numpy as np
import torch
import matplotlib.pyplot as plt


a = [0,5,10]
b = [1,6,12]

plt.plot(a,b)
plt.show()

I = np.loadtxt('../data/test.txt')
print(I)

aa = ['abc', 'dsd', 'sss']
aa = np.array(aa)
print(aa)
# np.savetxt('../data/test.txt',aa,fmt='%s')


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
# I = mpimg.imread('../data/hymenoptera_data/train/ants/0013035.png')
# print(I.shape)
# plt.imshow(I)