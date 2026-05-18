import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("TkAgg")

m1 = plt.imread("./data/bizhi_r.jpg")


m1 = plt.imshow(m1)
plt.savefig("./savedata/bizhi.jpg")

plt.show()
