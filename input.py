import numpy as np
import matplotlib.pyplot as plt

images = []
for i in range(10):
    for j in range(1, 289):
        img = plt.imread('./devnum/{}/{}.jpg'.format(i, j)).flatten()
        images.append(img)

        print('Digit: ', i)
        print("\r" + "{}% |".format(int(100 * i / 288) + 1) + '#'*int((int(100 * i / 288) + 1)/5) +
          ' '*(20 - int((int(100 * i / 288) + 1)/5)) + '|',
          end="") if not i % (288 / 100) else print("", end="")


images = np.array(images)
labels = np.zeros(2880).reshape(2880, 1)

j = 0
for i in range(10):
    labels[j:j+j+288] = i
    j += 288

mean = images.mean(axis=0)
sigma = images.max(axis=0) - images.min(axis=0)
images = (images - mean) / sigma

np.save('data_normalized', images)
np.save('labels', labels)
