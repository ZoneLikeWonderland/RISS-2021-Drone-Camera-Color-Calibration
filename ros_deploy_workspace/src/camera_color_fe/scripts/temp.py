import numpy as np
import matplotlib.pyplot as plt

a1 = np.array([
    [10, 15, 8],
    [41, 67, 32],
    [92, 163, 77],
    [171, 255, 145],
    [253, 254, 222],
    [254, 254, 254],
])
diff = a1[1:] - a1[:-1]
diff *= 1 if diff.mean() > 0 else - 1
crstd = np.std(a1 / a1.mean(axis=1)[..., None], axis=0).sum()

plt.figure()
plt.plot(a1)
plt.title(str(crstd))


a2 = np.array([
    [42, 67, 32],
    [55, 190, 52],
    [181, 254, 76],
    [244, 250, 104],
])
diff = a2[1:] - a2[:-1]
diff *= 1 if diff.mean() > 0 else - 1
crstd = np.std(a2 / a2.mean(axis=1)[..., None], axis=0).sum()

plt.figure()
plt.plot(a2)
plt.title(str(crstd))

plt.show()