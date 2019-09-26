#%%
import numpy as np
import matplotlib.pyplot as plt
import pyfftw

a = np.linspace(0, 1)
plt.plot(a, np.sin(a))
plt.show()