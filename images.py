import numpy as np
import matplotlib.pyplot as plt

# Load the 'r' component data
data = np.fromfile("initial_conditions_r.raw", dtype=np.float64)
data = data.reshape((500, 500))

# Display the 'r' component using Matplotlib
plt.imshow(data, cmap='gray')
plt.title("Initial Conditions (r component)")
plt.colorbar()
plt.show()
