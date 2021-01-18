import numpy as np
N = 12000
epsilon = 0.1


ans = 4*2*(2*N)*np.exp((epsilon**2)*N*(-1/8))

print(ans)
