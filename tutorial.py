import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.linear_model import Lasso
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error
from scipy.integrate import solve_ivp
import os

import pysindy as ps

# Ignore matplotlib deprecation warnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Seed the random number generators for reproducibility
np.random.seed(100)

integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12

pysindy_path = r'C:\Users\phili\PycharmProjects\pysindy'

c = 1
nx, nt = (300, 200)
x = np.linspace(0, 10, nx)
t = np.linspace(0, 10, nt)
tv, xv = np.meshgrid(t, x)

u = c / 2 * np.cosh(np.sqrt(c) / 2 * (xv - c * tv)) ** (-2)

u = u.reshape(len(x), len(t), 1)

dt = t[1] - t[0]
u_dot = ps.FiniteDifference(axis=1)._differentiate(u, t=dt)

# Define PDE library that is quadratic in u, and
# third-order in spatial derivatives of u.
library_functions = [lambda x: x, lambda x: x * x]
library_function_names = [lambda x: x, lambda x: x + x]
pde_lib = ps.PDELibrary(library_functions=library_functions,
                        function_names=library_function_names,
                        derivative_order=3, spatial_grid=x,
                        include_bias=True, is_uniform=True)

# Fit the model with different optimizers.
# Using normalize_columns = True to improve performance.
print('STLSQ model: ')
optimizer = ps.STLSQ(threshold=5, alpha=1e-5, normalize_columns=True)
model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
model.fit(u, t=dt)
model.print()

import torch
from torch import Tensor
from torch.nn import Linear, MSELoss, functional as F
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable
import numpy as np

device = torch.device("cpu")
x = torch.ones(3, dtype=torch.float32, device=device, requires_grad=True)
y = torch.tensor([2, 0, 2], dtype=torch.float32, device=device, requires_grad=False)
optimizer = torch.optim.Adam([x], lr=0.001)

n_steps = 100
for step in range(n_steps):
    loss = torch.norm(x - y)
    # Perform optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
x


# define the model
class Polynomial(torch.nn.Module):
    def __init__(self):
        super(Polynomial, self).__init__()
        self.fc1 = Linear(2, 1)

    def forward(self, x):
        return self.fc1(x)


m = 100
x = torch.randn(m, 2, dtype=torch.float32, device=device, requires_grad=False)
y = 7 * x[:, 0] + 3 * x[:, 1]

model = Polynomial()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

loss_list = []
n_steps = 1000
for step in range(n_steps):
    loss = torch.norm(model(x) - y)
    # Perform optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_list.append(loss.detach().numpy())

print(model.fc1.weight)
print(model.fc1.bias)

plt.plot(loss_list)

model(x[0])

z = torch.ones(3, dtype=torch.float32, device=device, requires_grad=True)
optimizer = torch.optim.Adam([z], lr=0.01)

n_steps = 1000
for step in range(n_steps):
    loss = torch.norm(z[0] + torch.matmul(z[1:], x.T) - y)
    # Perform optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
z

m = 100
x = torch.randn(m, 2, dtype=torch.float32, device=device, requires_grad=False)
feature_matrix = torch.stack([torch.ones(m), x[:, 0], x[:, 1], x[:, 0] ** 2, x[:, 0] * x[:, 1], x[:, 1] ** 2])
coeffs = torch.tensor([-2, 7, 3, 0.5, -1, 5], dtype=torch.float32, device=device, requires_grad=False)
y = torch.matmul(coeffs, feature_matrix)
