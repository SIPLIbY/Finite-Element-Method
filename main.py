import random

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

a = 1.0
b = 1.2

delta = 10**(-5)
h = 0.02
t = 0.08



def f(x, y):
    return 0.2*np.exp(x)*np.cos(y)

def u(x, y):
    return np.exp(x)*np.cos(y)

#найти решение системы уравнений методом простой итерации. вывести на экран график u(x, y) в точках пунктирной прямой

N = int(1/h)

def get_indices():
    # i - x, j - y
    idx = set()
    N_2 = int(N/2)

    for i in range(N_2):
        ind = N - i
        for j in range(0, ind):
            idx.add((i,j))


    for i in range(N_2, N):
        for j in range(N_2+1):
            idx.add((i, j))

    return idx

def get_boundary():
    bound = set()
    for j in range(N):
        bound.add((0, j))
        bound.add((j, 0))

    for i in range(round(N / 2) + 1):
        bound.add((N, i))

    for i in range(round(N / 2) + 1, N):
        bound.add((i, round(N / 2)))

    for i in range(round(N / 2) + 1):
        ind = N - i
        bound.add((i, ind))

    return bound

def dot_prod(A, B):
    res = 0
    for i in range(N):
        for j in range(N):
            res += A[i][j]*B[i][j]
    return res

def norma(A):
    return pow(dot_prod(A, A), 0.5)

def normalise(A):
    norm_val = norma(A)
    try:
        A = np.divide(A, norm_val)
    except ZeroDivisionError:
        print("Исключение ZeroDivisionError.")
    return A

#матрица решений
def solution_matrix(s, bound):
    U = np.zeros([N+1, N+1])
    for k in s:
        x, y = k
        U[x][y] = u(x*h, y*h)
    for k in bound:
        x, y = k
        U[x][y] = u(x*h, y*h)
    return U

def operator_step(U, s):
    Y = np.zeros([N+1, N+1])
    for k in s:
        i, j = k
        Y[i][j] = - a * (U[i - 1][j] - 2 * U[i][j] + U[i + 1][j]) / pow(h, 2) - b * (U[i][j - 1] - 2 * U[i][j] + U[i][j + 1]) / pow(h, 2)
    return Y

def operator_step_f(U, s):
    Y = np.zeros([N + 1, N + 1])
    for k in s:
        i, j = k
        Y[i][j] = - a * (U[i - 1][j] - 2 * U[i][j] + U[i + 1][j]) / pow(h, 2) - b * (U[i][j - 1] - 2 * U[i][j] + U[i][j + 1]) / pow(h, 2) - f(i * h, j * h)
    return Y

def operator_step_reversed(U, lambda_max, s):
    Y = np.zeros([N+1, N+1])
    for k in s:
        i, j = k
        Y[i][j] = lambda_max * U[i][j] + a * (U[i - 1][j] - 2 * U[i][j] + U[i + 1][j]) / pow(h, 2) + b * (U[i][j - 1] - 2 * U[i][j] + U[i][j + 1]) / pow(h, 2)
    return Y


def jacobi_step_operator(U, s, bound, tau):
    Y = np.zeros([N + 1, N + 1])
    for k in s:
        i, j = k
        Y[i][j] = U[i][j] - tau * (- a * (U[i - 1][j] - 2 * U[i][j] + U[i + 1][j]) / pow(h, 2) - b * (U[i][j - 1] - 2 * U[i][j] + U[i][j + 1]) / pow(h, 2) - f(i * h, j * h))
    for k in bound:
        i, j = k
        Y[i][j] = u(i*h, j*h)
    return Y

def min_max_eigenvalues(delta, s):
    Y = np.zeros([N + 1, N + 1])
    for k in s:
        i, j = k
        Y[i][j] = 1
    Y = normalise(Y)
    Z = operator_step(Y, s)

    lambda_prev = 1
    lambda_max = dot_prod(Z, Y)
    max_count = 0

    while abs(1 - lambda_max / lambda_prev) > delta:
        max_count += 1
        Y = Z
        Y = normalise(Y)
        Z = operator_step(Y, s)

        lambda_prev = lambda_max
        lambda_max = dot_prod(Y, Z)

    # минимальное собств значение
    Y = np.zeros([N + 1, N + 1])
    for k in s:
        i, j = k
        Y[i][j] = 1
    Y = normalise(Y)
    Z = operator_step_reversed(Y, lambda_max, s)

    lambda_prev = lambda_max
    lambda_min = dot_prod(Z, Y)
    min_count = 0
    if norma(Z) != 0:
        while abs(1 - lambda_min / lambda_prev) > delta:
            min_count += 1
            Y = np.copy(Z)
            Y = normalise(Y)
            Z = operator_step_reversed(Y, lambda_max, s)
            lambda_prev = lambda_min
            lambda_min = dot_prod(Y, Z)

    lambda_min = lambda_max - lambda_min
    return [lambda_max, max_count, lambda_min, min_count]

def next_step(U, i, j):
    return -a * (U[i - 1][j] - 2 * U[i][j] + U[i + 1][j]) / pow(h, 2) - b * (U[i][j - 1] - 2 * U[i][j] + U[i][j + 1]) / pow(h, 2)


#метод Якоби

def jacobi_method(delta, s, bound, tau):
    U = np.zeros([N+1, N+1])
    for k in s:
        i, j = k
        U[i][j] = u(i*h, j*h)
    for k in bound:
        i, j = k
        U[i][j] = 1

    count = 0
    Z = jacobi_step_operator(U, s, bound, tau)
    while norma(Z - U) > delta:
        count += 1
        U = np.copy(Z)
        Z = jacobi_step_operator(U, s, bound, tau)
    return Z, count

indices = get_indices()
bound = get_boundary()
U = solution_matrix(indices, bound)


print("Epsilon = ", delta)
print("h = ", h)

lam = min_max_eigenvalues(delta, indices)
print("Max eigenvalue: ", lam[0])
print("Number of iterations: ", lam[1])
print("Min eigenvalue: ", lam[2])
print("Number of iterations: ", lam[3])
tau = 1 / (lam[0] + lam[2])



start_time = time.time()

Sol_simp, c = jacobi_method(delta, indices, bound, tau)
print('\n')
print("Error: ", norma(Sol_simp - U)*h)
print("Count: ", round(c/5))
print("--- %s seconds ---" % (time.time() - start_time))

X = []
Y = []
for k in indices:
    xi, yi = k
    X.append(xi * h)
    Y.append(yi * h)
for k in bound:
    xi, yi = k
    X.append(xi * h)
    Y.append(yi * h)
X = list(set(X))
Y = list(set(Y))
X.sort()
Y.sort()

#Y2D = Sol_simp[10,:]
#plt.plot(X, Y2D)
#plt.ylim([0, 2])
#plt.show()

X, Y = np.meshgrid(X, Y)
print(Y.shape)
print(X.shape)
#print(X)
#print(Y)

fig = plt.figure()
ax1 = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax1)
ax1.plot_surface(X, Y, Sol_simp, cmap='magma')
#plt.show()

fig2 = plt.figure()
ax2 = Axes3D(fig2, auto_add_to_figure=False)
fig2.add_axes(ax2)
ax2.plot_surface(X, Y, U, cmap='inferno')
plt.show()

