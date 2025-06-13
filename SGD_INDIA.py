import numpy as np 
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler  # Добавим масштабирование
import matplotlib.pyplot as plt

# Загрузка данных
california = fetch_california_housing()
X = california.data
Y = california.target.reshape(-1, 1)  # Y должен быть 2D (n_samples, 1)

# Масштабирование данных (очень важно для SGD!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Добавляем столбец единиц для intercept (bias)
X_final = np.concatenate([np.ones((len(X_scaled), 1)), X_scaled], axis=1)

# Параметры обучения
theta_init = np.zeros((X_final.shape[1], 1))  # Инициализация весов
learning_rate = 0.01  # Уменьшим learning rate, если нужно
num_iterations = 1000  # Увеличим число итераций

def cost_function(X, Y, theta):
    m = len(Y)
    y_hat = X.dot(theta)
    J = (1/(2*m)) * np.sum((y_hat - Y)**2)  # MSE с 1/2 для удобства
    return J

def gradient(X, Y, theta):
    m = len(Y)
    y_hat = X.dot(theta)
    grad = (1/m) * X.T.dot(y_hat - Y)
    return grad

def stochastic_gradient_descent(X, Y, theta_init, learning_rate, num_iterations):
    theta = theta_init.copy()
    J_history = []
    
    for i in range(num_iterations):
        rand_index = np.random.randint(0, len(Y))
        X_i = X[rand_index:rand_index+1]  # Берем одну строку (сохраняя 2D форму)
        Y_i = Y[rand_index:rand_index+1]
        
        grad = gradient(X_i, Y_i, theta)
        theta -= learning_rate * grad
        
        # Сохраняем стоимость (можно делать реже, чтобы ускорить)
        if i % 10 == 0:
            J_history.append(cost_function(X, Y, theta))
    
    return theta, np.array(J_history)

# Запуск SGD
theta, J_history = stochastic_gradient_descent(X_final, Y, theta_init, learning_rate, num_iterations)

# График сходимости
plt.plot(J_history)
plt.xlabel("Iterations (x10)")
plt.ylabel("Cost")
plt.title("SGD Convergence")
plt.show()

print("Оптимальные веса:", theta[1:].flatten())  # Исключаем bias (θ₀)
print("Оптимизированное смещение (θ₀):", theta[0][0])
print("Минимальное значение функции:", J_history[-1])