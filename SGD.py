import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Загрузка и подготовка данных
california = fetch_california_housing()
X = california.data
y = california.target.reshape(-1, 1)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

# Стандартизиурем данные (приводит к среднему = 0, стандартному отклонению = 1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Добавляем столбец -1 для intercept (w0)
X_train_final = np.concatenate([-np.ones((len(X_train_scaled), 1)), X_train_scaled], axis=1)
X_test_final = np.concatenate([-np.ones((len(X_test_scaled), 1)), X_test_scaled], axis=1)

def manual_r2_score(y_true, y_pred):
    y_mean = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_mean)**2)
    return 1 - (ss_res / ss_tot)

def compute_cost(X, y, weights, tau):
    m = len(y)
    y_pred = X.dot(weights)
    error = y_pred - y
    mse = (1/(2*m)) * np.sum(error**2)
    regularization = (tau/2) * np.sum(weights[1:]**2)
    return mse + regularization

def compute_gradient(X_i, y_i, weights, tau):
    error = X_i.dot(weights) - y_i
    grad = X_i.T.dot(error) + tau * np.vstack([np.zeros((1, 1)), weights[1:]])
    return grad

def sgd_ridge(X, y,lambda_, tau=0.1, eps=1e-5, max_iter=1000):
    n_features = X.shape[1]
    weights = np.zeros((n_features, 1))
    
    Q_history = []
    Q_prev = 0
    Q_ema = 0
    
    for i in range(1, max_iter+1):
        rand_idx = np.random.randint(0, len(y))
        X_i = X[rand_idx:rand_idx+1]
        y_i = y[rand_idx:rand_idx+1]
        
        h = 1.0 / i
        grad = compute_gradient(X_i, y_i, weights, tau)
        weights = weights * (1 - h * tau) - h * grad
        
        Q_current = compute_cost(X_i, y_i, weights, tau)
        Q_ema = lambda_ * Q_current + (1 - lambda_) * Q_ema if i > 1 else Q_current
        
        
        Q_history.append(Q_ema)
        
        if i > 10 and abs(Q_ema - Q_prev) < eps:
            print(f"Ранняя остановка на итерации {i}")
            break
            
        Q_prev = Q_ema
    
    return weights, Q_history

def predict(X, weights):
    return X.dot(weights)

# Параметры
lambda_ = 0.9
eps = 1e-6

# Подбор оптимального tau
tau_values = np.logspace(-5, 2, 50)
r2_scores = []
best_r2 = -1
best_tau = 0
best_weights = None
best_Q_history = None

for tau in tau_values:
    weights, Q_history = sgd_ridge(X_train_final, y_train, lambda_=lambda_, tau=tau, eps=eps, max_iter=1000)
    y_pred = predict(X_test_final, weights)
    r2 = manual_r2_score(y_test, y_pred)
    if r2 > -10:
        r2_scores.append(r2)
    else:
        r2_scores.append(np.nan)
        
    if r2 > best_r2:
        best_r2 = r2
        best_tau = tau
        best_weights = weights
        best_Q_history = Q_history

print(f"Лучший коэффициент регуляризации tau: {best_tau:.6f}")
print(f"Лучший R^2 на тестовой выборке: {best_r2:.4f}")
print("Оптимальные веса модели:", best_weights.flatten())

# Графики
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(best_Q_history)
plt.xlabel("Iterations")
plt.ylabel("Q (EMA)")
plt.title("SGD Convergence")

plt.subplot(1, 3, 2)
plt.semilogx(tau_values, r2_scores, 'o-')
plt.axvline(best_tau, color='r', linestyle='--')
plt.xlabel("tau (log scale)")
plt.ylabel("R^2 score")
plt.title("Model Performance vs Regularization")

plt.subplot(1, 3, 3)
y_pred = predict(X_test_final, best_weights)
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("True vs Predicted Values")

plt.tight_layout()
plt.show()

