# import numpy as np
#
# def f(x, y):
#     return x * (1.98 * np.sin((0.1*x + 0.051*x*x) / 1.98)) - (y * 10**-7 + 0.1)
#
# def df(x):
#     return 1.98 * np.sin((0.1*x + 0.051*x*x) / 1.98) + x * (1.98 * np.cos((0.1*x + 0.051*x*x) / 1.98)) * (0.1 + 0.102 * x) / 1.98
#
# def newton_raphson(y, x0=1.0, tol=1e-9, max_iter=100):
#     x = x0
#     iter_count = 0
#
#     while np.abs(f(x, y)) > tol and iter_count < max_iter:
#         x = x - f(x, y) / df(x)
#         iter_count += 1
#
#     if iter_count == max_iter:
#         print("Maximum number of iterations reached without convergence.")
#
#     return x
#
#
# Y = 0.5
# X = str(round(newton_raphson(float(Y)), 4))
# print("X =", X)


# 二分法

# import math
#
#
# def equation(x):
#     return ((1.98 + x) / (0.1 * x + 1e-9)) * math.sin((0.1 * x + 0.051 * x ** 2) / (1.92 + x)) - 1
#
#
# def find_root(y, tolerance=1e-6, max_iterations=1000):
#     lower_bound = -100  # 选择一个较大的负数作为下界
#     upper_bound = 100  # 选择一个较大的正数作为上界
#
#     for _ in range(max_iterations):
#         x = (lower_bound + upper_bound) / 2
#         y_approx = equation(x)
#
#         if abs(y_approx - y) < tolerance:
#             return x
#
#         if y_approx < y:
#             lower_bound = x
#         else:
#             upper_bound = x
#
#     raise ValueError("未找到方程的根。")
#
#
# # 测试函数
# y_value = 1000 * 10 ** -6
# result = find_root(y_value)
# print(f"方程的解为 x = {result}")

# 牛顿法
# import math
#
#
# def f(x, y):
#     return y * 10 ** -6 - ((1.98 + x) / (0.1 * x)) * math.sin((0.1 * x + 0.051 * x ** 2) / (1.92 + x)) - 1
#
#
# def f_prime(x):
#     return (0.1 * (0.051 * x ** 2 + 0.1 * x + 1.92) * math.cos((0.1 * x + 0.051 * x ** 2) / (1.92 + x)) - (1.98 + x) * (
#                 1.92 + x) * math.sin((0.1 * x + 0.051 * x ** 2) / (1.92 + x)) / (0.1 * x) ** 2) / (1.92 + x)
#
#
# def newton_method(y, x0, tol=1e-6, max_iter=100):
#     x = x0
#     for i in range(max_iter):
#         x_next = x - f(x, y) / f_prime(x)
#         if abs(x_next - x) < tol:
#             return x_next
#         x = x_next
#     return None
#
#
# # 示例使用
# y_value = 1000  # 替换为你的已知 y 值
# initial_guess = 1.0  # 替换为你的初始猜测
# result = newton_method(y_value, initial_guess)
#
# if result is not None:
#     print("解 x =", result)
# else:
#     print("无法收敛到解，请尝试不同的初始猜测或增大迭代次数。")


# from scipy.optimize import fsolve
# import numpy as np
#
# def equation(x):
#     y = 1000
#     return (y * 1e-6) - (((1.98 + x) / (0.1 * x)) * np.sin((0.1 * x + 0.051 * x ** 2) / (1.92 + x)) - 1)
#
# # 使用fsolve函数求解方程的解
# x_solution = fsolve(equation, x0=1)
#
# print("X的解为:", x_solution)


# from sympy import symbols, Eq, sin, solve
#
# # Step 1: 将x表示为符号
# x = symbols('x')
#
# # Step 2: 将方程整理成标准形式
# equation = Eq((1.98 + x) / (0.1 * x) * sin((0.1 * x + 0.051 * x**2) / (1.92 + x)) - 1, 1000 * 10**-6)
#
# # Step 3: 使用SymPy的solve函数求解方程
# solutions = solve(equation, x)
#
# # 输出结果
# print("解:")
# for sol in solutions:
#     print(f"x = {sol:.4f}")

# import numpy as np
# from scipy.optimize import curve_fit
#
# # 定义原方程 f(x) = 0
# def f(x, y):
#     return ((1.98 + x) / (0.1 * x)) * np.sin((0.1 * x + 0.051 * x ** 2) / (1.92 + x)) - y * 1e-6 - 1
#
# # 已知的y值，用于求得对应的x值
# known_y = 1234  # 将这里的值替换为已知的y值
#
# # 定义回归模型函数
# def regression_model(x, a, b, c):
#     return ((a + x) / (b * x)) * np.sin((b * x + c * x ** 2) / (a + x))
#
# # 使用curve_fit函数进行回归拟合
# x_data = np.linspace(0.1, 10, 100)  # 调整范围和点数以适合数据
# y_data = f(x_data, known_y) * 1e6  # 用已知的y值计算对应的数据点，并还原回原来的规模
#
# params, _ = curve_fit(regression_model, x_data, y_data)
#
# # 从回归拟合中提取参数
# a_fit, b_fit, c_fit = params
#
# # 定义一个函数来近似求解x给定y的值
# def approximate_x(y):
#     # 重新组织回归模型的方程，解出x
#     return (a_fit + np.sqrt(a_fit**2 + 4 * b_fit * y * np.sin(c_fit / a_fit))) / (2 * b_fit)
#
# # 使用回归模型求得对应的x值
# approx_x = approximate_x(known_y)
#
# print("近似得到的x值:", approx_x)


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import fsolve
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from math import sin
#
# # 定义方程
# def equation(y, x):
#     return x * 1e-6 - (((1.98 + y) / (0.1 * y)) * sin((0.1 * y + 0.051 * y ** 2) / (1.92 + y)) - 1)
#
# # 生成一系列x值
# x_values = np.linspace(0, 1500, num=100)
#
# # 计算对应的y值
# y_values = []
# for x in x_values:
#     y_initial_guess = 1.0
#     y_solution = fsolve(equation, y_initial_guess, args=(x,))
#     y_values.append(y_solution[0])
#
# # 转换成数组
# x_values = np.array(x_values).reshape(-1, 1)
# y_values = np.array(y_values)
#
# # 使用多项式回归进行拟合
# poly_features = PolynomialFeatures(degree=5)  # 这里可以调整多项式的阶数
# x_poly = poly_features.fit_transform(x_values)
#
# # 创建线性回归模型并进行拟合
# model = LinearRegression()
# model.fit(x_poly, y_values)
#
# # 指定需要预测y值的x值
# x_to_predict = 1000  # 这里可以替换成其他值
#
# # 对x值进行多项式转换
# x_to_predict_poly = poly_features.transform(np.array(x_to_predict).reshape(-1, 1))
#
# # 预测对应的y值
# y_predicted = model.predict(x_to_predict_poly)[0]
#
# print(f"For x = {x_to_predict}, the approximate y value is: {y_predicted}")

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
#
# # 假设这是你有的实际数据点
# x_data = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
# y_data = np.array([0.01, 0.02, 0.03, 0.05, 0.07, 0.09, 0.12, 0.15, 0.18, 0.21])
#
# # 将x_data转换为二维数组
# X = x_data.reshape(-1, 1)
#
# # 初始化线性回归模型
# model = LinearRegression()
#
# # 拟合模型
# model.fit(X, y_data)
#
# # 得到回归方程的系数和截距
# coefficients = model.coef_
# intercept = model.intercept_
#
# # 打印回归方程
# print(f"回归方程：y = {coefficients[0]:.4f} * x + {intercept:.4f}")
#
# # 使用拟合的模型进行预测
# x_new = np.array([1100, 1200, 1300, 1400, 1500]).reshape(-1, 1)
# y_pred = model.predict(x_new)
#
# # 打印预测值
# print("预测值:")
# for x_val, y_val in zip(x_new, y_pred):
#     print(f"x = {x_val[0]}, y ≈ {y_val:.4f}")
#
# # 绘制拟合曲线和实际数据点
# plt.scatter(x_data, y_data, label="实际数据")
# plt.plot(x_new, y_pred, color='red', label="拟合曲线")
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.metrics import mean_squared_error
#
# def generate_data(n_samples):
#     np.random.seed(0)
#     x = np.random.uniform(0, 1500, n_samples) * 10**-6
#     y = ((1.98 + x) / (0.1 * x)) * np.sin((0.1 * x + 0.051 * x**2) / (1.92 + x)) - 1
#     noise = np.random.normal(0, 0.1, n_samples)
#     y_noisy = y + noise
#     return x, y_noisy
#
# def predict_y(x_value, model, poly_features):
#     # Convert x_value to a 2D array
#     x_value = np.array(x_value).reshape(-1, 1)
#
#     # Transform x_value to polynomial features
#     x_value_poly = poly_features.transform(x_value)
#
#     # Predict y value using the fitted model
#     y_pred = model.predict(x_value_poly)
#
#     return y_pred[0]
#
# n_samples = 100
# x, y = generate_data(n_samples)
#
# # Reshape x to a column vector
# x = x.reshape(-1, 1)
#
# # Create polynomial features
# poly_features = PolynomialFeatures(degree=2)
# x_poly = poly_features.fit_transform(x)
#
# # Fit the polynomial regression model
# model = LinearRegression()
# model.fit(x_poly, y)
#
# # Predict for a given x value
# x_value = 1000 * 10**-6
# predicted_y = predict_y(x_value, model, poly_features)
# print("Predicted y value:", predicted_y)
#
# # Visualize the data and the fitted curve
# x_plot = np.linspace(0, 1500 * 10**-6, 100).reshape(-1, 1)
# x_plot_poly = poly_features.transform(x_plot)
# y_plot = model.predict(x_plot_poly)
#
# plt.scatter(x, y, label='Data points')
# plt.plot(x_plot, y_plot, color='red', label='Polynomial fit (degree=2)')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.show()

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def y_equation(x):
    return ((1.98 + x) / (0.1 * x)) * np.sin((0.1 * x + 0.051 * x ** 2) / (1.92 + x)) - 1


def generate_data(n_samples):
    np.random.seed(0)
    x = np.random.uniform(low=0, high=1500, size=n_samples)
    y = y_equation(x)
    # 加入一些随机噪声以模拟真实数据
    y += np.random.normal(loc=0, scale=0.1, size=n_samples)
    y = np.clip(y, 0, None)  # Clip y values to ensure they are non-negative
    return x, y


def predict_y(x_pred, degree):
    x, y = generate_data(100)

    poly_features = PolynomialFeatures(degree=degree)
    x_poly = poly_features.fit_transform(x.reshape(-1, 1))

    model = LinearRegression()
    model.fit(x_poly, y)

    x_pred_poly = poly_features.transform(x_pred.reshape(-1, 1))
    y_pred = model.predict(x_pred_poly)
    y_pred = np.clip(y_pred, 0, None)  # Clip predicted y values to ensure they are non-negative
    return y_pred

x_values = np.linspace(0, 1500, 1500)  # 创建x值的数组
y_predicted_values = predict_y(x_values, degree=3)  # 使用三次多项式回归进行预测

# 绘制预测结果
plt.plot(x_values, y_predicted_values, label='Predicted y')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

