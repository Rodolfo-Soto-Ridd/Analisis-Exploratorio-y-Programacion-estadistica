import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# 1. Creación de Datos Simulados (3 puntos)
# ● Generar dos listas de datos numéricos simulados que representen variables relacionadas
# (por ejemplo, temperatura ambiente y consumo de energía).
# ● Hay que asegurar que los datos tengan cierta variabilidad y una relación lineal aproximada.
np.random.seed(42)
temperatura = np.random.uniform(15, 35, 100)  # 100 valores entre 15°C y 35°C
ruido = np.random.normal(0, 5, 100)  # ruido para simular variabilidad
consumo_energia = 100 - 2 * temperatura + ruido  # relación lineal inversa
X = temperatura.reshape(-1, 1)
y = consumo_energia

# 2. Implementación del Modelo de Regresión Lineal (3 puntos)
# ● Utilizar la librería scikit-learn para ajustar un modelo de regresión lineal simple.
# ● Obtener e imprimir los coeficientes de la regresión (intercepto y pendiente).
modelo=LinearRegression()
modelo.fit(X,y)
beta_0=modelo.intercept_
beta_1=modelo.coef_[0]
print(f"Intercepto con Y ={beta_0:2f}")
print(f"Intercepto con Y ={beta_1:2f}")
# 3. Predicción de Valores con el Modelo (2 puntos)
# ● Usar el modelo entrenado para hacer predicciones sobre los datos.
# ● Guardar los valores predichos en una lista y mostrar los primeros 5 resultados.
Y_pred=modelo.predict(X)
print(f"El valor predicho es de:{Y_pred[:5]}")
# 4. Evaluación del Modelo (2 puntos)
# ● Calcular métricas de error como el Error Cuadrático Medio (MSE) y el Error Absoluto Medio (MAE).
# ● Interpretar los resultados y comentar el ajuste del modelo.
mse = mean_squared_error(y,Y_pred)
print(f"Error Cuadrático Medio (MSE):{mse}")
mae=mean_absolute_error(y,Y_pred)
print(f"Error Absoluto Medio (MAE):{mae}")
# Interpretación
# MSE y MAE bajos indican un buen ajuste del modelo.
# Como el modelo fue entrenado sobre datos generados con una relación lineal más ruido, un buen ajuste es esperado.
# El gráfico muestra una clara tendencia descendente (mayor temperatura → menor consumo).


