import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X=np.array([1,2,3,4,5]).reshape(-1,1)
Y=np.array([2,3,5,6,8])

modelo=LinearRegression()
modelo.fit(X,Y)

beta_0=modelo.intercept_
beta_1=modelo.coef_[0]
print(f"Intercepto con Y ={beta_0}")
print(f"Intercepto con Y ={beta_1}")
Y_pred=modelo.predict(X)
print(f"El valor predicho es de:{Y_pred}")
plt.scatter(X,Y, color="blue",label="Datos reales")
plt.plot(X,Y_pred,color="red",label="Regresión lineal")
plt.xlabel("Variable X")
plt.ylabel("Variable Y")
plt.title("Regresión Lineal Simple")
plt.legend()
plt.grid()
plt.show()

print(f"Intercepto (B0): {beta_0}")
print(f"Pendiente (B1): {beta_1}")
import statsmodels.api as sm

X=sm.add_constant(X)
model = sm.OLS(Y,X).fit()
print(model.summary())
