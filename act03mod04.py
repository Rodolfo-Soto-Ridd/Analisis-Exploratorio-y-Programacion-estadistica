import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
 
# Datos ficticios de tiempo de ejercicio semanal versus presión arterial 
np.random.seed(11)
minutos_ejercicios = np.random.randint(1, 300, 100)  # Número de minutos de ejercicio semanales [min]
presion_arterial = np.random.randint(40, 220, 50)  # Rango de presión arterial con variación aleatoria
df = pd.DataFrame({"Minutos_Ejercicio":[minutos_ejercicios],"Presion_Arterial":[presion_arterial]})
print(df)


print(cantidad_vendida)
print(precio_promedio)

# Crear el scatterplot
plt.figure(figsize=(8, 5))
plt.scatter(cantidad_vendida, precio_promedio, color='blue', alpha=0.5)
plt.xlabel('Cantidad Vendida')
plt.ylabel('Precio Promedio')
plt.title('Relación entre Cantidad Vendida y Precio Promedio')
plt.grid(True)
plt.show()

presupuesto_marketing = np.array([1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500])
ventas = np.array([200, 270, 340, 410, 480, 550, 600, 660])

coef, p_valor = stats.pearsonr(presupuesto_marketing, ventas)
print("Coef. de Pearson:",coef)
print("P-Valor", p_valor)

corr_matrix = np.corrcoef(presupuesto_marketing, ventas)
print(f"Coeficiente de Pearson usando Numpy: {corr_matrix[0,1]:.2f}")

plt.scatter(presupuesto_marketing,ventas, color ='Blue',alpha=0.5)
plt.xlabel('Presupuesto de Marketing')
plt.ylabel('Ventas')
plt.title('Relación entre Presupuesto y Ventas')
plt.grid(True)
plt.show()