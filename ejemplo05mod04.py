# 1. Instalación e importación de librerías (2 puntos) 
# • Instalar Seaborn y otras librerías necesarias. 
# • Importar correctamente Seaborn, Pandas y Matplotlib. 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# 2. Creación del conjunto de datos (2 puntos) 
# • Crear un conjunto de datos simple con al menos 4 variables numéricas (por ejemplo: Edad, Ingresos, Años de Educación, Horas de Sueño). 
data = {
    'Edad': [25, 32, 47, 51, 38, 29, 41, 33],
    'Ingresos': [3000, 4000, 5500, 6200, 4800, 3200, 5100, 3900],
    'Años_de_Educacion': [12, 16, 18, 20, 16, 14, 18, 15],
    'Horas_de_Sueño': [7, 6, 5, 6, 7, 8, 5, 7]}
df = pd.DataFrame(data)
# 3. Cálculo de la matriz de correlación (2 puntos) • Calcular la matriz de correlación entre las variables numéricas del conjunto de 
# datos usando Pandas. 
correlation_matrix = df.corr(numeric_only=True)
print(correlation_matrix)
# 4. Generación del heatmap (3 puntos) 
# • Utilizar Seaborn para generar el heatmap que visualiza la matriz de correlación. 
# • Personalizar el gráfico: usar el parámetro annot=True para mostrar los valores en cada celda, y seleccionar una paleta de colores 
# apropiada (cmap='coolwarm'). 
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, linecolor='black')
plt.title('Matriz de Correlación')
plt.show()
# 5. Interpretación y Personalización del gráfico (1 punto) 
# • Interpretar brevemente los resultados del heatmap. 
# • Personalizar el gráfico ajustando el tamaño de la figura y añadiendo líneas de borde entre las celdas. 
# Hay una alta correlación positiva entre Edad e Ingresos.
# Años de Educación tiene un correlación moderada con Ingresos.
# Horas de Sueño tiene una correlación negativa debil con Edad y Años de Educación.
