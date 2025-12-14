# 1. Importar las librerias necesarias
# Importar Matplotlib, NumPy y Pandas.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 2. Cargar los datos desde un archivo CSV
# Cargar el dataset "datos_trafico.csv" en un DataFrame de Pandas.
df = pd.read_csv("datos_trafico.csv")
# 3. Crear un grafico de lineas para visualizar el trafico diario
# Usar la función plt.plot() para crear un gráfico de líneas que muestre la cantidad de
# vehículos por día.
plt.figure(figsize=(10, 5))
plt.plot(df['Día'], df['Vehículos'], color='blue', linestyle='-', marker='o', label='Vehículos')
plt.title("Cantidad de vehículos por día")
plt.xlabel("Día del mes")
plt.ylabel("Número de vehículos")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# 4. Crear un Histograma para analizar la distribución de velocidades
# Usar plt.hist() para visualizar la distribución de velocidades promedio de los vehículos.
# Usar 8 bins para dividir los datos en intervalos.
plt.figure(figsize=(8, 5))
plt.hist(df['Velocidad_Promedio'], bins=8, color='orange', edgecolor='black')
plt.title("Distribución de velocidades promedio")
plt.xlabel("Velocidad promedio (km/h)")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.tight_layout()
plt.show()
# 5. Crear un grafico de dispersión para analizar la relación entre vehiculos y accidentes
# Añade un título, etiquetas para los ejes X e Y, y cambia el color de las barras.
plt.figure(figsize=(10, 6))
plt.scatter(df['Vehículos'], df['Accidentes'], color='red', edgecolor='black')
plt.title("Relación entre vehículos y accidentes")
plt.xlabel("Número de vehículos")
plt.ylabel("Número de accidentes")
plt.grid(True)
plt.tight_layout()
# 6. Personalización de gráficos
# Los gráficos deben incluir:
# Títulos claros que expliquen el contenido del gráfico.
# Etiquetas en los ejes X e Y.
# Leyendas cuando sea necesario.
# Colores adecuados y estilos personalizados.
# Tamaño ajustado para mejor visibilidad (figsize)
# 7. Guardar el grafico de dispersión como PNG
# Usar plt.savefig() para guardar el gráfico de dispersión como "grafico_accidentes.png".
plt.savefig("grafico_accidentes.png")
plt.show()