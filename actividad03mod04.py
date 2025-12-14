import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
# 1. Creación de Datos Simulados (2 puntos)
# • Generar dos listas de datos numéricos simulados que representen variables
# relacionadas (ejemplo: horas de ejercicio por semana y presión arterial). 
# Datos ficticios de tiempo de ejercicio semanal [min] versus presión arterial [mm Hg] 
np.random.seed(11)
minutos_ejercicios = np.random.randint(0, 300, 20)  # Número de minutos de ejercicio semanales [min]
presion_arterial = np.random.randint(40, 220, 20)  # Rango de presión arterial con variación aleatoria [mm Hg]
print("\nDatos de Tiempo de Ejercicio Semanal [min]\n",minutos_ejercicios)
print("\nDatos de Presión Arterial [mm Hg]\n",presion_arterial)
#Construcción de una Tabla de Contingencia (2 puntos)
# • Crear una tabla de contingencia con datos categóricos (ejemplo: grupo de edad y tipo de dieta).
df = pd.DataFrame({"Grupo_Edad":["Infante","Adolecente","Adulto_Mayor","Adulto","Adulto_Mayor","Adolecente"],
                   "Tipo_Dieta":["Chatarra","Chatarra","Casera","Ketogenica","Vegano","Chatarra"]})
print(f"\nEl DataFrame Original de Grupo de Edad vs Tipo de Dieta es:\n",df)
tabla_contingencia = pd.crosstab(df["Grupo_Edad"],df["Tipo_Dieta"])
print(f"\nLa Tabla de Contingencia del Grupo de Edad vs Tipo de Dieta es:\n",tabla_contingencia)
# 3. Visualización con Scatterplot (2 puntos)
# • Crear un gráfico de dispersión (scatterplot) para analizar la relación entre dos variables numéricas.
plt.figure(figsize=(8,5))
plt.scatter(minutos_ejercicios,presion_arterial,color='Red', alpha=0.5)
plt.xlabel("Tiempo Ejercicio Semanal [min]")
plt.ylabel("Presión Arterial [mm Hg]")
plt.title("Relación entre el Tiempo de Ejercicio Semanal [min] versus la Presión Arterial [mm Hg]")
plt.grid(True)
plt.show()
# 4. Cálculo del Coeficiente de Correlación de Pearson (2 puntos)
# • Calcular el coeficiente de correlación de Pearson e interpretar su resultado
coef,pvalue = stats.pearsonr(minutos_ejercicios,presion_arterial)
print(f"\nCoeficciente de Correlación de Pearson:{coef:.2f}")
print(f"\nP-Value:{pvalue:.4f}")
# Interpretación: De acuerdo al Coeficiente de correlación de Pearson de r = -0,13, esto indica que
# existe una correlación despreciable o nula entre las variables tiempo de ejercicio semanal [min] versus
# la presión arterial [mm Hg]. Además, si se agrega al análisis el p-value, si se considera que valores
# de p-value < 0.05 indican una correlación estadisticamente significativa; en el caso de análisis el
# valor obtenido a partir de los resultados de campo de p-value = 0.5955, lo que corrobora que no hay
# correlación estadisticamente significativas entre las variables tiempo de ejercicios semanal y la 
# presión arterial.

#5. Reflexión sobre Correlación vs. Causalidad (2 puntos)
# • Explicar con ejemplos si la correlación encontrada implica causalidad o no
# Reflexión: En el presente estudio no se consideraron otro tipo de relaciones que no fueran lineales
# tampoco se hizo de filtrado de outliers, lo que puede esconder un relación lineal, en caso de que
# existiera entre las variables tiempo de ejercicio semanal y presión arterial. Es razonable pensar que 
# a mayor tiempo de ejercicio, la presión arterial debería ser menor, salvo en el caso de medir la 
# presión arterial inmediatamente post ejercicio. Además, faltan variables importante en el estudio 
# como la edad especifica y/o el tipo de ejercicio ejecutado (ej: cardio o esfuerzo), lo que sugeriria
# sería agregar las variables faltantes señaladas como incrementar la cantidad de datos en todas las
# variables sugeridas dentro de las opciones de financiación de estudio en cuestión.