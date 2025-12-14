# 1. Análisis Exploratorio de Datos (2 puntos) 
# • Carga el dataset en un DataFrame de Pandas. 
# • Muestra las primeras 5 filas y usa .info() para obtener información sobre los datos. 
# • Calcula estadísticas descriptivas con .describe(). 
# • Genera un histograma del número de entrenamientos semanales. 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
from sklearn.metrics import r2_score

# Cargar el dataset olimpicos.csv en un dataframe de Pandas 
datos=pd.read_csv("olimpicos.csv")
df = pd.DataFrame(datos)

# Mostrar las primeras 5 filas y usar .info() para obtener información sobre los datos
print("\n",df.head(),"\n")
print("\n",df.info(),"\n")

# Calculo de estadistica descriptiva con .describe()
print("\n",df.describe(),"\n")

# Generar un histograma del numero de entrenamientos semanales
plt.hist(df["Entrenamientos_Semanales"],bins = 5, edgecolor="green")
plt.xlabel("Cantidad de Entrenamientos Semanales")
plt.ylabel("Frecuencia")
plt.title("Histograma de la Frecuencia de Entrenamientos Semanales")
plt.show()

# 2. Estadística Descriptiva (2 puntos) 
# • Determina el tipo de variable de cada columna. 
# • Calcula la media, mediana y moda de la cantidad de medallas obtenidas. 
# • Calcula la desviación estándar de la altura de los atletas. 
# • Identifica valores atípicos en la columna de peso utilizando un boxplot. 

# Determinación del tipo de variable de cada columna
print(df.dtypes)
# Son variables cualitativas: Atleta, Deporte, Pais.
# Son variables cuantitativas: Edad, Altura_cm, Peso_kg, Entrenamientos_Semanales, Medallas_Totales.

# Calculo de la media, mediana y moda de la cantidad de medallas obtenidas
print("\nLa Media de Medallas_Totales es:",df["Medallas_Totales"].mean(),"\n")
print("\nLa Mediana de Medallas_Totales es:",df["Medallas_Totales"].median(),"\n")
print("\nLa Moda de Medallas_Totales es:",df["Medallas_Totales"].mode()[0],"\n")

# Calculo de la desviación estandar de la altura de los atletas
print("\nLa Desviación Estandar de la Altura de los Atletas es:",df["Altura_cm"].std(),"\n")

# Identificación de valores atipicos en el peso con grafico de boxplot
sns.boxplot(x=df["Peso_kg"])
plt.title("Boxplot del Peso de los Atletas [Kg]")
plt.xlabel("Peso [Kg]")
plt.show()

# 3. Análisis de Correlación (2 puntos) 
# • Calcula la correlación de Pearson entre entrenamientos semanales y medallas totales. 
# • Crea un gráfico de dispersión (scatterplot) entre peso y medallas totales con Seaborn.
# • Explica si existe correlación entre estas variables. 

# Calculo de la correlación de Pearson entre entrenamientos semanales y medallas totales
coef = df["Entrenamientos_Semanales"].corr(df["Medallas_Totales"])
print("\nCoeficiente de Correlación de Pearson:",coef,"\n")
if coef > 0.8:
    print("De acuerdo a la tabla del coeficiente de Pearson, y el valor obtenido del coeficiente de Pearson calculado de:",coef, "se puede decir que existe una correlación positiva fuerte entre las variables entrenamiento semanal y medallas totales")
elif 0.8 >= coef > 0.5:
    print("De acuerdo a la tabla del coeficiente de Pearson, y el valor obtenido del coeficiente de Pearson calculado de:",coef, "se puede decir que existe una correlación positiva moderada entre las variables entrenamiento semanal y medallas totales")
elif 0.5>= coef > 0.2:
    print("De acuerdo a la tabla del coeficiente de Pearson, y el valor obtenido del coeficiente de Pearson calculado de:",coef, "se puede decir que existe una correlación positiva debil entre las variables entrenamiento semanal y medallas totales")
elif 0.2 >= coef > -0.2:
    print("De acuerdo a la tabla del coeficiente de Pearson, y el valor obtenido del coeficiente de Pearson calculado de:",coef, "se puede decir que existe una correlación despreciable o nula entre las variables entrenamiento semanal y medallas totales")
elif -0.2 >= coef > -0.5:
    print("De acuerdo a la tabla del coeficiente de Pearson, y el valor obtenido del coeficiente de Pearson calculado de:",coef, "se puede decir que existe una correlación negativa débil entre las variables entrenamiento semanal y medallas totales")
elif -0.5 >= coef > 0.8:
    print("De acuerdo a la tabla del coeficiente de Pearson, y el valor obtenido del coeficiente de Pearson calculado de:",coef, "se puede decir que existe una correlación negativa moderada entre las variables entrenamiento semanal y medallas totales")
else:
    print("De acuerdo a la tabla del coeficiente de Pearson, y el valor obtenido del coeficiente de Pearson calculado de:",coef, "se puede decir que existe una correlación negativa fuerte entre las variables entrenamiento semanal y medallas totales")

coef2 = df["Peso_kg"].corr(df["Medallas_Totales"])
print("\nCoeficiente de Correlación de Pearson:",coef2,"\n")
if coef2 > 0.8:
    print("De acuerdo a la tabla del coeficiente de Pearson, y el valor obtenido del coeficiente de Pearson calculado de:",coef2, "se puede decir que existe una correlación positiva fuerte entre las variables peso y medallas totales")
elif 0.8 >= coef2 > 0.5:
    print("De acuerdo a la tabla del coeficiente de Pearson, y el valor obtenido del coeficiente de Pearson calculado de:",coef2, "se puede decir que existe una correlación positiva moderada entre las variables peso y medallas totales")
elif 0.5>= coef2 > 0.2:
    print("De acuerdo a la tabla del coeficiente de Pearson, y el valor obtenido del coeficiente de Pearson calculado de:",coef2, "se puede decir que existe una correlación positiva debil entre las variables peso y medallas totales")
elif 0.2 >= coef2 > -0.2:
    print("De acuerdo a la tabla del coeficiente de Pearson, y el valor obtenido del coeficiente de Pearson calculado de:",coef2, "se puede decir que existe una correlación despreciable o nula entre las variables peso y medallas totales")
elif -0.2 >= coef2 > -0.5:
    print("De acuerdo a la tabla del coeficiente de Pearson, y el valor obtenido del coeficiente de Pearson calculado de:",coef2, "se puede decir que existe una correlación negativa débil entre las variables peso y medallas totales")
elif -0.5 >= coef2 > 0.8:
    print("De acuerdo a la tabla del coeficiente de Pearson, y el valor obtenido del coeficiente de Pearson calculado de:",coef2, "se puede decir que existe una correlación negativa moderada entre las variables peso y medallas totales")
else:
    print("De acuerdo a la tabla del coeficiente de Pearson, y el valor obtenido del coeficiente de Pearson calculado de:",coef2, "se puede decir que existe una correlación negativa fuerte entre las variables peso y medallas totales")

# Elaboración grafico scatterplot entre el peso y las medallas totales con Seaborn
sns.scatterplot(data=df, x="Peso_kg",y="Medallas_Totales")
plt.xlabel("Peso [Kg]")
plt.ylabel("Medallas Totales")
plt.title("Gráfico de Dispersión del Peso [Kg] versus las Medallas Totales")
plt.grid(True)
plt.show()

# 4. Regresión Lineal (2 puntos) 
# • Implementa un modelo de regresión lineal para predecir el número de medallas obtenidas en función del número de entrenamientos semanales. 
# • Obtén los coeficientes de regresión e interpreta el resultado. 
# • Calcula el R² para medir el ajuste del modelo.
# • Usa Seaborn (regplot) para graficar la regresión lineal. 

# Implementación de modelo de regresión lineal de predicción de numero de medallas versus la cantidad de entrenamientos semanales
print(df)
X=df["Entrenamientos_Semanales"].values.reshape(-1,1)
Y=df["Medallas_Totales"]
modelo=LinearRegression()
modelo.fit(X,Y)

# Obtención de los coeficientes de regresión 
beta_0=modelo.intercept_
beta_1=modelo.coef_[0]
print(f"\nEl Coeficiente Beta Cero es:",beta_0,"y el coeficiente Beta Uno es:", beta_1," del Modelo de Regresión Lineal Obtenido a partir de los Datos de DataFrame olimpicos.csv")
Y_pred=modelo.predict(X)
print("\n",Y_pred)
print("\n",df["Medallas_Totales"])
# Interpretación del Resultado: El modelo de regresión lineal no es acertado, para valores hasta 10 sus estimaciones son bastante correctas
# pero en valores mayores, la predicción se desvia demasiado del valor real

# Determinación del R2 del Modelo de la Regresión Lineal
Y_real=pd.Series(df["Medallas_Totales"].values)
r2=r2_score(Y_real,Y_pred)
print(f"Coeficiente de Determinación R2:{r2:.2f}")

# Grafico Regplot de la Regresión Lineal
plt.figure(figsize=(6,4))
sns.regplot(x="Entrenamientos_Semanales",y="Medallas_Totales", data=df, line_kws={"color":"red"})
plt.title("Regresión Lineal: Entrenamiento Semanal versus Medallas Totales")
plt.xlabel("Entrenamientos Semanales")
plt.ylabel("Medallas Totales")
plt.show()

# 5. Visualización de Datos con Seaborn y Matplotlib (2 puntos) 
# • Crea un heatmap de correlación entre todas las variables numéricas. 
# • Crea un boxplot de la cantidad de medallas por disciplina deportiva. 
# • Personaliza los gráficos con títulos, etiquetas y colores.  

# Heatmap de correlación entre todas las variables numericas
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Mapa de Correlaciones entre todas las variables numéricas")
plt.show()

# Boxplot de la cantidad de medallas por disciplina deportiva
plt.figure(figsize=(8,5))
sns.boxplot(x="Deporte",y="Medallas_Totales",hue=(), data=df, legend=False, palette=("Spectral"))
plt.title("Distribución de Medallas por Deporte")
plt.xlabel("Deporte")
plt.ylabel("Medallas Totales")
plt.xticks(rotation=30)
plt.show()