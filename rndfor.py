import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Supongamos que df es tu DataFrame con las columnas 'Date' y 'Temperature'
df = pd.read_csv('LatLonFiltered.csv')
df['Fecha'] = pd.to_datetime(df['Fecha'])

# Configuración de la ventana deslizante
window_size = 12 # 36 meses
prediction_step = 1  # Predecir 6 meses adelante
train_start = 0
train_end = train_start + window_size

# Predicciones
predictions = []
actual = []

# Bucle para entrenar y predecir usando la ventana deslizante
while train_end + prediction_step - 1 < len(df):
    # Datos de entrenamiento y prueba
    train_data = df.iloc[train_start:train_end]
    test_data = df.iloc[train_end + prediction_step - 1:train_end + prediction_step]  # Predecir el sexto mes adelante

    # Asegurarse de que estamos en el rango de fechas deseado (2020-2023)
    if test_data['Fecha'].dt.year.values[0] < 2020:
        train_start += 1
        train_end += 1
        continue
    if test_data['Fecha'].dt.year.values[0] > 2022:
        break

    # Entrenamiento del modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(train_data[['SST']], train_data['SST'])

    # Predicción
    pred = model.predict(test_data[['SST']])
    predictions.append(pred[0])
    actual.append(test_data['SST'].values[0])

    # Mover la ventana
    train_start += 1
    train_end += 1

resultados = pd.DataFrame({
    'Prediccion_SST': predictions
})


resultados.to_csv("output_rndForest1.csv", index=False)

df_adjust = df[(df["Fecha"] >= '2020-01-01') & (df['Fecha'] < '2023-01-01')]

df_adjust = df_adjust.reset_index(drop=True)

df_adjust = pd.DataFrame(df_adjust)

#df_adjust.to_csv("2020_2022.csv", index=False)

# Evaluación del modelo
mse = mean_squared_error(actual, predictions)
print('Mean Squared Error:', mse)


resultados = pd.read_csv("output_rndForest1.csv")
# Asumiendo que data['SST'] son tus observaciones reales y y_ts_pred son tus predicciones.
df_adjust = pd.read_csv("2020_2022.csv")

resultados.insert(0, 'Fecha', df_adjust['Fecha'])

print((resultados))
"""# Convertir año y mes a una fecha (primer día de cada mes)
sst_promedio_mensual['Fecha'] = pd.to_datetime(sst_promedio_mensual['Año'].astype(str) + '-' +
                                               sst_promedio_mensual['Mes'].astype(str) + '-01')
"""
df_adjust['Fecha'] = pd.to_datetime(df_adjust['Fecha'])
resultados['Fecha'] = pd.to_datetime(resultados['Fecha'])

# Gráfico
plt.figure(figsize=(12, 6))
plt.plot(df_adjust['Fecha'], df_adjust['SST'], marker='o', linestyle='-', label='Observaciones Reales')
plt.plot(df_adjust['Fecha'], resultados['Prediccion_SST'], marker='*', linestyle='-', label='Predicciones')
plt.title('Temperatura Superficial del Mar (SST) Real vs Random Forest a 1 mes')
plt.xlabel('Fecha')
plt.ylabel('SST')
plt.legend()

# Formatear el eje x para mostrar cada año
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y,%m'))
plt.gcf().autofmt_xdate()  # Auto-rotar las fechas para que sean legibles
plt.grid(True)
plt.xticks(rotation=90)
plt.show()