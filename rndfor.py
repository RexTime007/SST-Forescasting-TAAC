import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Supongamos que df es tu DataFrame con las columnas 'Date' y 'Temperature'
df = pd.read_csv('LatLonFiltered.csv')
df['Fecha'] = pd.to_datetime(df['Fecha'])

# Configuración de la ventana deslizante
window_size = 12  # 36 meses
prediction_step = 6  # Predecir 6 meses adelante
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

resultados.to_csv("output_rndForest1.csv")
print(resultados)
# Evaluación del modelo
mse = mean_squared_error(actual, predictions)
print('Mean Squared Error:', mse)
