import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# 1. Carregar os dados
print("Carregando os dados...")
df = pd.read_csv("ATV02-Resposta/steam_store_games_clean.csv")
print("Dados carregados com sucesso!\n")

# 2. Preparar os dados
print("Preparando os dados...")
# Converter release_date para datetime
df['release_date'] = pd.to_datetime(df['release_date'])

# Criar features adicionais
df['year'] = df['release_date'].dt.year
df['month'] = df['release_date'].dt.month

# Calcular métricas agregadas por dia
daily_data = df.groupby('release_date').agg({
    'price': 'mean',
    'positive_ratings': 'mean',
    'negative_ratings': 'mean',
    'owners': lambda x: x.map(lambda s: float(s.split('-')[0])).mean()  # Pegar o valor mínimo do range
}).reset_index()

# Preparar dados para o Prophet
prophet_df = daily_data.rename(columns={'release_date': 'ds', 'price': 'y'})
prophet_df = prophet_df.sort_values('ds')

# Adicionar features extras como regressores
prophet_df['rating_ratio'] = (prophet_df['positive_ratings'] + 1) / (prophet_df['negative_ratings'] + 1)
prophet_df['log_owners'] = np.log1p(prophet_df['owners'])

print("Dados preparados!\n")

# 3. Plot dos dados originais
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(prophet_df['ds'], prophet_df['y'])
plt.title("Preço Médio dos Jogos ao Longo do Tempo")
plt.xlabel("Data de Lançamento")
plt.ylabel("Preço Médio ($)")

plt.subplot(1, 2, 2)
plt.scatter(prophet_df['rating_ratio'], prophet_df['y'], alpha=0.5)
plt.title("Relação entre Avaliações e Preço")
plt.xlabel("Razão de Avaliações Positivas/Negativas")
plt.ylabel("Preço Médio ($)")
plt.tight_layout()
plt.show()

# 4. Separar treino e teste (últimos 60 dias para melhor avaliação)
train = prophet_df.iloc[:-60]
test = prophet_df.iloc[-60:]

# 5. Criar e treinar modelo Prophet com configurações otimizadas
print("Treinando modelo Prophet...\n")
model = Prophet(
    changepoint_prior_scale=0.05,  # Controla flexibilidade da tendência
    seasonality_prior_scale=10,    # Aumenta importância da sazonalidade
    seasonality_mode='multiplicative',  # Melhor para dados com variação sazonal
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    interval_width=0.95
)

# Adicionar regressores
model.add_regressor('rating_ratio')
model.add_regressor('log_owners')

# Treinar modelo
model.fit(train)

# 6. Fazer previsões
future = test.copy()
forecast = model.predict(future)

# 7. Avaliação do modelo
y_true = test['y'].values
y_pred = forecast['yhat'].values

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("\nMétricas de Avaliação do Modelo:")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}\n")

# 8. Plotar resultados
plt.figure(figsize=(15, 10))

# Plot principal
plt.subplot(2, 1, 1)
plt.plot(train['ds'], train['y'], label='Treino', alpha=0.6)
plt.plot(test['ds'], test['y'], label='Real', alpha=0.6)
plt.plot(test['ds'], forecast['yhat'], label='Previsto', linestyle='--')
plt.fill_between(test['ds'], 
                 forecast['yhat_lower'], 
                 forecast['yhat_upper'], 
                 color='gray', 
                 alpha=0.2, 
                 label='Intervalo de Confiança')
plt.title("Previsão de Preços dos Jogos (Prophet)")
plt.xlabel("Data")
plt.ylabel("Preço ($)")
plt.legend()

# Plot dos componentes
plt.subplot(2, 1, 2)
model.plot_components(forecast)
plt.tight_layout()
plt.show()

# 9. Análise de Importância das Features
components = pd.DataFrame({
    'Feature': ['Tendência', 'Sazonalidade Anual', 'Sazonalidade Semanal'],
    'Impacto': [
        np.std(forecast['trend']),
        np.std(forecast['yearly']),
        np.std(forecast['weekly'])
    ]
})
components = components.sort_values('Impacto', ascending=False)

plt.figure(figsize=(10, 5))
plt.bar(components['Feature'], components['Impacto'])
plt.title("Importância das Features na Previsão")
plt.xticks(rotation=45)
plt.ylabel("Desvio Padrão do Impacto")
plt.tight_layout()
plt.show()
