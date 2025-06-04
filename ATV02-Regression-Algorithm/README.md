# Atividade 02 – Algoritmo de Regressão com Séries Temporais

## Visão Geral
Este projeto, denominado **Atividade 02**, tem como objetivo desenvolver um algoritmo de regressão de séries temporais usando dados reais. O modelo prevê o preço médio dos jogos na Steam ao longo do tempo, incorporando variáveis adicionais como avaliações e número de proprietários.

## Dataset Utilizado
- Arquivo: `steam_store_games_clean.csv`
- Origem: Dados de jogos da Steam, contendo informações como data de lançamento, preço, avaliações positivas/negativas e faixa de proprietários.
- Descrição resumida:  
  - `release_date`: Data de lançamento do jogo.  
  - `price`: Preço médio do jogo em dólares.  
  - `positive_ratings`: Número de avaliações positivas.  
  - `negative_ratings`: Número de avaliações negativas.  
  - `owners`: Faixa estimada de proprietários (por ex.: "10000-20000").

## Estrutura de Arquivos
- **ATV02-Douglas.py**: Script principal que realiza todo o fluxo do projeto.
- **steam_store_games_clean.csv**: Arquivo CSV com os dados limpos utilizados para modelagem.
- **README_ATV02.md**: Este arquivo de descrição do projeto.

## Dependências
Para executar o script, é necessário ter instalado:
- Python 3.7 ou superior
- Bibliotecas Python:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `prophet` (Facebook Prophet ou PyPI `prophet`)
  - `scikit-learn`

Instalação:
```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
pip install pandas numpy matplotlib prophet scikit-learn
```

## Descrição do Script (ATV02-Douglas.py)
1. **Carregamento dos Dados**  
   - O script lê o arquivo `steam_store_games_clean.csv` em um DataFrame do pandas.  
   - Converte a coluna `release_date` para o tipo datetime.

2. **Preparação dos Dados**  
   - Extrai `year` e `month` da data de lançamento.
   - Agrega métricas por dia:
     - `price`: média dos preços dos jogos lançados em cada data.  
     - `positive_ratings`: média das avaliações positivas.  
     - `negative_ratings`: média das avaliações negativas.  
     - `owners`: extrai o valor mínimo do intervalo de proprietários (por exemplo, de "10000-20000" pega 10000) e calcula a média.  
   - Renomeia colunas para adequar ao Prophet:  
     - `release_date` → `ds`  
     - `price` → `y`  
   - Ordena os dados por data (`ds`).
   - Adiciona regressores extras:  
     - `rating_ratio`: razão entre avaliações positivas e negativas: `(positive_ratings + 1)/(negative_ratings + 1)`.  
     - `log_owners`: logaritmo natural de `(owners + 1)`.

3. **Visualização dos Dados Originais**  
   - Gera dois gráficos lado a lado:  
     - Gráfico de linhas com a evolução do preço médio dos jogos ao longo do tempo.  
     - Scatter plot relacionando `rating_ratio` ao `price`.

4. **Divisão entre Conjunto de Treino e Teste**  
   - Separa os últimos 60 dias de dados para teste e o restante para treino.

5. **Configuração e Treinamento do Modelo Prophet**  
   - Inicializa um modelo Prophet com parâmetros otimizados:  
     - `changepoint_prior_scale=0.05`  
     - `seasonality_prior_scale=10`  
     - `seasonality_mode='multiplicative'`  
     - `yearly_seasonality=True`  
     - `weekly_seasonality=True`  
     - `daily_seasonality=False`  
     - `interval_width=0.95`  
   - Adiciona os regressores `rating_ratio` e `log_owners`.  
   - Treina o modelo no conjunto de treino.

6. **Previsões**  
   - Gera previsões para os 60 dias de teste.  
   - Obtém valores previstos (`yhat`), limites inferior (`yhat_lower`) e superior (`yhat_upper`).

7. **Avaliação do Modelo**  
   - Calcula métricas de desempenho no conjunto de teste:  
     - **MSE (Mean Squared Error)**  
     - **RMSE (Root Mean Squared Error)**  
     - **MAE (Mean Absolute Error)**  
     - **R² (Coeficiente de Determinação)**  
   - Exibe essas métricas no console.

8. **Visualização dos Resultados**  
   - Plota três séries no mesmo gráfico para o conjunto de teste:  
     - Valores reais (`y`)  
     - Valores previstos (`yhat`)  
     - Intervalo de confiança (`yhat_lower` e `yhat_upper`)  
   - Exibe gráficos dos componentes: tendência, sazonalidade anual e semanal derivados do Prophet.

9. **Análise de Importância das Features**  
   - Calcula o impacto de cada componente (tendência, sazonalidade anual e sazonalidade semanal) usando o desvio padrão de cada componente previsto.  
   - Plota um gráfico de barras mostrando a importância relativa de cada componente.

## Como Executar
1. Navegue até a pasta do projeto no terminal.  
2. Ative o ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```  
3. Instale as dependências:
   ```bash
   pip install pandas numpy matplotlib prophet scikit-learn
   ```  
4. Execute o script:
   ```bash
   python ATV02-Douglas.py
   ```  
5. O script exibirá no console as etapas de carregamento, preparação, treinamento e avaliação, além de gerar gráficos interativos.

## Resultados Esperados
- Gráficos com a série temporal de preços, scatter plot de `rating_ratio` vs. preço e plot de componentes trimestrais/anual.  
- Métricas quantitativas de desempenho do modelo (MSE, RMSE, MAE, R²).  
- Gráfico de barras mostrando a importância dos componentes de tendência e sazonalidade.

## Considerações Finais
Este projeto ilustra a construção de um modelo de regressão de séries temporais usando Prophet, enriquecendo os dados com regressores derivados de métricas de avaliação e número de proprietários. Possíveis aprimoramentos incluem:
- Ajuste adicional de hiperparâmetros com Grid Search ou Bayesian Optimization.  
- Inclusão de variáveis exógenas adicionais (por exemplo, gênero dos jogos, categorias).  
- Análise de outliers e tratamento de datas sem lançamentos.  

---
