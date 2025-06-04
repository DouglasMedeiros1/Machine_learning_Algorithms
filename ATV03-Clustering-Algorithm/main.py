import matplotlib.pyplot as plt                      # Bibliotecas para plotagem
import pandas as pd                                   # Biblioteca para manipulação de DataFrames
import numpy as np                                    # Biblioteca para operações numéricas
from math import sqrt                                 # Função para cálculos geométricos
from sklearn.cluster import KMeans                    # Algoritmo K-Means
from sklearn.preprocessing import StandardScaler       # Para normalização de variáveis contínuas
from sklearn.decomposition import PCA                  # PCA para redução de dimensionalidade
import matplotlib.cm as cm                             # Mapeamento de cores
from mpl_toolkits.mplot3d import Axes3D                # Suporte para gráficos 3D 

def optimal_number_of_clusters(wcss):
    """
    Função que aplica o método do cotovelo automatizado para determinar
    o número ótimo de clusters (k) a partir dos valores de WCSS.
    - wcss (list): lista de valores de within-cluster sum of squares para 
      cada k testado (começando em k=2).
    Retorna:
    - k ótimo (int)
    """
    x1, y1 = 2, wcss[0]                # Primeiro ponto: k=2
    x2, y2 = len(wcss) + 1, wcss[-1]   # Último ponto: k = valor máximo testado

    distances = []
    for i, y0 in enumerate(wcss):
        x0 = i + 2                    # k correspondente
        # Distância ponto (x0, y0) à reta que liga (x1, y1) e (x2, y2)
        numerator = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
        denominator = sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator / denominator)
    
    # O k ótimo é o índice do maior valor de distância somado de 2 (pois a lista começa em k=2)
    return distances.index(max(distances)) + 2

# -------------------------------
# Leitura dos dados
# -------------------------------
# O arquivo 'steam.csv' contém o dataset para análise
df = pd.read_csv('ATV03/steam.csv')  # Usando o caminho correto para o arquivo

# -------------------------------
# Pré-processamento: remover duplicatas e tratar valores ausentes
# -------------------------------
df = df.drop_duplicates(subset='appid')  # Remove linhas duplicadas pelo mesmo appid
df = df.fillna({
    'genres': '',
    'steamspy_tags': '',
    'platforms': ''
})

# -------------------------------
# One-Hot Encoding para gêneros
# -------------------------------
# Transforma a coluna 'genres' (p.e., "Action;Adventure") em várias colunas binárias.
# OBS: como o CSV separa por ';', usamos sep=';' em get_dummies para criar uma coluna para cada gênero individual.
df_genres = df['genres'].str.get_dummies(sep=';')

# -------------------------------
# One-Hot Encoding para tags
# -------------------------------
# Identifica as 20 tags mais frequentes
all_tags = df['steamspy_tags'].str.split(',').explode()         # Lista todas as tags
top_tags = all_tags.value_counts().head(20).index.tolist()       # Seleciona as 20 tags mais comuns 

# Cria colunas binárias indicando presença de cada tag selecionada
for tag in top_tags:
    df[f'tag_{tag}'] = df['steamspy_tags'].str.contains(tag, regex=False).astype(int)

# -------------------------------
# Conversão de plataforma (Linux)
# -------------------------------
df['linux_binary'] = df['platforms'].str.contains('linux', case=False).astype(int)  
# 1 para True (suporte), 0 para False (sem suporte)

# -------------------------------
# Combinação final de features
# -------------------------------
# Concatena colunas de gêneros, tags e a coluna de compatibilidade Linux
features = pd.concat(
    [df_genres, df[[f'tag_{tag}' for tag in top_tags]], df['linux_binary']],
    axis=1
)

# -------------------------------
# Normalização opcional
# -------------------------------
# Como as colunas aqui são binárias (0/1), não é necessária padronização.
# Caso adicionem variáveis contínuas (ex.: owners_estimate), usar StandardScaler:
# scaler = StandardScaler()
# continuous_cols = ['owners_estimate', 'average_playtime_forever']
# features_continuous = scaler.fit_transform(df[continuous_cols])
# features = np.concatenate([features.values, features_continuous], axis=1)

X = features.values  # Matriz de features para clustering 

# -------------------------------
# Cálculo do WCSS (inércia) para valores de k variando de 2 a 15
# -------------------------------
print("Calculando WCSS para diferentes valores de k...")
wcss = []
k_range = range(2, 16)  # Testar k de 2 até 15
for k in k_range:
    print(f"Testando k={k}...")
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42, max_iter=100)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)  # Soma das distâncias quadráticas até o centróide

# -------------------------------
# Determinar o número ótimo de clusters via método do cotovelo automático
# -------------------------------
optimal_k = optimal_number_of_clusters(wcss)
print(f"Melhor número de clusters (Método do Cotovelo Automático): {optimal_k}")

# Plot da curva de WCSS vs. k
plt.figure(figsize=(8, 5))
plt.plot(k_range, wcss, marker='o')
plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'k ótimo = {optimal_k}')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('WSS (Within-Cluster Sum of Squares)')
plt.title('Método do Cotovelo Automático')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# Aplicar K-Means com o número ótimo de clusters
# -------------------------------
model = KMeans(
    n_clusters=optimal_k,
    init='k-means++',
    n_init=10,
    random_state=42,
    tol=0.0001,
    max_iter=300
)
model.fit(X)
clusters = model.labels_  # Vetor de rótulos de cluster para cada jogo        
df['cluster'] = clusters  # Atribui rótulo de cluster a cada linha do DataFrame original 

# -------------------------------
# 2. Inspecionar os centróides
# -------------------------------
centroids = model.cluster_centers_
centroids_df = pd.DataFrame(centroids, columns=features.columns)
centroids_df['cluster'] = range(optimal_k)

print("\nPerfis Médios dos Centróides (média das features binárias por cluster):")
print(centroids_df)

# -------------------------------
# 3. Analisar a proporção de Linux em cada cluster
# -------------------------------
proporcao_linux_por_cluster = df.groupby('cluster')['linux_binary'].mean()
print("\nProporção de Suporte nativo a Linux (linux_binary=1) por Cluster:")
print(proporcao_linux_por_cluster)

# -------------------------------
# 4. Visualizar em PCA 2D para verificação
# -------------------------------
pca_2d = PCA(n_components=2)
pca_array_2d = pca_2d.fit_transform(X)            # Projeta as features em 2 componentes principais
centroids_pca_2d = pca_2d.transform(centroids)     # Projeta também os centróides

df_pca_2d = pd.DataFrame(data=pca_array_2d, columns=['PC1', 'PC2'])
df_pca_2d['cluster'] = clusters

# Atribui cores a cada cluster
colors = [cm.tab20(i / optimal_k) for i in range(optimal_k)]
df_pca_2d['color'] = df_pca_2d['cluster'].map({i: colors[i] for i in range(optimal_k)})
df_pca_2d['color'] = df_pca_2d['color'].fillna('#000000')

plt.figure(figsize=(8, 6))
plt.scatter(df_pca_2d['PC1'], df_pca_2d['PC2'], c=df_pca_2d['color'], alpha=0.6, s=10)
plt.scatter(centroids_pca_2d[:, 0], centroids_pca_2d[:, 1], marker='*', s=200, c='black', label='Centróides')
plt.title("Clusters de Jogos (PCA 2D) com Centróides")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# 5. Visualizar em PCA 3D para verificação
# -------------------------------
pca_3d = PCA(n_components=3)
pca_array_3d = pca_3d.fit_transform(X)             # Projeta as features em 3 componentes principais
centroids_pca_3d = pca_3d.transform(centroids)      # Projeta também os centróides

df_pca_3d = pd.DataFrame(data=pca_array_3d, columns=['PC1', 'PC2', 'PC3'])
df_pca_3d['cluster'] = clusters
df_pca_3d['color'] = df_pca_3d['cluster'].map({i: colors[i] for i in range(optimal_k)})
df_pca_3d['color'] = df_pca_3d['color'].fillna('#000000')

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_pca_3d['PC1'], df_pca_3d['PC2'], df_pca_3d['PC3'], c=df_pca_3d['color'], s=20, alpha=0.7)
ax.scatter(centroids_pca_3d[:, 0], centroids_pca_3d[:, 1], centroids_pca_3d[:, 2],
           c='black', s=200, marker='*', label='Centróides')
ax.set_title("Clusters de Jogos (PCA 3D) com Centróides")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.legend()
plt.show()

# -------------------------------
# 6. Cálculo da proporção de compatibilidade Linux por gênero
# -------------------------------
df_aux = pd.concat([df_genres, df[['cluster', 'linux_binary']]], axis=1)

total_por_genero = df_genres.sum()  
linux_por_genero = df_aux[df_aux['linux_binary'] == 1][df_genres.columns].sum()  

prop_linux_genero = (linux_por_genero / total_por_genero).sort_values()
prop_linux_df = prop_linux_genero.reset_index()
prop_linux_df.columns = ['Genero', 'Proporcao_Linux']

print("\nProporção de compatibilidade nativa Linux por Gênero (todas as combinações):")
print(prop_linux_df)

# -------------------------------
# 7. Top 10 gêneros com menor e maior suporte Linux
# -------------------------------
top10_menor = prop_linux_df.head(10)
top10_maior = prop_linux_df.tail(10)

print("\nTop 10 gêneros com MENOR suporte a Linux:")
print(top10_menor)

print("\nTop 10 gêneros com MAIOR suporte a Linux:")
print(top10_maior)

# -------------------------------
# 8. Exibir “pódio” dos 3 piores e 3 melhores gêneros em termos de suporte a Linux
# -------------------------------
print("\n--- Pódio de Incompatibilidade (3 gêneros com menor Proporção_Linux) ---")
for rank, row in enumerate(top10_menor.head(3).itertuples(index=False), start=1):
    print(f"{rank}º: {row.Genero} — Proporção_Linux = {row.Proporcao_Linux:.2f}")

print("\n--- Pódio de Compatibilidade (3 gêneros com maior Proporção_Linux) ---")
# note que top10_maior foi retirado do fim, mas queremos os 3 maiores valores em ordem decrescente
best_three = top10_maior.tail(3).sort_values(by='Proporcao_Linux', ascending=False)
for rank, row in enumerate(best_three.itertuples(index=False), start=1):
    print(f"{rank}º: {row.Genero} — Proporção_Linux = {row.Proporcao_Linux:.2f}")
