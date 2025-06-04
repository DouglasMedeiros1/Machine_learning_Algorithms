# Objetivo: treinar um modelo de classificação para prever se um comentário é:
#           - crítica negativa (não acrescenta ou desenvolve um debate)
#           - crítica construtiva (acrescenta ao debate criticando uma decisão ou situação)
#           - proposta (acrescenta ao debate com uma possível solução)
# Escolha do algoritmo: Random Forest, combinada com a vetorização de texto via TfidfVectorizer,
#                       pois permite lidar com dados textuais de forma simples e eficiente.
#
# Link: https://www.kaggle.com/datasets/aravindhmp/the-people-opinions-about-the-war-in-2023
# 
# Para rodar:
# py -m venv venv 
# pip install pandas
# pip install scikit-learn
# python .\ATV01-Douglas.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

print("Início da execução do script...")

# 1. Carregar os dados
try:
    print("Carregando os dados...")
    df = pd.read_csv("War_Data_Sentiment.csv")
    print("Dados carregados com sucesso!\n")
except Exception as e:
    print("Ocorreu um erro:", e)

# 2. Visualizar as primeiras linhas
print("Visualizando as primeiras linhas do dataset:")
print(df.head(), "\n")

# 3. Remover colunas desnecessárias
print("Verificando e removendo colunas desnecessárias...")
if "id" in df.columns:
    df.drop(columns=["id"], inplace=True)
# Mantendo a coluna "Category" para uso no mapeamento
print("Verificação de colunas concluída!\n")

# 4. Remover valores nulos
print("Verificando valores nulos antes da remoção:")
print(df.isnull().sum(), "\n")

df.dropna(inplace=True)
print("Valores nulos removidos!\n")
print("Verificando valores nulos após a remoção:")
print(df.isnull().sum(), "\n")


# 5. Mapeamento simples baseado em palavras-chave, sentimento e emoção
print("Realizando mapeamento de categorias personalizadas...")

def map_category(row):
    text = row['comments'].lower()
    compound = row['Compound']
    emotion = row['Category']  # Considerando a emoção do post
    
    # Palavras-chave simples para ajudar na classificação
    proposal_keywords = ['should', 'need to', 'must', 'recommend', 'suggest', 'could', 'we can']
    constructive_keywords = ['i think', 'i believe', 'it would be better', 'instead', 'why not', 'concern']
    
    if any(word in text for word in proposal_keywords):
        return 'proposta'
    elif any(word in text for word in constructive_keywords):
        return 'crítica construtiva'
    elif compound < -0.3 or emotion == 'negativo':  # Incluindo emoção negativa
        return 'crítica negativa'
    elif emotion == 'positivo':  # Incluindo emoção positiva
        return 'crítica construtiva'
    else:
        return 'crítica construtiva'

df['MappedCategory'] = df.apply(map_category, axis=1)
print("Mapeamento concluído! Exemplo:\n", df[['comments', 'Category', 'MappedCategory']].head(), "\n")

# 6. Separar variáveis independentes e dependentes
print("Separando variáveis independentes (X) e dependentes (y)...")
X = df["comments"]
y = df["MappedCategory"]
print("Separação concluída!\n")

# 7. Vetorização TF-IDF
print("Convertendo os comentários em representações numéricas (TF-IDF)...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vectorized = vectorizer.fit_transform(X)
print("Conversão concluída!\n")

# 8. Dividir dados
print("Dividindo os dados em treino e teste...")
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42, stratify=y)
print(f"Total de amostras de treino: {X_train.shape[0]}")
print(f"Total de amostras de teste: {X_test.shape[0]}")
print("Divisão concluída!\n")

# 9. Treinar modelo
print("Treinando o modelo Random Forest...")
model = RandomForestClassifier(
    n_estimators=300, # Número de árvores na floresta - padrão 100
    min_samples_split=10, # Número mínimo de amostras necessárias - padrão 2
    min_samples_leaf=5, # Número mínimo de amostras que um nó folha  - padrão 1
    random_state=42 # Semente de aleatoriedade
)
model.fit(X_train, y_train)
print("Modelo treinado com sucesso!\n")


# 10. Previsões e avaliaçãoa
print("Realizando previsões no conjunto de teste...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nMétricas de Avaliação do Modelo:")
print(f"Acurácia: {accuracy:.2f}")
print(f"Precisão: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}\n")

print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))
