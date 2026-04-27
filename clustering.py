import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from normalizar1 import Normalizador
import pickle

# carregar dataset com import, pois nao reconhecia
import os
caminho = os.path.join(os.path.dirname(__file__), 'ObesityDataSet_raw_and_data_sinthetic.csv')
df = pd.read_csv(caminho)
colunas_numericas = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
colunas_categoricas = ['Gender', 'family_history_with_overweight', 'FAVC',
                       'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

# Treinamento
# -------------------------
def treinar(n_clusters=5):

    print("Treinando modelo...")

    # tratar categóricos
    encoders = {}
    dados_cat = []

    for col in colunas_categoricas:
        norm_col = Normalizador()
        valores = norm_col.label_fit_transform(df[col].tolist())
        encoders[col] = norm_col
        dados_cat.append(valores)

    dados_cat = np.array(dados_cat, dtype=float).T

    scaler_num = Normalizador()
    dados_num = scaler_num.minmax_fit_transform(df[colunas_numericas].values.astype(float))

    scaler_cat = Normalizador()
    dados_cat = scaler_cat.minmax_fit_transform(dados_cat)

    # juntar tudo
    X = np.hstack([dados_num, dados_cat])

    # modelo
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)

    print("Treinamento concluído")
    print("Inércia:", kmeans.inertia_)

    # salvar
    pickle.dump(kmeans, open("kmeans.pkl", "wb"))
    scaler_num.salvar("scaler_num.pkl")
    scaler_cat.salvar("scaler_cat.pkl")
    pickle.dump(encoders, open("encoders.pkl", "wb"))

    return kmeans, scaler_num, scaler_cat, encoders

# Clusters
# -------------------------
def descrever_clusters(kmeans, labels):

    print("\nDescrição dos clusters:")

    for i in range(kmeans.n_clusters):
        grupo = df[labels == i]

        print(f"\nCluster {i} - {len(grupo)} pessoas")

        print("Médias:")
        print(grupo[colunas_numericas].mean().round(2))

        # IMC simples
        imc = grupo["Weight"].mean() / (grupo["Height"].mean() ** 2)
        print("IMC médio:", round(imc, 2))

        print("Desvio padrão:")
        print(grupo[colunas_numericas].std().round(2))

        print("Distribuição NObeyesdad:")
        print(grupo['NObeyesdad'].value_counts().to_string())

# Inferencia
# -------------------------
def prever_paciente(paciente_num, paciente_cat):

    print("\nPrevendo paciente...")

    kmeans = pickle.load(open("kmeans.pkl", "rb"))
    scaler_num = Normalizador.carregar("scaler_num.pkl")
    scaler_cat = Normalizador.carregar("scaler_cat.pkl")
    encoders = pickle.load(open("encoders.pkl", "rb"))

    # numérico
    num = scaler_num.minmax_transform(np.array([paciente_num], dtype=float))

    # categórico
    cat = []
    for i, col in enumerate(colunas_categoricas):
        valor = encoders[col].label_transform([paciente_cat[i]])
        cat.append(float(valor[0]))

    cat = scaler_cat.minmax_transform(np.array([cat], dtype=float))

    X = np.hstack([num, cat])

    cluster = kmeans.predict(X)[0]

    print("Cluster previsto:", cluster)

    return cluster

# Main
# -------------------------
if __name__ == "__main__":

    kmeans, scaler_num, scaler_cat, encoders = treinar(5)

    labels = kmeans.labels_
    descrever_clusters(kmeans, labels)

    paciente_num = [30, 1.75, 95, 2, 3, 2, 1, 1]
    paciente_cat = ['Male', 'yes', 'yes', 'Sometimes', 'no', 'no', 'Sometimes', 'Public_Transportation']

    prever_paciente(paciente_num, paciente_cat)
