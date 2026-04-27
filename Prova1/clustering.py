import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pickle

#carregar dataset com import, pois nao reconhecia
import os
caminho = os.path.join(os.path.dirname(__file__), 'ObesityDataSet_raw_and_data_sinthetic.csv')
df = pd.read_csv(caminho)
colunas_numericas = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
colunas_categoricas = ['Gender', 'family_history_with_overweight', 'FAVC',
                       'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

#Treinamento
# -------------------------
def treinar(n_clusters=5):

    print("Treinando modelo...")

    #tratar categóricos
    encoders = {}
    dados_cat = []

    for col in colunas_categoricas:
        le = LabelEncoder()
        valores = le.fit_transform(df[col])
        encoders[col] = le
        dados_cat.append(valores)

    dados_cat = np.array(dados_cat).T

    #normalizar
    scaler_num = MinMaxScaler()
    dados_num = scaler_num.fit_transform(df[colunas_numericas])

    scaler_cat = MinMaxScaler()
    dados_cat = scaler_cat.fit_transform(dados_cat)

    #juntar tudo
    X = np.hstack([dados_num, dados_cat])

    #modelo
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)

    print("Treinamento concluído")
    print("Inércia:", kmeans.inertia_)

    #salvar
    pickle.dump(kmeans, open("kmeans.pkl", "wb"))
    pickle.dump(scaler_num, open("scaler_num.pkl", "wb"))
    pickle.dump(scaler_cat, open("scaler_cat.pkl", "wb"))
    pickle.dump(encoders, open("encoders.pkl", "wb"))

    return kmeans, scaler_num, scaler_cat, encoders

#Clusters
# -------------------------
def descrever_clusters(kmeans, labels):

    print("\nDescrição dos clusters:")

    for i in range(kmeans.n_clusters):
        grupo = df[labels == i]

        print(f"\nCluster {i} - {len(grupo)} pessoas")

        print("Médias:")
        print(grupo[colunas_numericas].mean())

        #IMC simples
        imc = grupo["Weight"].mean() / (grupo["Height"].mean() ** 2)
        print("IMC médio:", round(imc, 2))

#Inferencia
# -------------------------
def prever_paciente(paciente_num, paciente_cat):

    print("\nPrevendo paciente...")

    kmeans = pickle.load(open("kmeans.pkl", "rb"))
    scaler_num = pickle.load(open("scaler_num.pkl", "rb"))
    scaler_cat = pickle.load(open("scaler_cat.pkl", "rb"))
    encoders = pickle.load(open("encoders.pkl", "rb"))

    # numérico
    num = scaler_num.transform([paciente_num])

    # categórico
    cat = []
    for i, col in enumerate(colunas_categoricas):
        valor = encoders[col].transform([paciente_cat[i]])
        cat.append(valor[0])

    cat = scaler_cat.transform([cat])

    X = np.hstack([num, cat])

    cluster = kmeans.predict(X)[0]

    print("Cluster previsto:", cluster)

    return cluster

#Main
# -------------------------
if __name__ == "__main__":

    kmeans, scaler_num, scaler_cat, encoders = treinar(5)

    labels = kmeans.labels_
    descrever_clusters(kmeans, labels)

    paciente_num = [30, 1.75, 95, 2, 3, 2, 1, 1]
    paciente_cat = ['Male', 'yes', 'yes', 'Sometimes', 'no', 'no', 'Sometimes', 'Public_Transportation']

    prever_paciente(paciente_num, paciente_cat)