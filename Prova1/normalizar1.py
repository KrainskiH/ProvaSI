import numpy as np
import pickle


class Normalizador:

    def __init__(self):
        self._minmax_min = None
        self._minmax_max = None

        self._label_to_int = {}
        self._int_to_label = {}

        self._ohe_classes = None

    def minmax_fit_transform(self, dados: np.ndarray) -> np.ndarray:
        self._minmax_min = dados.min(axis=0)
        self._minmax_max = dados.max(axis=0)

        amplitude = self._minmax_max - self._minmax_min

        amplitude[amplitude == 0] = 1

        return (dados - self._minmax_min) / amplitude

    def minmax_transform(self, dados: np.ndarray) -> np.ndarray:
        if self._minmax_min is None:
            raise ValueError("Execute minmax_fit_transform antes de usar minmax_transform.")

        amplitude = self._minmax_max - self._minmax_min
        amplitude[amplitude == 0] = 1

        return (dados - self._minmax_min) / amplitude

    def minmax_inverse(self, dados_norm: np.ndarray) -> np.ndarray:
        if self._minmax_min is None:
            raise ValueError("Execute minmax_fit_transform antes de usar minmax_inverse.")

        return dados_norm * (self._minmax_max - self._minmax_min) + self._minmax_min

    def label_fit_transform(self, categorias: list) -> np.ndarray:
        classes_unicas = sorted(set(categorias))

        self._label_to_int = {label: idx for idx, label in enumerate(classes_unicas)}
        self._int_to_label = {idx: label for label, idx in self._label_to_int.items()}

        return np.array([self._label_to_int[c] for c in categorias])

    def label_transform(self, categorias: list) -> np.ndarray:
        if not self._label_to_int:
            raise ValueError("Execute label_fit_transform antes de usar label_transform.")

        return np.array([self._label_to_int[c] for c in categorias])

    def label_inverse(self, codificado: np.ndarray) -> list:
        if not self._int_to_label:
            raise ValueError("Execute label_fit_transform antes de usar label_inverse.")

        return [self._int_to_label[i] for i in codificado]

    def ohe_fit_transform(self, categorias: list) -> np.ndarray:
        self._ohe_classes = sorted(set(categorias)) 
        n_classes = len(self._ohe_classes)

        resultado = []
        for cat in categorias:
            vetor = [0] * n_classes
            idx = self._ohe_classes.index(cat)
            vetor[idx] = 1
            resultado.append(vetor)

        return np.array(resultado)

    def ohe_transform(self, categorias: list) -> np.ndarray:
        if self._ohe_classes is None:
            raise ValueError("Execute ohe_fit_transform antes de usar ohe_transform.")

        n_classes = len(self._ohe_classes)
        resultado = []

        for cat in categorias:
            vetor = [0] * n_classes
            idx = self._ohe_classes.index(cat)
            vetor[idx] = 1
            resultado.append(vetor)

        return np.array(resultado)

    def ohe_inverse(self, matriz_ohe: np.ndarray) -> list:
        if self._ohe_classes is None:
            raise ValueError("Execute ohe_fit_transform antes de usar ohe_inverse.")

        return [self._ohe_classes[np.argmax(linha)] for linha in matriz_ohe]

    def salvar(self, caminho: str):
        with open(caminho, 'wb') as f:
            pickle.dump(self, f)
        print(f"Modelo salvo em '{caminho}'.")

    @staticmethod
    def carregar(caminho: str) -> 'Normalizador':
        with open(caminho, 'rb') as f:
            modelo = pickle.load(f)
        print(f"Modelo carregado de '{caminho}'.")
        return modelo
