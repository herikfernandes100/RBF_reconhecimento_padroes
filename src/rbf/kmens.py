import math


class KMeansRBF:

    def __init__(self, n_centros):

        self.n_centros = n_centros

        self.centros = []
        self.grupos = []

    def distancia(self, a, b):

        soma = 0.0

        for i in range(len(a)):
            soma += (a[i] - b[i]) ** 2

        return math.sqrt(soma)

    def treinar(self, X):

        self.centros = X[:self.n_centros]

        mudou = True

        while mudou:

            grupos = [[] for _ in range(self.n_centros)]

            for amostra in X:

                menor = float("inf")
                indice = 0

                for j, centro in enumerate(self.centros):

                    d = self.distancia(amostra, centro)

                    if d < menor:
                        menor = d
                        indice = j

                grupos[indice].append(amostra)

            novos_centros = []

            for grupo in grupos:

                if len(grupo) == 0:
                    novos_centros.append([0.0] * len(X[0]))
                    continue

                centro = []

                for i in range(len(X[0])):

                    media = sum(amostra[i] for amostra in grupo) / len(grupo)

                    centro.append(media)

                novos_centros.append(centro)

            mudou = novos_centros != self.centros

            self.centros = novos_centros

        self.grupos = grupos

        return self.centros