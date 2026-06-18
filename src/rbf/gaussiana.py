import math


class FuncaoGaussiana:

    @staticmethod
    def calcular_sigma(grupo, centro):

        if len(grupo) == 0:
            return 1.0

        soma = 0.0

        for amostra in grupo:

            for i in range(len(amostra)):

                soma += (amostra[i] - centro[i]) ** 2

        sigma2 = soma / len(grupo)

        return math.sqrt(sigma2)

    @staticmethod
    def ativacao(x, centro, sigma):

        distancia2 = 0.0

        for i in range(len(x)):
            distancia2 += (x[i] - centro[i]) ** 2

        expoente = -distancia2 / (2 * sigma ** 2)

        return math.exp(expoente)