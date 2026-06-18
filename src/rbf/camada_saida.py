import random


class CamadaSaida:

    def __init__(self, entradas):

        self.pesos = [
            random.uniform(-0.5, 0.5)
            for _ in range(entradas)
        ]

        self.bias = random.uniform(-0.5, 0.5)

    def forward(self, x):

        soma = self.bias

        for i in range(len(x)):
            soma += x[i] * self.pesos[i]

        return soma