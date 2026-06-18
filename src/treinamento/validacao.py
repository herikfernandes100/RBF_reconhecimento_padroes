class Validador:

    @staticmethod
    def mse(real, previsto):

        soma = 0.0

        for r, p in zip(real, previsto):

            soma += (r - p) ** 2

        return soma / len(real)