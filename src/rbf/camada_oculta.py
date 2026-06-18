from rbf.gaussiana import FuncaoGaussiana


class CamadaOculta:

    def __init__(self, centros, sigmas):

        self.centros = centros
        self.sigmas = sigmas

    def transformar(self, X):

        H = []

        for amostra in X:

            linha = []

            for centro, sigma in zip(self.centros, self.sigmas):

                valor = FuncaoGaussiana.ativacao(
                    amostra,
                    centro,
                    sigma
                )

                linha.append(valor)

            H.append(linha)

        return H