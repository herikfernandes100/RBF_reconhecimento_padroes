from rbf.kmeans import KMeansRBF
from rbf.gaussiana import FuncaoGaussiana
from rbf.camada_oculta import CamadaOculta
from rbf.camada_saida import CamadaSaida


class RedeRBF:

    def __init__(self, n_centros):

        self.n_centros = n_centros

    def treinar_primeiro_estagio(self, X):

        kmeans = KMeansRBF(self.n_centros)

        centros = kmeans.treinar(X)

        sigmas = []

        for grupo, centro in zip(
                kmeans.grupos,
                centros):

            sigma = FuncaoGaussiana.calcular_sigma(
                grupo,
                centro
            )

            sigmas.append(sigma)

        self.centros = centros
        self.sigmas = sigmas

        self.camada_oculta = CamadaOculta(
            centros,
            sigmas
        )