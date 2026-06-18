from rbf.rede_rbf import RedeRBF


class TreinadorRBF:

    def __init__(self, n_centros):

        self.rede = RedeRBF(n_centros)

    def treinar(self, X, y):

        self.rede.treinar_primeiro_estagio(X)

        H = self.rede.camada_oculta.transformar(X)

        return H