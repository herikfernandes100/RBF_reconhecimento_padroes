class NormalizadorMinMax:

    def fit_transform(self, X):

        colunas = len(X[0])

        self.minimos = []
        self.maximos = []

        for j in range(colunas):

            coluna = [linha[j] for linha in X]

            self.minimos.append(min(coluna))
            self.maximos.append(max(coluna))

        resultado = []

        for linha in X:

            nova_linha = []

            for j in range(colunas):

                minimo = self.minimos[j]
                maximo = self.maximos[j]

                if maximo == minimo:
                    nova_linha.append(0.0)
                else:
                    valor = (linha[j] - minimo) / (maximo - minimo)
                    nova_linha.append(valor)

            resultado.append(nova_linha)

        return resultado