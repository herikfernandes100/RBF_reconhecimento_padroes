import csv

class Loader:

    @staticmethod
    def carregar_csv(caminho):
        dados = []

        with open(caminho, "r", encoding="utf-8") as arquivo:
            leitor = csv.reader(arquivo)

            next(leitor)

            for linha in leitor:
                dados.append([float(valor) for valor in linha])

        return dados