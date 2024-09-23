from django.db import models
import numpy as np
import json

class MadalineModel:
    def __init__(self):
        # Inicialização de atributos
        self.entAux = None  # Entradas (padrões)
        self.targAux = None  # Saídas (alvos)
        self.padroes = 0  # Número de padrões
        self.entradas = 0  # Número de entradas
        self.numSaidas = 5  # Número de saídas
        self.limiar = 0.0  # Limiar para a ativação
        self.alfa = 0.01  # Taxa de aprendizado
        self.erro_tolerado = 0.01  # Erro tolerado
        self.v = None  # Pesos
        self.v0 = None  # Bias
        self.yin = None  # Somas de entrada
        self.y = None  # Saídas
        
    def save_weights(self, filename='weights.json'):
        data_to_save = {
            'v': self.v.tolist(),  # Converter para lista
            'v0': self.v0.tolist()  # Converter para lista
        }
        with open(filename, 'w') as json_file:
            json.dump(data_to_save, json_file)

    def load_weights(self, filename='weights.json'):
        with open(filename, 'r') as json_file:
            data = json.load(json_file)
            self.v = np.array(data['v'])  # Converter de volta para numpy array
            self.v0 = np.array(data['v0'])  # Converter de volta para numpy array

    def train(self, entradas, targets):
        self.entAux = np.array(entradas)
        self.targAux = np.array(targets)
        self.padroes, self.entradas = self.entAux.shape
        self.numSaidas = self.targAux.shape[1]

        # Inicialização dos pesos e bias
        self.v = np.random.uniform(-0.1, 0.1, (self.entradas, self.numSaidas))
        self.v0 = np.random.uniform(-0.1, 0.1, self.numSaidas)

        # Inicializa yin e y
        self.yin = np.zeros((self.numSaidas,))
        self.y = np.zeros((self.numSaidas,))

        erro = float('inf')  # Definido para entrar no loop
        ciclo = 0

        if self.numSaidas <= 0:
         raise ValueError("Número de saídas não pode ser zero.")
        else:
            print("NumSaida", self.numSaidas)


        while erro > self.erro_tolerado:
            ciclo += 1
            erro = 0
            for i in range(self.padroes):
                padraoLetra = self.entAux[i, :]

                for m in range(self.numSaidas):
                    soma = np.dot(padraoLetra, self.v[:, m]) + self.v0[m]
                    self.yin[m] = soma

                # Normalizando a saída
                self.y = np.array([1.0 if self.yin[j] >= self.limiar else 0.0 for j in range(self.numSaidas)])

                # Cálculo do erro
                for j in range(self.numSaidas):
                    erro += 0.5 * ((self.targAux[i][j] - self.y[j]) ** 2)

                # Atualização dos pesos
                for m in range(self.entradas):
                    for n in range(self.numSaidas):
                        self.v[m][n] += self.alfa * (self.targAux[i][n] - self.y[n]) * padraoLetra[m]

                for j in range(self.numSaidas):
                    self.v0[j] += self.alfa * (self.targAux[i][j] - self.y[j])

            print(f"Ciclo: {ciclo}, Erro: {erro}")
            
            if erro <= self.erro_tolerado:
                self.save_weights()
                print("Salvo com sucesso")

    def predict(self, input_data):
    # Verificação se os pesos foram carregados
     if self.v is None or self.v0 is None:
        self.load_weights()
        print("Pesos carregados:")
        print(f"Dimensões de v: {self.v.shape}")
        print(f"Dimensões de v0: {self.v0.shape}")

    # Verificação do tamanho da entrada
     if input_data is None or len(input_data) == 0:
        raise ValueError("Arquivo não pode ser vazio.")
    
     if len(input_data) != 100:
        raise ValueError(f"O tamanho da entrada deve ser 100, mas recebeu {len(input_data)}.")

    # Inicialização de yin
     self.yin = np.zeros((self.numSaidas,))
    
    # Debug: Verificar o número de saídas
     print(f"Calculando valores de yin para {self.numSaidas} saídas.")

    # Cálculo das somas e verificação dos valores intermediários
     for j in range(self.numSaidas):
        soma = np.dot(input_data, self.v[:, j]) + self.v0[j]
        self.yin[j] = soma
        
        # Debugging prints
        print(f"Para saída {j}: soma = {soma}, yin[{j}] = {self.yin[j]}")

    # Verificar os valores de yin antes da ativação
     print(f"Valores finais de yin: {self.yin}")
     print(f"Limite de ativação: {self.limiar}")

    # Aplicação da função de ativação
     self.y = np.array([1.0 if val >= self.limiar else 0.0 for val in self.yin.flatten()])
    
    # Verificação do resultado final
     print(f"Saídas (y): {self.y}")
    
     return self.y


 
    
class MadalineLetterMap(models.Model):
    letter_map = models.JSONField()  # Campo JSON para armazenar o letter_map
    created_at = models.DateTimeField(auto_now_add=True)  # Para saber quando foi salvo

    def __str__(self):
        return f"Letter Map - {self.created_at}"




