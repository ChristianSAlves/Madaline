# backend/app.py
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Permite que o frontend acesse o backend

# Classe MADALINE para treinar e prever letras
class Madaline:
    def __init__(self, input_size, num_classes, learning_rate=0.01, max_epochs=1000):
        self.input_size = input_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = np.random.uniform(-1, 1, (num_classes, input_size))
        self.bias = np.random.uniform(-1, 1, num_classes)

    def activation_function(self, x):
        return np.where(x >= 0, 1, -1)

    def train(self, inputs, targets):
        for epoch in range(self.max_epochs):
            error_count = 0
            for x, target in zip(inputs, targets):
                outputs = self.predict(x)
                errors = target - outputs
                error_count += np.sum(errors != 0)
                for i in range(self.num_classes):
                    if errors[i] != 0:
                        self.weights[i] += self.learning_rate * errors[i] * x
                        self.bias[i] += self.learning_rate * errors[i]
            if error_count == 0:
                break

    def predict(self, input_data):
        net_input = np.dot(self.weights, input_data) + self.bias
        return self.activation_function(net_input)

# Inicializa o MADALINE
madaline = None
letras_dict = {}  # Dicionário para mapear índices a letras dinâmicas
num_classes = 0  # Número de classes (letras) a ser definido dinamicamente

# Rota para treinar o modelo com as matrizes e letras correspondentes
@app.route('/letras/train/', methods=['POST'])
def train():
    global madaline, letras_dict, num_classes
    data = request.json
    matrices = data['matrices']  # Lista de matrizes de entrada (10x10 para cada letra)
    labels = data['labels']      # Lista de letras correspondentes às matrizes

    # Definindo o número de classes e o dicionário de letras
    letras_dict = {i: label for i, label in enumerate(labels)}
    num_classes = len(labels)
    madaline = Madaline(input_size=100, num_classes=num_classes)  # 100 entradas (10x10)

    # Converter as letras em vetores de destino binários
    targets = []
    for label in labels:
        target_vector = [-1] * num_classes
        for key, letra in letras_dict.items():
            if letra == label:
                target_vector[key] = 1
        targets.append(target_vector)
    
    # Achatar as matrizes de 10x10 para vetores de 100 elementos
    flattened_matrices = [np.array(matrix).flatten() for matrix in matrices]

    # Treinar o modelo Madaline
    madaline.train(flattened_matrices, targets)

    return jsonify({'message': 'Modelo treinado com sucesso!', 'labels': letras_dict})

# Rota para fazer predição com uma matriz de teste
@app.route('/letras/predict/', methods=['POST'])
def predict():
    data = request.json
    matrix = data['matrix']  # Matriz 10x10 para previsão
    flattened_matrix = np.array(matrix).flatten()  # Achatar a matriz

    prediction = madaline.predict(flattened_matrix)
    predicted_index = np.argmax(prediction)  # Encontrar o índice da letra com maior ativação
    predicted_label = letras_dict[predicted_index]  # Obter a letra correspondente

    return jsonify({'letra_predita': predicted_label})

if __name__ == '__main__':
    app.run(debug=True, port=8000)
