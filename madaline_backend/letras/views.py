from django.http import JsonResponse, HttpResponse
from django.views import View
from pathlib import Path
import json
import os
import numpy as np
from .models import MadalineModel, MadalineLetterMap  # Certifique-se de importar sua classe MadalineModel

def letras_home(request):
    return HttpResponse("Welcome to the Letras Home Page!")

class MadalineTrainView(View):
    def post(self, request, *args, **kwargs):
        try:
            data = json.loads(request.body)
            matrices = data.get('matrices')
            labels = data.get('labels')

            # Validação dos dados de entrada
            if not matrices or not labels or len(matrices) != 5 or len(labels) != 5:
                return JsonResponse({"message": "Dados de entrada inválidos."}, status=400)

            # Cria a instância do modelo Madaline
            model = MadalineModel()

            # Preparação dos dados de entrada
            input_data = [np.array(matrix).flatten() for matrix in matrices]

            # Criação do letter_map baseado na ordem dos labels recebidos
            letter_map = {label: idx for idx, label in enumerate(labels)}

            # Inicializando saidas_desejadas com zeros, tamanho baseado no número de letras no letter_map
            saidas_desejadas = [[0] * len(letter_map) for _ in range(len(labels))]

            # Preenchendo saidas_desejadas de acordo com a posição de cada label no letter_map
            for i, label in enumerate(labels):
             saidas_desejadas[i][letter_map[label]] = 1

            model.train(input_data, saidas_desejadas)  # Treina o modelo

            # Salvar o letter_map em um arquivo JSON
            maps_directory = 'maps'
            os.makedirs(maps_directory, exist_ok=True)  # Cria o diretório se não existir
            letter_map_path = os.path.join(maps_directory, 'letter_map.json')

            with open(letter_map_path, 'w') as f:
                json.dump(letter_map, f)

            return JsonResponse({"message": "Treinamento realizado com sucesso!"})

        except Exception as e:
            return JsonResponse({"message": f"Erro no treinamento: {str(e)}"}, status=500)

class MadalinePredictView(View):
    def post(self, request, *args, **kwargs):
        try:
            model = MadalineModel()
            data = json.loads(request.body)
            matriz = data.get('matrix')

            # Verificar se a matriz é válida
            if matriz is None or len(matriz) != 10 or any(len(arr) != 10 for arr in matriz):
                return JsonResponse({"error": "Matriz inválida."}, status=400)

            # Normalizar a entrada
            entrada = np.array(matriz).flatten() / 255.0  # Supondo que os valores vão de 0 a 255
            print("Entrada normalizada:", entrada)  # Debugging

            saidas = model.predict(entrada)
            
            print('Saidas', saidas)

            if len(saidas) == 0:  # Adicionando verificação para saídas
                return JsonResponse({"error": "Erro: Previsão retornou vazio."}, status=500)

            max_saida_idx = np.argmax(saidas)
            max_saida = saidas[max_saida_idx]

            # Carregar o letter_map do arquivo JSON
            file_path = Path("maps/letter_map.json")
            if not file_path.exists():
                return JsonResponse({"error": "Mapeamento de letras não encontrado."}, status=400)

            with file_path.open("r") as file:
                letter_map = json.load(file)

            letra_predita = "Não reconhecida"
            for letra, idx in letter_map.items():
                if idx == max_saida_idx:
                    letra_predita = letra
                    break

            return JsonResponse({"letra_predita": letra_predita})

        except Exception as e:
            print(f"Matriz recebida para previsão: {matriz}")
            return JsonResponse({"error": f"Erro durante a previsão: {str(e)}"}, status=500)


