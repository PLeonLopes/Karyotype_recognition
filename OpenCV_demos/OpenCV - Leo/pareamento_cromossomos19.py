import cv2
import numpy as np
import os

# Nome da imagem esperada
nome_imagem_esperado = "2.jpg"
caminho_imagem = "2.jpg"  # Substitua pelo nome correto

if os.path.exists(caminho_imagem):
    os.rename(caminho_imagem, nome_imagem_esperado)

# Carregar imagem
imagem_original = cv2.imread(nome_imagem_esperado)
if imagem_original is None:
    raise Exception("Erro: A imagem não foi carregada corretamente!")

# Converter para escala de cinza
gray = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2GRAY)

# Aplicar threshold para segmentação
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Encontrar contornos
contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Lista de cromossomos extraídos
cromossomos = []
for contorno in contornos:
    x, y, w, h = cv2.boundingRect(contorno)
    cromossomo = imagem_original[y:y+h, x:x+w]
    cromossomos.append((cromossomo, (x, y, w, h)))  # Guardar posição para referência

# Ordenar cromossomos da esquerda para a direita
cromossomos.sort(key=lambda item: item[1][0])

# Função para calcular similaridade
def calcular_similaridade(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Redimensiona para o mesmo tamanho antes de comparar
    altura = min(img1.shape[0], img2.shape[0])
    largura = min(img1.shape[1], img2.shape[1])
    
    img1 = cv2.resize(img1, (largura, altura))
    img2 = cv2.resize(img2, (largura, altura))

    return cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED).max()

# Criar pares com base na similaridade
pares = []
usados = set()

for i in range(len(cromossomos)):
    if i in usados:
        continue

    melhor_match = None
    melhor_similaridade = -1
    cromossomo1, pos1 = cromossomos[i]

    for j in range(i+1, len(cromossomos)):
        if j in usados:
            continue

        cromossomo2, pos2 = cromossomos[j]
        similaridade = calcular_similaridade(cromossomo1, cromossomo2)

        if similaridade > melhor_similaridade:
            melhor_similaridade = similaridade
            melhor_match = j

    if melhor_match is not None:
        usados.add(i)
        usados.add(melhor_match)
        pares.append((cromossomos[i][0], cromossomos[melhor_match][0]))

# Definir tamanho fixo para os pares
nova_largura = 35  # Reduzido para melhor distribuição
nova_altura = 75   # Mantendo proporção

pares_redimensionados = []
for cromossomo1, cromossomo2 in pares:
    cromossomo1_resized = cv2.resize(cromossomo1, (nova_largura, nova_altura))
    cromossomo2_resized = cv2.resize(cromossomo2, (nova_largura, nova_altura))
    
    pares_redimensionados.append((cromossomo1_resized, cromossomo2_resized))

# Criar imagem final organizada como na referência
linhas = 5
colunas = 6
espacamento = 10
altura_total = (nova_altura + espacamento) * linhas
largura_total = (nova_largura * 2 + espacamento) * colunas

imagem_final = np.ones((altura_total, largura_total, 3), dtype=np.uint8) * 255

x_offset = espacamento
y_offset = espacamento

for i, (cromossomo1, cromossomo2) in enumerate(pares_redimensionados):
    if i % colunas == 0 and i != 0:
        x_offset = espacamento
        y_offset += nova_altura + espacamento

    # Ajuste final para garantir o tamanho correto
    cromossomo1 = cv2.resize(cromossomo1, (nova_largura, nova_altura))
    cromossomo2 = cv2.resize(cromossomo2, (nova_largura, nova_altura))

    imagem_final[y_offset:y_offset + nova_altura, x_offset:x_offset + nova_largura] = cromossomo1
    imagem_final[y_offset:y_offset + nova_altura, x_offset + nova_largura + espacamento:x_offset + nova_largura * 2 + espacamento] = cromossomo2
    
    x_offset += (nova_largura * 2 + espacamento)

# Salvar a imagem final
cv2.imwrite("resultado_pareamento.png", imagem_final)

# Exibir o resultado
cv2.imshow("Pares de Cromossomos", imagem_final)
cv2.waitKey(0)
cv2.destroyAllWindows()
