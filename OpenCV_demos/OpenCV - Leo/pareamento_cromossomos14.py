import cv2
import numpy as np
import os

# Carregar imagem
imagem = cv2.imread("2.jpg")
gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Pré-processamento (melhorias para robustez)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Aumentei o tamanho do kernel para suavizar mais
thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Segmentação com Watershed
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
sure_bg = cv2.dilate(opening, kernel, iterations=3)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0
markers = cv2.watershed(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB), markers)

# Detecção de contornos
contornos = []
for i in range(1, np.max(markers)+1):
    mascara = np.uint8(markers == i)
    contornos_segmentados = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for c in contornos_segmentados:
        if cv2.contourArea(c) > 30:  # Min área
            contornos.append(c)

# Pareamento com mais filtros (distância, tamanho e forma)
pares = []
usados = [False] * len(contornos)
ranking_pares = []  # Lista para armazenar a "qualidade" de cada par

for i, c1 in enumerate(contornos):
    if not usados[i]:
        M1 = cv2.moments(c1)
        cX1 = int(M1["m10"] / M1["m00"])
        cY1 = int(M1["m01"] / M1["m00"])
        area1 = cv2.contourArea(c1)

        for j, c2 in enumerate(contornos):
            if i != j and not usados[j]:
                M2 = cv2.moments(c2)
                cX2 = int(M2["m10"] / M2["m00"])
                cY2 = int(M2["m01"] / M2["m00"])
                area2 = cv2.contourArea(c2)

                # Distância entre os centros
                distancia = np.sqrt((cX1 - cX2)**2 + (cY1 - cY2)**2)
                limiar_distancia = 150  # Ajuste este valor

                # Diferença de tamanho
                diferenca_tamanho = abs(area1 - area2)
                limiar_tamanho = 7000  # Ajuste este valor

                # Similaridade de forma (momento)
                momento1 = cv2.HuMoments(M1).flatten()
                momento2 = cv2.HuMoments(M2).flatten()
                forma_diferenca = np.sum(np.abs(momento1 - momento2))  # Diferença de forma

                limiar_forma = 0.1  # Ajuste este valor conforme necessário

                if distancia < limiar_distancia and diferenca_tamanho < limiar_tamanho and forma_diferenca < limiar_forma:
                    # Calcula um "score" para o par (quanto menor, melhor)
                    score = distancia + diferenca_tamanho / 1000 + forma_diferenca * 10  # Peso maior para forma
                    ranking_pares.append((score, c1, c2))
                    usados[i] = usados[j] = True
                    break

# Seleciona os 23 pares mais prováveis
ranking_pares.sort()  # Ordena por score
pares_selecionados = [
    (c1, c2) for score, c1, c2 in ranking_pares[:23]
]  # Pega os 23 primeiros

# Define o número máximo de pares a serem exibidos
num_pares = min(23, len(pares_selecionados))

if pares_selecionados:
    # Calcula o número de linhas e colunas da grade
    num_linhas = 4  # Ajuste este valor para controlar o número de linhas
    num_colunas = (num_pares + num_linhas - 1) // num_linhas  # Cálculo dinâmico

    # Altura e largura fixas para os pares
    altura_par = 80
    largura_par = 160

    # Cria uma imagem em branco para os pares
    largura_total = largura_par * 2 * num_colunas
    altura_total = altura_par * num_linhas
    img_pares = np.ones((altura_total, largura_total, 3), dtype=np.uint8) * 255  # Fundo branco

    # Loop para exibir os pares
    indice_par = 0
    for linha in range(num_linhas):
        for coluna in range(num_colunas):
            if indice_par < num_pares:
                # Obtém o par de cromossomos
                c1, c2 = pares_selecionados[indice_par]

                # Extrai os cromossomos das imagens originais
                x1, y1, w1, h1 = cv2.boundingRect(c1)
                x2, y2, w2, h2 = cv2.boundingRect(c2)
                cromossomo1 = imagem[y1:y1 + h1, x1:x1 + w1]
                cromossomo2 = imagem[y2:y2 + h2, x2:x2 + w2]

                # Redimensiona os cromossomos para a altura padrão
                cromossomo1_redimensionado = cv2.resize(cromossomo1, (largura_par, altura_par))
                cromossomo2_redimensionado = cv2.resize(cromossomo2, (largura_par, altura_par))

                # Concatena os cromossomos horizontalmente
                par_horizontal = np.concatenate((cromossomo1_redimensionado, cromossomo2_redimensionado), axis=1)

                # Calcula as coordenadas de inserção do par na imagem
                x_offset = coluna * largura_par * 2
                y_offset = linha * altura_par

                # Insere o par na imagem final
                img_pares[y_offset:y_offset + altura_par, x_offset:x_offset + largura_par * 2] = par_horizontal

                # Adiciona o número do par
                cv2.putText(img_pares, str(indice_par + 1), (x_offset + largura_par, y_offset + altura_par // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

                indice_par += 1

    # Salvar a imagem no mesmo diretório
    nome_arquivo = "pares_de_cromossomos.jpg"
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_completo = os.path.join(diretorio_atual, nome_arquivo)
    cv2.imwrite(caminho_completo, img_pares)

    # Exibir a imagem
    cv2.imshow("Pares de Cromossomos", img_pares)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Nenhum par encontrado.")
