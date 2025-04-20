import cv2
import numpy as np

# Carregar imagem
imagem = cv2.imread("2.jpg")
gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Pré-processamento (melhorias para robustez)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
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
markers = markers+1
markers[unknown==255] = 0
markers = cv2.watershed(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB), markers)

# Detecção de contornos
contornos = []
for i in range(1, np.max(markers)+1):
    mascara = np.uint8(markers == i)
    contornos_segmentados = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for c in contornos_segmentados:
        if cv2.contourArea(c) > 30: #min_area
            contornos.append(c)

# Pareamento (abordagem simples e rápida)
pares = []
usados = [False] * len(contornos)

for i, c1 in enumerate(contornos):
    if not usados[i]:
        M1 = cv2.moments(c1)
        cX1 = int(M1["m10"] / M1["m00"])
        cY1 = int(M1["m01"] / M1["m00"])

        for j, c2 in enumerate(contornos):
            if i != j and not usados[j]:
                # Critério de pareamento: distância entre centros
                M2 = cv2.moments(c2)
                cX2 = int(M2["m10"] / M2["m00"])
                cY2 = int(M2["m01"] / M2["m00"])

                distancia = np.sqrt((cX1 - cX2)**2 + (cY1 - cY2)**2)
                limiar_distancia = 150  # Ajuste este valor

                # Novo critério: diferença de tamanho (área)
                area1 = cv2.contourArea(c1)
                area2 = cv2.contourArea(c2)
                diferenca_tamanho = abs(area1 - area2)
                limiar_tamanho = 7000

                if distancia < limiar_distancia and diferenca_tamanho < limiar_tamanho:
                    pares.append((c1, c2))
                    usados[i] = usados[j] = True
                    break

# Nova imagem para os pares
if not pares:
    imagem_pares = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.putText(imagem_pares, "Nenhum par encontrado", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
else:
    largura_total = 0
    altura_maxima = 0
    for c1, c2 in pares:
        x1, y1, w1, h1 = cv2.boundingRect(c1)
        x2, y2, w2, h2 = cv2.boundingRect(c2)

        largura_total += w1 + w2 + 10
        altura_maxima = max(altura_maxima, h1, h2)

    imagem_pares = np.zeros((altura_maxima + 20, largura_total, 3), dtype=np.uint8)

    x_offset = 10
    for i, (c1, c2) in enumerate(pares):
        x1, y1, w1, h1 = cv2.boundingRect(c1)
        x2, y2, w2, h2 = cv2.boundingRect(c2)

        roi1 = imagem[y1:y1+h1, x1:x1+w1]
        roi2 = imagem[y2:y2+h2, x2:x2+w2]

        novo_tamanho = (100, 100)  # Ajuste o tamanho conforme necessário

        # Redimensiona ROI1, mantendo a proporção e ajustando a altura
        h_roi1, w_roi1 = roi1.shape[:2]
        if h_roi1 != novo_tamanho[0]:
            proporcao_roi1 = w_roi1 / h_roi1
            novo_w_roi1 = int(novo_tamanho[0] * proporcao_roi1)
            roi1 = cv2.resize(roi1, (novo_w_roi1, novo_tamanho[0]))

        # Redimensiona ROI2, mantendo a proporção e ajustando a altura
        h_roi2, w_roi2 = roi2.shape[:2]
        if h_roi2 != novo_tamanho[0]:
            proporcao_roi2 = w_roi2 / h_roi2
            novo_w_roi2 = int(novo_tamanho[0] * proporcao_roi2)
            roi2 = cv2.resize(roi2, (novo_w_roi2, novo_tamanho[0]))

        # Ajusta a altura e largura de inserção para ROI1
        h_insercao1 = min(roi1.shape[0], imagem_pares.shape[0] - 10)
        w_insercao1 = min(roi1.shape[1], imagem_pares.shape[1] - x_offset)
        imagem_pares[10:10+h_insercao1, x_offset:x_offset+w_insercao1] = roi1[:h_insercao1, :w_insercao1]

        # Ajusta a altura e largura de inserção para ROI2, levando em conta o espaço
        h_insercao2 = min(roi2.shape[0], imagem_pares.shape[0] - 10)
        espaco_restante = imagem_pares.shape[1] - x_offset - roi1.shape[1] - 10
        w_insercao2 = min(roi2.shape[1], espaco_restante)
        
        # Verifica se há espaço suficiente antes de inserir ROI2
        if w_insercao2 > 0:  # Adiciona esta condição
            imagem_pares[10:10+h_insercao2, x_offset+roi1.shape[1]+10:x_offset+roi1.shape[1]+10+w_insercao2] = roi2[:h_insercao2, :w_insercao2]

        x_offset += roi1.shape[1] + w_insercao2 + 10  # Atualiza x_offset corretamente

# Exibir e salvar imagem
cv2.imshow("Pares organizados", imagem_pares)
cv2.imwrite("pares_organizados.jpg", imagem_pares)
cv2.waitKey(0)
cv2.destroyAllWindows()