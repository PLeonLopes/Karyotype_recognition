import cv2
import numpy as np

# Carregar imagem
imagem = cv2.imread("2.jpg")
gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Pré-processamento (melhorias para robustez)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Segmentação com Watershed (melhoria para separar sobreposição)
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

# Detecção de contornos (otimizado para performance)
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
                # Critério de pareamento simples: distância entre centros
                M2 = cv2.moments(c2)
                cX2 = int(M2["m10"] / M2["m00"])
                cY2 = int(M2["m01"] / M2["m00"])

                distancia = np.sqrt((cX1 - cX2)**2 + (cY1 - cY2)**2)
                limiar_distancia = 100  # Ajuste este valor

                if distancia < limiar_distancia:
                    pares.append((c1, c2))
                    usados[i] = usados[j] = True
                    break

# Numeração dos pares (melhorias)
for i, (c1, c2) in enumerate(pares):
    M1 = cv2.moments(c1)
    cX1 = int(M1["m10"] / M1["m00"])
    cY1 = int(M1["m01"] / M1["m00"])

    M2 = cv2.moments(c2)
    cX2 = int(M2["m10"] / M2["m00"])
    cY2 = int(M2["m01"] / M2["m00"])

    cv2.putText(imagem, str(i + 1), (cX1, cY1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(imagem, str(i + 1), (cX2, cY2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# Exibir e salvar imagem
cv2.imshow("Cromossomos pareados", imagem)
cv2.imwrite("cromossomos_pareados.jpg", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()