import cv2
import numpy as np

# Carregar imagem
imagem = cv2.imread("1.jpg")
gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Segmentação
thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Detecção e filtragem de contornos
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

min_area = 50  # Ajuste este valor se necessário
cromossomos = []
for c in cnts:
    if cv2.contourArea(c) > min_area:
        cromossomos.append(c)

# Pareamento (usando momentos de Hu)
pares = []
for i, c1 in enumerate(cromossomos):
    M1 = cv2.moments(c1)
    hu1 = cv2.HuMoments(M1).flatten()

    for j, c2 in enumerate(cromossomos):
        if i != j:
            M2 = cv2.moments(c2)
            hu2 = cv2.HuMoments(M2).flatten()

            # Comparar momentos de Hu (distância Euclidiana)
            distancia = np.linalg.norm(hu1 - hu2)
            similaridade = 1 / (1 + distancia)

            limiar = 0.8  # Ajuste este valor se necessário
            if similaridade > limiar:
                pares.append((c1, c2))

# Numeração dos pares
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