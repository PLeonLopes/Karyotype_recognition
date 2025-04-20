import cv2
import numpy as np
import sys
from scipy import stats

def analyze_karyotype(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "Erro: Não foi possível carregar a imagem."

    # Pré-processamento
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Binarização
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Operações morfológicas
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Transformação de distância
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5*dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Watershed para separar cromossomos unidos
    unknown = cv2.subtract(binary, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)
    
    # Encontrar contornos
    contours = []
    for label in np.unique(markers):
        if label == 0 or label == -1:
            continue
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == label] = 255
        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(cnts)
    
    # Filtrar contornos
    min_area = 20
    max_area = 10000
    min_aspect_ratio = 1.02
    chromosomes = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
    chromosomes = [cnt for cnt in chromosomes if cv2.boundingRect(cnt)[3] / cv2.boundingRect(cnt)[2] > min_aspect_ratio]
    
    chromosome_count = len(chromosomes)
    
    # Análise detalhada dos cromossomos
    chromosome_details = []
    for i, cnt in enumerate(chromosomes):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = h / w
        
        # Calcular o índice centromérico
        mask = np.zeros(gray.shape, dtype="uint8")
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        chromosome_image = cv2.bitwise_and(gray, gray, mask=mask)
        profile = np.sum(chromosome_image[y:y+h, x:x+w], axis=1)
        centromere_pos = np.argmin(profile)
        centromeric_index = centromere_pos / h
        
        # Classificar o cromossomo
        if centromeric_index <= 0.125:
            classification = "Telocêntrico"
        elif 0.125 < centromeric_index <= 0.25:
            classification = "Acrocêntrico"
        elif 0.25 < centromeric_index <= 0.375:
            classification = "Submetacêntrico"
        else:
            classification = "Metacêntrico"
        
        chromosome_details.append({
            'number': i+1,
            'area': area,
            'perimeter': perimeter,
            'aspect_ratio': aspect_ratio,
            'centromeric_index': centromeric_index,
            'classification': classification
        })
    
    # Visualizar cromossomos detectados
    vis_image = image.copy()
    for i, cnt in enumerate(chromosomes):
        cv2.drawContours(vis_image, [cnt], -1, (0, 255, 0), 2)
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.putText(vis_image, str(i+1), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imwrite('detected_chromosomes.png', vis_image)
    
    # Preparar resultado
    result = f"Detectados {chromosome_count} cromossomos.\n"
    if chromosome_count == 46:
        result += "Este parece ser um cariótipo humano normal.\n"
    elif chromosome_count < 46:
        result += "Isso pode indicar uma deleção cromossômica ou perda.\n"
    elif chromosome_count > 46:
        result += "Isso pode indicar uma duplicação cromossômica ou ganho.\n"
    
    result += "\nDetalhes dos cromossomos:\n"
    for chrom in chromosome_details:
        result += f"Cromossomo {chrom['number']}:\n"
        result += f"  Área: {chrom['area']:.2f} pixels quadrados\n"
        result += f"  Perímetro: {chrom['perimeter']:.2f} pixels\n"
        result += f"  Razão de aspecto: {chrom['aspect_ratio']:.2f}\n"
        result += f"  Índice centromérico: {chrom['centromeric_index']:.2f}\n"
        result += f"  Classificação: {chrom['classification']}\n\n"
    
    # Análise estatística
    areas = [chrom['area'] for chrom in chromosome_details]
    mean_area = np.mean(areas)
    std_area = np.std(areas)
    z_scores = stats.zscore(areas)
    
    result += "Análise estatística:\n"
    result += f"  Área média dos cromossomos: {mean_area:.2f} pixels quadrados\n"
    result += f"  Desvio padrão da área: {std_area:.2f} pixels quadrados\n"
    result += "  Cromossomos potencialmente anormais (z-score > 2 ou < -2):\n"
    for i, z in enumerate(z_scores):
        if abs(z) > 2:
            result += f"    Cromossomo {i+1}: z-score = {z:.2f}\n"
    
    return result

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Uso: python karyotype_analyzer.py <caminho_da_imagem>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    result = analyze_karyotype(image_path)
    print(result)
    print("Visualização salva como 'detected_chromosomes.png'")
