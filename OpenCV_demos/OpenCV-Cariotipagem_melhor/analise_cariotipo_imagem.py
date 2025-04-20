import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, filters, morphology, feature
from scipy import ndimage, stats
from sklearn.cluster import KMeans

def carregar_imagem(caminho):
    imagem = cv2.imread(caminho, cv2.IMREAD_COLOR)
    return cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

def pre_processar_imagem(imagem):
    # Converter para escala de cinza
    gray = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
    
    # Equalização de histograma
    equ = cv2.equalizeHist(gray)
    
    # Redução de ruído
    denoised = cv2.fastNlMeansDenoising(equ)
    
    # Detecção de bordas
    edges = feature.canny(denoised, sigma=2)
    
    # Fechamento morfológico
    closed = morphology.closing(edges, morphology.square(3))
    
    return closed

def segmentar_cromossomos(imagem_processada):
    # Rotulagem de componentes conectados
    rotulos = measure.label(imagem_processada)
    
    # Filtrar componentes por área
    areas = [r.area for r in measure.regionprops(rotulos)]
    area_media = np.mean(areas)
    rotulos_filtrados = morphology.remove_small_objects(rotulos, min_size=area_media*0.2)
    
    return rotulos_filtrados

def analisar_cromossomos(rotulos_filtrados, imagem_original):
    propriedades = measure.regionprops(rotulos_filtrados, intensity_image=cv2.cvtColor(imagem_original, cv2.COLOR_RGB2GRAY))
    
    resultados = []
    for prop in propriedades:
        # Características básicas
        area = prop.area
        perimetro = prop.perimeter
        comprimento = prop.major_axis_length
        largura = prop.minor_axis_length
        
        # Índice centromérico
        y, x = prop.centroid
        indice_centrometrico = y / prop.bbox[2]
        
        # Intensidade e textura
        intensidade_media = prop.mean_intensity
        # desvio_padrao_intensidade = prop.standard_deviation_intensity
        
        # Forma
        circularidade = 4 * np.pi * area / (perimetro ** 2)
        excentricidade = prop.eccentricity
        
        # Padrão de bandas (simplificado)
        perfil_intensidade = prop.intensity_image.mean(axis=1)
        picos_bandas = len(feature.peak_local_max(perfil_intensidade, min_distance=5))
        
        resultados.append({
            'area': area,
            'perimetro': perimetro,
            'comprimento': comprimento,
            'largura': largura,
            'indice_centrometrico': indice_centrometrico,
            'intensidade_media': intensidade_media,
            'circularidade': circularidade,
            'excentricidade': excentricidade,
            'numero_bandas': picos_bandas
        })
    
    return resultados

def classificar_cromossomos(resultados):
    # Extrair características para classificação
    features = np.array([[r['comprimento'], r['indice_centrometrico'], r['circularidade']] for r in resultados])
    
    # Normalizar características
    features_norm = (features - features.mean(axis=0)) / features.std(axis=0)
    
    # Aplicar K-means para agrupar cromossomos (assumindo 23 pares)
    kmeans = KMeans(n_clusters=23, random_state=42)
    classificacao = kmeans.fit_predict(features_norm)
    
    # Adicionar classificação aos resultados
    for i, resultado in enumerate(resultados):
        resultado['classificacao'] = int(classificacao[i])
    
    return resultados

def visualizar_resultados(imagem_original, rotulos_filtrados, resultados):
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    
    # Imagem original
    axes[0, 0].imshow(imagem_original)
    axes[0, 0].set_title('Imagem Original')
    axes[0, 0].axis('off')
    
    # Imagem segmentada
    axes[0, 1].imshow(rotulos_filtrados, cmap='nipy_spectral')
    axes[0, 1].set_title('Cromossomos Segmentados')
    axes[0, 1].axis('off')
    
    # Gráfico de dispersão: Comprimento vs. Índice Centromérico
    comprimentos = [r['comprimento'] for r in resultados]
    indices_centrometricos = [r['indice_centrometrico'] for r in resultados]
    classificacoes = [r['classificacao'] for r in resultados]
    scatter = axes[1, 0].scatter(comprimentos, indices_centrometricos, c=classificacoes, cmap='viridis')
    axes[1, 0].set_xlabel('Comprimento')
    axes[1, 0].set_ylabel('Índice Centromérico')
    axes[1, 0].set_title('Comprimento vs. Índice Centromérico')
    plt.colorbar(scatter, ax=axes[1, 0], label='Classificação')
    
    # Histograma de intensidades médias
    intensidades = [r['intensidade_media'] for r in resultados]
    axes[1, 1].hist(intensidades, bins=20)
    axes[1, 1].set_xlabel('Intensidade Média')
    axes[1, 1].set_ylabel('Frequência')
    axes[1, 1].set_title('Distribuição de Intensidades Médias')
    
    plt.tight_layout()
    plt.savefig('resultados_analise_cariotipo.png')
    plt.close()

def gerar_relatorio(resultados):
    with open('relatorio_cariotipo.txt', 'w') as f:
        f.write("Relatório de Análise de Cariótipo\n")
        f.write("=================================\n\n")
        f.write(f"Número total de cromossomos identificados: {len(resultados)}\n\n")
        
        for classe in range(23):
            cromossomos_classe = [r for r in resultados if r['classificacao'] == classe]
            if cromossomos_classe:
                f.write(f"Grupo {classe + 1}:\n")
                comprimentos = [c['comprimento'] for c in cromossomos_classe]
                f.write(f"  Número de cromossomos: {len(cromossomos_classe)}\n")
                f.write(f"  Comprimento médio: {np.mean(comprimentos):.2f}\n")
                f.write(f"  Índice centromérico médio: {np.mean([c['indice_centrometrico'] for c in cromossomos_classe]):.2f}\n")
                f.write(f"  Número médio de bandas: {np.mean([c['numero_bandas'] for c in cromossomos_classe]):.2f}\n\n")

def main(caminho_imagem):
    imagem_original = carregar_imagem(caminho_imagem)
    imagem_processada = pre_processar_imagem(imagem_original)
    rotulos_filtrados = segmentar_cromossomos(imagem_processada)
    resultados = analisar_cromossomos(rotulos_filtrados, imagem_original)
    resultados_classificados = classificar_cromossomos(resultados)
    visualizar_resultados(imagem_original, rotulos_filtrados, resultados_classificados)
    gerar_relatorio(resultados_classificados)
    
    print(f"Análise concluída. {len(resultados)} cromossomos identificados.")
    print("Resultados salvos em 'resultados_analise_cariotipo.png' e 'relatorio_cariotipo.txt'")

if __name__ == "__main__":
    # caminho_imagem = "exemplo_cariotipo.jpg"  # Substitua pelo caminho da sua imagem

    # caminho_imagem = "ITChromo-Leo.png"           # 1 
    # caminho_imagem = "ITChromo-Leo2.png"          # 2
    # caminho_imagem = "ITChromo-FundoPreto.png"    # 3
    caminho_imagem = "Meta.jpg"                     # 4
    
    main(caminho_imagem)
