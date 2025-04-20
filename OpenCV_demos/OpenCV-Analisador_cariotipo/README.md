# Analisador de Cariótipo

Este projeto é um analisador de cariótipo automatizado desenvolvido em Python. Ele utiliza técnicas de processamento de imagem e visão computacional para detectar, contar e analisar cromossomos em imagens de cariótipos.

## Funcionalidades

- Detecção automática de cromossomos em imagens de cariótipos
- Contagem precisa do número de cromossomos
- Análise detalhada de cada cromossomo, incluindo:
  - Área
  - Perímetro
  - Razão de aspecto
  - Índice centromérico
  - Classificação (Telocêntrico, Acrocêntrico, Submetacêntrico, Metacêntrico)
- Análise estatística dos cromossomos detectados
- Identificação de possíveis anomalias cromossômicas
- Visualização dos cromossomos detectados

## Requisitos

- Python 3.7+
- OpenCV
- NumPy
- SciPy

## Uso

Execute o script principal fornecendo o caminho para a imagem do cariótipo:

```
python karyotype_analyzer.py caminho/para/sua/imagem.png
```

O script irá analisar a imagem e fornecer um relatório detalhado sobre os cromossomos detectados. Além disso, ele salvará uma imagem com os cromossomos detectados e numerados como 'detected_chromosomes.png'.

## Saída

O script fornece as seguintes informações:

1. Número total de cromossomos detectados
2. Possível interpretação do número de cromossomos (normal, deleção, duplicação)
3. Detalhes de cada cromossomo detectado
4. Análise estatística dos cromossomos
5. Identificação de cromossomos potencialmente anormais

## Limitações

- O desempenho do analisador pode variar dependendo da qualidade da imagem de entrada
- A precisão da detecção e classificação dos cromossomos pode ser afetada por sobreposições ou distorções na imagem
