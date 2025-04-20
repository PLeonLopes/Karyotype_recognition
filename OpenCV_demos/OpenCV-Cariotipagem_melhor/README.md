# Análise de Cariótipos por Imagem

Este projeto realiza uma análise detalhada de cariótipos utilizando técnicas avançadas de processamento de imagens e aprendizado de máquina. Desenvolvido por Mariana Silva de Oliveira.

## Descrição

O script `analise_cariotipo_imagem.py` processa imagens de cariótipos e realiza as seguintes análises:

1. Pré-processamento da imagem
2. Segmentação dos cromossomos
3. Extração de características dos cromossomos
4. Classificação dos cromossomos
5. Visualização dos resultados
6. Geração de relatório detalhado

## Requisitos

- Python 3.6+
- OpenCV (cv2)
- NumPy
- Matplotlib
- scikit-image
- SciPy
- scikit-learn

## Uso

1. Coloque sua imagem de cariótipo no diretório do projeto.

2. Edite a linha 165 do arquivo `analise_cariotipo_imagem.py` para apontar para o caminho da sua imagem:
   ```python
   caminho_imagem = "sua_imagem_cariotipo.jpg"
   ```

3. Execute o script:
   ```
   python3 analise_cariotipo_imagem.py
   ```

4. Os resultados serão salvos em:
   - `resultados_analise_cariotipo.png`: Visualizações gráficas da análise
   - `relatorio_cariotipo.txt`: Relatório detalhado da análise

## Funcionalidades

- Pré-processamento avançado da imagem
- Segmentação precisa dos cromossomos
- Extração de múltiplas características dos cromossomos
- Classificação automática dos cromossomos usando K-means
- Visualização detalhada dos resultados
- Geração de relatório com estatísticas por grupo de cromossomos

## Limitações e Melhorias Futuras

- A classificação atual assume 23 pares de cromossomos (humanos). Pode ser necessário ajustar para outras espécies.
- A detecção de bandas é simplificada e pode ser melhorada com técnicas mais avançadas.
- Futuros desenvolvimentos poderiam incluir:
  - Detecção automática de anomalias cromossômicas
  - Comparação entre múltiplos cariótipos
  - Interface gráfica para facilitar o uso
