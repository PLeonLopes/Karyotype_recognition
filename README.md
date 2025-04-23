# **Reconhecimento de Cariótipo**  
**Conjunto de testes para implementação de sistema de reconhecimento de cariótipos**  

🔬 *Explorando abordagens computacionais para análise automatizada de cariótipos*  

---  

## **📝 Descrição**  
Repositório dedicado ao estudo e implementação de técnicas de visão computacional e aprendizado profundo para reconhecimento e classificação de cariótipos humanos.  

Foram testados quatro diferentes abordagens de implementação:

1. **OpenCV**: _Processamento de imagem tradicional_
2. **Sam (Meta)**: _Segmentação avançada de Objetos_
3. **VGG16**: _Classificação baseada em redes convolucionais_
4. **YoloV10**: _Detecção de objetos com YOLOv10_

Os testes de implementação foram realizados com base em experimentos com o projeto ITChromo e bancos de dados públicos de citogenética.

---  

## **🔍 Abordagens Exploradas**  

### **🟢 OpenCV**  
- Pré-processamento de imagens de cariótipos  
- Filtros para realce de bandas cromossômicas  
- Técnicas de thresholding e contorno  
- Classificação baseada em morfologia  

📌 *Vantagens*: O método se mostra ser mais leve computcionalmente e de fácil prototipagem.  
⚠ *Desafios*: Baixa robustez e muito limitado para cariótipos complexos  

---  

### **🟣 Segment Anything (SAM) - Meta**
- Segmentação zero-shot de cromossomos
- Aplicação direta em imagens complexas
- Possibilidade de integração com alguns classificadores personalizados

📌 Vantagens: Maior poder de segmentação de objetos.
⚠ Desafios: Requer um maior poder computacional e comete certos erros.

---

### 🟠 VGG16
- Uso de rede neural convolucional profunda (CNN)
- Extração de características cromossômicas e bounding boxes
- Classificação supervisada por pares cromossômicos
- Treinamento com imagens segmentadas ou "cropadas"

📌 Vantagens: Bom desempenho com dados balanceados e bem preparados
⚠ Desafios: Maior dificuldade de implementação e exige maior esforço de pré-processamento/tuning

---

### **🔵 YOLOv10**  
- Detecção individual de cromossomos  
- Treinamento com conjunto de dados anotados  
- Comparação entre YOLOv8 e v10  
- Pós-processamento para ordenação  

📌 *Vantagens*: Possui maior precisão para objetos distintos  
⚠ *Desafios*: Requer grande conjunto de dados anotados e ALTÍSSIMO custo computacional

---

## **📂 Estrutura do Projeto**  

.<br>
├── OpenCV_demos/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# teste de implementação com o OpenCV
<br>├── SAM2_demos/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# teste de implementação com o SAM2
<br>├── VGG16_demos/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# teste de implementação com o VGG16
<br>├── datasets/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Datasets utilizados
<br>├── pesquisas.md&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Compilado de pesquisas sobre o tema
<br>└── README.md&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Este documento

---  
