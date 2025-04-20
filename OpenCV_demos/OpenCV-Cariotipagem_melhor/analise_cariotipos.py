
import os
import re
from Bio import SeqIO
from Bio.SeqUtils import GC, molecular_weight, CodonUsage
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.Seq import Seq
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def analisar_cariotipos(arquivo_fasta):
    sequencias = list(SeqIO.parse(arquivo_fasta, "fasta"))
    resultados = []
    
    for seq in sequencias:
        tamanho = len(seq)
        gc_content = GC(seq.seq)
        at_content = 100 - gc_content
        peso_molecular = molecular_weight(seq.seq)
        regioes_repetitivas = identificar_regioes_repetitivas(seq.seq)
        densidade_genes = simular_densidade_genes(tamanho)
        ilhas_cpg = identificar_ilhas_cpg(seq.seq)
        complexidade = calcular_complexidade_sequencia(seq.seq)
        motivos = encontrar_motivos(seq.seq)
        estruturas_secundarias = prever_estruturas_secundarias(seq.seq)
        codon_usage = analisar_codon_usage(seq.seq)
        orfs = identificar_orfs(seq.seq)
        composicao_aminoacidos = analisar_composicao_aminoacidos(orfs)
        
        resultados.append({
            'nome': seq.id,
            'tamanho': tamanho,
            'gc_content': gc_content,
            'at_content': at_content,
            'peso_molecular': peso_molecular,
            'regioes_repetitivas': regioes_repetitivas,
            'densidade_genes': densidade_genes,
            'ilhas_cpg': ilhas_cpg,
            'complexidade': complexidade,
            'motivos': motivos,
            'estruturas_secundarias': estruturas_secundarias,
            'codon_usage': codon_usage,
            'orfs': orfs,
            'composicao_aminoacidos': composicao_aminoacidos
        })
    
    return resultados

def identificar_regioes_repetitivas(seq, min_repeat_length=10):
    repeats = []
    for i in range(len(seq) - min_repeat_length + 1):
        for j in range(i + min_repeat_length, len(seq) + 1):
            subseq = seq[i:j]
            if seq.count(subseq) > 1:
                repeats.append((i, j, len(subseq), seq.count(subseq)))
    return repeats

def simular_densidade_genes(tamanho, gene_density=0.01):
    num_genes = int(tamanho * gene_density)
    gene_positions = sorted(np.random.choice(tamanho, num_genes, replace=False))
    return gene_positions

def identificar_ilhas_cpg(seq, min_length=200, min_gc=50, min_obs_exp=0.6):
    ilhas = []
    for i in range(len(seq) - min_length + 1):
        subseq = seq[i:i+min_length]
        gc_content = GC(subseq)
        obs_exp = calcular_obs_exp_cpg(subseq)
        if gc_content >= min_gc and obs_exp >= min_obs_exp:
            ilhas.append((i, i+min_length))
    return ilhas

def calcular_obs_exp_cpg(seq):
    c_count = seq.count('C')
    g_count = seq.count('G')
    cg_count = seq.count('CG')
    if c_count > 0 and g_count > 0:
        exp_cg = (c_count * g_count) / len(seq)
        return cg_count / exp_cg
    return 0

def calcular_complexidade_sequencia(seq, window_size=100):
    complexidades = []
    for i in range(0, len(seq) - window_size + 1, window_size):
        subseq = seq[i:i+window_size]
        complexidade = len(set(subseq)) / window_size
        complexidades.append(complexidade)
    return np.mean(complexidades)


def encontrar_motivos(seq, motivos=['GAATTC', 'GGATCC', 'CTGCAG']):  # EcoRI, BamHI, PstI
    resultados = {}
    for motivo in motivos:
        resultados[motivo] = len(re.findall(motivo, str(seq)))
    return resultados

def prever_estruturas_secundarias(seq):
    # Simplificação: apenas contagem de possíveis grampos
    return len(re.findall(r'G{3,}.{1,7}C{3,}', str(seq)))

def analisar_codon_usage(seq):
    return CodonUsage.CodonAdaptationIndex().cai_for_gene(str(seq))

def identificar_orfs(seq, min_length=100):
    orfs = []
    for strand, nuc in [(+1, seq), (-1, seq.reverse_complement())]:
        for frame in range(3):
            for pro in nuc[frame:].translate(table="Standard").split("*"):
                if len(pro) >= min_length/3:
                    orfs.append(str(pro))
    return orfs

def analisar_composicao_aminoacidos(orfs):
    if not orfs:
        return {}
    
    composicao_total = Counter()
    for orf in orfs:
        analise = ProteinAnalysis(orf)
        composicao_total.update(analise.count_amino_acids())
    
    # Normalizar para porcentagem
    total = sum(composicao_total.values())
    return {aa: (count / total) * 100 for aa, count in composicao_total.items()}

def visualizar_resultados(resultados):
    nomes = [r['nome'] for r in resultados]
    tamanhos = [r['tamanho'] for r in resultados]
    gc_contents = [r['gc_content'] for r in resultados]
    at_contents = [r['at_content'] for r in resultados]
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 20))
    
    # Gráficos anteriores...
    axes[0, 0].bar(nomes, tamanhos)
    axes[0, 0].set_title('Tamanho dos cromossomos')
    axes[0, 0].set_xlabel('Cromossomos')
    axes[0, 0].set_ylabel('Tamanho (pb)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    axes[0, 1].bar(nomes, gc_contents, label='GC')
    axes[0, 1].bar(nomes, at_contents, bottom=gc_contents, label='AT')
    axes[0, 1].set_title('Composição de bases dos cromossomos')
    axes[0, 1].set_xlabel('Cromossomos')
    axes[0, 1].set_ylabel('Porcentagem (%)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].legend()
    
    axes[0, 2].scatter(gc_contents, tamanhos)
    for i, nome in enumerate(nomes):
        axes[0, 2].annotate(nome, (gc_contents[i], tamanhos[i]))
    axes[0, 2].set_title('Relação entre conteúdo GC e tamanho dos cromossomos')
    axes[0, 2].set_xlabel('Conteúdo GC (%)')
    axes[0, 2].set_ylabel('Tamanho (pb)')
    
    gene_densities = [len(r['densidade_genes']) / r['tamanho'] for r in resultados]
    axes[1, 0].bar(nomes, gene_densities)
    axes[1, 0].set_title('Densidade de genes (simulada)')
    axes[1, 0].set_xlabel('Cromossomos')
    axes[1, 0].set_ylabel('Genes por base')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    complexidades = [r['complexidade'] for r in resultados]
    axes[1, 1].bar(nomes, complexidades)
    axes[1, 1].set_title('Complexidade da sequência')
    axes[1, 1].set_xlabel('Cromossomos')
    axes[1, 1].set_ylabel('Complexidade média')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Novos gráficos
    motivos_totais = [sum(r['motivos'].values()) for r in resultados]
    axes[1, 2].bar(nomes, motivos_totais)
    axes[1, 2].set_title('Total de motivos encontrados')
    axes[1, 2].set_xlabel('Cromossomos')
    axes[1, 2].set_ylabel('Número de motivos')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    estruturas_secundarias = [r['estruturas_secundarias'] for r in resultados]
    axes[2, 0].bar(nomes, estruturas_secundarias)
    axes[2, 0].set_title('Potenciais estruturas secundárias')
    axes[2, 0].set_xlabel('Cromossomos')
    axes[2, 0].set_ylabel('Número de estruturas')
    axes[2, 0].tick_params(axis='x', rotation=45)
    
    codon_usage = [r['codon_usage'] for r in resultados]
    axes[2, 1].bar(nomes, codon_usage)
    axes[2, 1].set_title('Índice de Adaptação de Códons')
    axes[2, 1].set_xlabel('Cromossomos')
    axes[2, 1].set_ylabel('CAI')
    axes[2, 1].tick_params(axis='x', rotation=45)
    
    # Ideograma do cariótipo
    ax_ideogram = axes[2, 2]
    y_positions = np.arange(len(nomes))
    ax_ideogram.barh(y_positions, tamanhos, height=0.5)
    ax_ideogram.set_yticks(y_positions)
    ax_ideogram.set_yticklabels(nomes)
    ax_ideogram.set_title('Ideograma do Cariótipo')
    ax_ideogram.set_xlabel('Tamanho (pb)')
    
    plt.tight_layout()
    plt.savefig('resultados_cariotipos.png')
    plt.close()

if __name__ == "__main__":
    arquivo_fasta = "exemplo_cariotipos.fasta"
    if not os.path.exists(arquivo_fasta):
        print(f"Erro: O arquivo {arquivo_fasta} não foi encontrado.")
    else:
        resultados = analisar_cariotipos(arquivo_fasta)
        visualizar_resultados(resultados)
        print("Análise concluída. Resultados salvos em 'resultados_cariotipos.png'")
        
        print("\nResultados detalhados:")
        for r in resultados:
            print(f"{r['nome']}:")
            print(f"  Tamanho: {r['tamanho']} pb")
            print(f"  Conteúdo GC: {r['gc_content']:.2f}%")
            print(f"  Conteúdo AT: {r['at_content']:.2f}%")
            print(f"  Peso molecular: {r['peso_molecular']:.2f}")
            print(f"  Número de regiões repetitivas: {len(r['regioes_repetitivas'])}")
            print(f"  Número de genes (simulado): {len(r['densidade_genes'])}")
            print(f"  Número de ilhas CpG: {len(r['ilhas_cpg'])}")
            print(f"  Complexidade da sequência: {r['complexidade']:.4f}")
            print(f"  Motivos encontrados: {r['motivos']}")
            print(f"  Potenciais estruturas secundárias: {r['estruturas_secundarias']}")
            print(f"  Índice de Adaptação de Códons: {r['codon_usage']:.4f}")
            print(f"  Número de ORFs encontradas: {len(r['orfs'])}")
            print(f"  Composição de aminoácidos (média):")
            for aa, perc in r['composicao_aminoacidos'].items():
                print(f"    {aa}: {perc:.2f}%")
            print()
