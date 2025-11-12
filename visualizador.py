"""
Módulo para visualização de métricas de treinamento.
"""
import matplotlib
matplotlib.use('Agg')  # Usar backend não-interativo por padrão
import matplotlib.pyplot as plt
import os


def plotar_curvas_treinamento(historico, salvar_arquivo='curvas_treinamento.png'):
    """
    Plota as curvas de loss e acurácia de treino e validação.
    
    Args:
        historico: Dicionário com as métricas de treinamento
        salvar_arquivo: Nome do arquivo para salvar o gráfico
    """
    if not historico:
        print("Erro: Histórico vazio. Não é possível gerar gráficos.")
        return
    
    epochs = range(1, len(historico['treino_loss']) + 1)
    
    # Criar figura com 2 subplots lado a lado
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico 1: Loss (Perda)
    ax1.plot(epochs, historico['treino_loss'], 'b-', label='Treino', linewidth=2)
    ax1.plot(epochs, historico['validacao_loss'], 'r-', label='Validação', linewidth=2)
    ax1.set_title('Curva de Loss (Perda)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Época', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=1)
    
    # Gráfico 2: Acurácia
    ax2.plot(epochs, historico['treino_acc'], 'b-', label='Treino', linewidth=2)
    ax2.plot(epochs, historico['validacao_acc'], 'r-', label='Validação', linewidth=2)
    ax2.set_title('Curva de Acurácia', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Época', fontsize=12)
    ax2.set_ylabel('Acurácia (%)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=1)
    ax2.set_ylim(bottom=0, top=100)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Salvar gráfico
    plt.savefig(salvar_arquivo, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico salvo em '{salvar_arquivo}'")
    
    # Fechar figura para liberar memória
    plt.close()


def plotar_curvas_combinadas(historico, salvar_arquivo='curvas_treinamento_combinadas.png'):
    """
    Plota as curvas de loss e acurácia em um único gráfico com 2 eixos Y.
    
    Args:
        historico: Dicionário com as métricas de treinamento
        salvar_arquivo: Nome do arquivo para salvar o gráfico
    """
    if not historico:
        print("Erro: Histórico vazio. Não é possível gerar gráficos.")
        return
    
    epochs = range(1, len(historico['treino_loss']) + 1)
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Eixo Y esquerdo: Loss
    color = 'tab:blue'
    ax1.set_xlabel('Época', fontsize=12)
    ax1.set_ylabel('Loss', color=color, fontsize=12)
    linha1 = ax1.plot(epochs, historico['treino_loss'], 'b-', label='Loss Treino', linewidth=2, alpha=0.7)
    linha2 = ax1.plot(epochs, historico['validacao_loss'], 'b--', label='Loss Validação', linewidth=2, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=1)
    
    # Eixo Y direito: Acurácia
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Acurácia (%)', color=color, fontsize=12)
    linha3 = ax2.plot(epochs, historico['treino_acc'], 'r-', label='Acc Treino', linewidth=2, alpha=0.7)
    linha4 = ax2.plot(epochs, historico['validacao_acc'], 'r--', label='Acc Validação', linewidth=2, alpha=0.7)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(bottom=0, top=100)
    
    # Título e legenda
    plt.title('Curvas de Treinamento - Loss e Acurácia', fontsize=14, fontweight='bold', pad=20)
    
    # Combinar legendas
    linhas = linha1 + linha2 + linha3 + linha4
    labels = [l.get_label() for l in linhas]
    ax1.legend(linhas, labels, loc='center right', fontsize=10)
    
    plt.tight_layout()
    
    # Salvar gráfico
    plt.savefig(salvar_arquivo, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico combinado salvo em '{salvar_arquivo}'")
    
    # Fechar figura para liberar memória
    plt.close()

