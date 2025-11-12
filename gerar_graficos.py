"""
Script para gerar gráficos de treinamento a partir de um histórico salvo em JSON.
Útil para visualizar resultados de treinamentos anteriores sem precisar treinar novamente.
"""
import json
import sys
from visualizador import plotar_curvas_treinamento, plotar_curvas_combinadas


def main():
    """
    Carrega histórico de treinamento e gera gráficos.
    """
    arquivo_historico = 'historico_treinamento.json'
    
    if len(sys.argv) > 1:
        arquivo_historico = sys.argv[1]
    
    try:
        print(f"Carregando histórico de '{arquivo_historico}'...")
        with open(arquivo_historico, 'r', encoding='utf-8') as f:
            historico = json.load(f)
        
        print("✓ Histórico carregado com sucesso!")
        print(f"  - Épocas: {len(historico['treino_loss'])}")
        if 'melhor_acc_validacao' in historico:
            print(f"  - Melhor acurácia de validação: {historico['melhor_acc_validacao']:.2f}%")
        
        print("\nGerando gráficos...")
        plotar_curvas_treinamento(historico, 'curvas_treinamento.png')
        plotar_curvas_combinadas(historico, 'curvas_treinamento_combinadas.png')
        
        print("\n✓ Gráficos gerados com sucesso!")
        print("  - curvas_treinamento.png")
        print("  - curvas_treinamento_combinadas.png")
        
    except FileNotFoundError:
        print(f"Erro: Arquivo '{arquivo_historico}' não encontrado!")
        print("\nPara gerar gráficos, você precisa:")
        print("  1. Treinar o modelo primeiro (python main_crops.py)")
        print("  2. Ou fornecer o caminho para um arquivo JSON com histórico")
        print("\nUso: python gerar_graficos.py [caminho_historico.json]")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Erro: Arquivo '{arquivo_historico}' não é um JSON válido!")
        sys.exit(1)
    except Exception as e:
        print(f"Erro ao gerar gráficos: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

