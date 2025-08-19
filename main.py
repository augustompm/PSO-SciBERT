#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import torch
import numpy as np
import random
from pathlib import Path
from typing import Dict

from configuracao import (
    ConfiguracaoGeral, 
    ConfiguracaoPSO, 
    ConfiguracaoTreinamento,
    EspacoBusca,
    HiperparametrosOtimizados
)
import sys
sys.path.append('.')

import carregar_dados
import pso_otimizador
import treinar_modelo
import metricas
import json
import argparse

def configurar_ambiente():
    # Configurações para melhor performance
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

def executar_baseline(dados: Dict, dominio: str) -> Dict:
    print(f"\n{'='*60}")
    print(f"BASELINE - {dominio.upper()}")
    print(f"{'='*60}")
    
    hiperparametros_baseline = {
        'taxa_aprendizado': 2e-5,
        'dropout': 0.3,
        'num_camadas_congelar': 0,
        'estrategia_balanceamento': 'nenhuma',
        'gamma_focal_loss': 0.0,
        'warmup_steps': 0.1,
        'weight_decay': 0.01,
        'max_seq_length': 512
    }
    
    resultado = treinar_modelo.treinar_com_validacao_cruzada(
        dados=dados,
        dominio=dominio,
        hiperparametros=hiperparametros_baseline,
        n_folds=5,
        verbose=True
    )
    
    return resultado

def executar_pso_otimizacao(dados: Dict, dominio: str) -> Dict:
    print(f"\n{'='*60}")
    print(f"PSO OTIMIZAÇÃO - {dominio.upper()}")
    print(f"{'='*60}")
    
    config_pso = ConfiguracaoPSO()
    espaco_busca = EspacoBusca()
    
    funcao_objetivo = pso_otimizador.criar_funcao_objetivo_scibert(
        dominio=dominio,
        dados_treino=dados,
        config_treino={}
    )
    
    pso = pso_otimizador.PSO(
        espaco_busca=espaco_busca,
        funcao_objetivo=funcao_objetivo,
        config=config_pso
    )
    
    resultado_pso = pso.otimizar()
    
    print(f"\nMelhores hiperparâmetros encontrados:")
    for param, valor in resultado_pso['melhor_posicao'].items():
        print(f"  {param}: {valor}")
    
    print(f"\nTreinando modelo final com melhores hiperparâmetros...")
    
    resultado_final = treinar_modelo.treinar_com_validacao_cruzada(
        dados=dados,
        dominio=dominio,
        hiperparametros=resultado_pso['melhor_posicao'],
        n_folds=5,
        verbose=True
    )
    
    return resultado_final, resultado_pso['melhor_posicao']

def carregar_hiperparametros_padrao(dominio: str) -> Dict:
    """Load default optimized hyperparameters for a given domain."""
    config_hiper = HiperparametrosOtimizados()
    
    if dominio == 'tipo_desastre':
        return config_hiper.tipo_desastre
    elif dominio == 'fase_gestao':
        return config_hiper.fase_gestao
    elif dominio == 'paradigma_ml':
        return config_hiper.paradigma_ml
    else:
        raise ValueError(f"Domínio não reconhecido: {dominio}")

def main():
    parser = argparse.ArgumentParser(description='PSO-SciBERT')
    parser.add_argument('--no-pso', action='store_true', help='Use default optimized hyperparameters without PSO optimization')
    args = parser.parse_args()
    
    print("="*60)
    print("PSO-SciBERT - Sistema de Classificação")
    print(f"Modo: {'Direto (sem PSO)' if args.no_pso else 'Com otimização PSO'}")
    print("="*60)
    
    configurar_ambiente()
    
    config_geral = ConfiguracaoGeral()
    
    print(f"\nDispositivo: {config_geral.dispositivo}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memória GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print("\nCarregando dados AMCLIMA-BR...")
    dados = carregar_dados.preparar_dados_completos()
    
    dominios = ['tipo_desastre', 'fase_gestao', 'paradigma_ml']
    
    resultados_todos = {}
    tempo_inicio_total = time.time()
    
    for dominio in dominios:
        print(f"\n{'#'*60}")
        print(f"PROCESSANDO DOMÍNIO: {dominio.upper()}")
        print(f"{'#'*60}")
        
        tempo_inicio_dominio = time.time()
        
        print("\n1. Executando Baseline...")
        resultado_baseline = executar_baseline(dados, dominio)
        
        if args.no_pso:
            print("\n2. Carregando hiperparâmetros padrão otimizados...")
            hiperparametros_otimos = carregar_hiperparametros_padrao(dominio)
            print("\nHiperparâmetros carregados:")
            for param, valor in hiperparametros_otimos.items():
                print(f"  {param}: {valor}")
            
            print("\n3. Treinando modelo com hiperparâmetros otimizados...")
            resultado_pso = treinar_modelo.treinar_com_validacao_cruzada(
                dados=dados,
                dominio=dominio,
                hiperparametros=hiperparametros_otimos,
                n_folds=5,
                verbose=True
            )
        else:
            print("\n2. Executando PSO-SciBERT...")
            resultado_pso, hiperparametros_otimos = executar_pso_otimizacao(dados, dominio)
        
        comparacao = metricas.comparar_baseline_pso(
            resultado_baseline,
            resultado_pso,
            dominio
        )
        
        tempo_fim_dominio = time.time()
        tempo_dominio = (tempo_fim_dominio - tempo_inicio_dominio) / 3600
        print(f"\nTempo total para {dominio}: {tempo_dominio:.2f} horas")
        
        resultados_todos[dominio] = {
            'baseline': resultado_baseline,
            'pso': resultado_pso,
            'comparacao': comparacao,
            'hiperparametros_otimos': hiperparametros_otimos,
            'tempo_horas': tempo_dominio
        }
        
    
    tempo_fim_total = time.time()
    tempo_total = (tempo_fim_total - tempo_inicio_total) / 3600
    
    print(f"\n{'='*60}")
    print("RESUMO FINAL")
    print(f"{'='*60}")
    
    print("\nResultados por Domínio:")
    for dominio, resultado in resultados_todos.items():
        print(f"\n{dominio.upper()}:")
        print(f"  Baseline: {resultado['baseline']['balanced_accuracy_mean']:.4f}")
        print(f"  PSO-SciBERT: {resultado['pso']['balanced_accuracy_mean']:.4f}")
        print(f"  Melhoria: {resultado['comparacao']['balanced_accuracy']['melhoria_percentual']:.1f}%")
        print(f"  Tempo: {resultado['tempo_horas']:.2f} horas")
    
    print(f"\nTempo total de execução: {tempo_total:.2f} horas")
    
    print("\nSalvando resultados...")
    import pickle
    Path("resultados").mkdir(exist_ok=True)
    
    with open("resultados/resultados_completos.pkl", "wb") as f:
        pickle.dump(resultados_todos, f)
    
    tabela = metricas.gerar_tabela_resultados(resultados_todos)
    tabela.to_csv("resultados/tabela_resultados.csv", index=False)
    
    print("\nResultados salvos em:")
    print("  - resultados/resultados_completos.pkl")
    print("  - resultados/tabela_resultados.csv")
    
    print("\nExecução concluída com sucesso!")

if __name__ == "__main__":
    main()