#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)
from typing import Dict, List, Tuple
import pandas as pd

def calcular_metricas_completas(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels_names: Dict[int, str] = None
) -> Dict:
    
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f2_macro = fbeta_score(y_true, y_pred, beta=2, average='macro')
    mcc = matthews_corrcoef(y_true, y_pred)
    
    matriz_confusao = confusion_matrix(y_true, y_pred)
    matriz_normalizada = matriz_confusao.astype('float') / matriz_confusao.sum(axis=1)[:, np.newaxis]
    
    metricas = {
        'balanced_accuracy': bal_acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f2_macro': f2_macro,
        'mcc': mcc,
        'matriz_confusao': matriz_confusao,
        'matriz_confusao_normalizada': matriz_normalizada
    }
    
    if labels_names:
        report = classification_report(
            y_true, y_pred,
            target_names=[labels_names[i] for i in sorted(labels_names.keys())],
            output_dict=True
        )
        metricas['classification_report'] = report
    
    return metricas

def calcular_metricas_cv(resultados_folds: List[Dict]) -> Dict:
    metricas_agregadas = {}
    
    metricas_numericas = [
        'balanced_accuracy', 'f1_macro', 'f1_weighted', 'f2_macro', 'mcc'
    ]
    
    for metrica in metricas_numericas:
        valores = [fold[metrica] for fold in resultados_folds]
        metricas_agregadas[f'{metrica}_mean'] = np.mean(valores)
        metricas_agregadas[f'{metrica}_std'] = np.std(valores)
    
    num_classes = resultados_folds[0]['matriz_confusao'].shape[0]
    matriz_media = np.zeros((num_classes, num_classes))
    
    for fold in resultados_folds:
        matriz_media += fold['matriz_confusao_normalizada']
    
    matriz_media /= len(resultados_folds)
    metricas_agregadas['matriz_confusao_media'] = matriz_media
    
    return metricas_agregadas

def comparar_baseline_pso(
    metricas_baseline: Dict,
    metricas_pso: Dict,
    dominio: str
) -> Dict:
    
    print(f"\nComparação {dominio}:")
    print("=" * 50)
    
    comparacao = {}
    
    metricas_principais = ['balanced_accuracy', 'f1_macro', 'mcc']
    
    for metrica in metricas_principais:
        baseline_mean = metricas_baseline.get(f'{metrica}_mean', 0)
        pso_mean = metricas_pso.get(f'{metrica}_mean', 0)
        
        melhoria_absoluta = pso_mean - baseline_mean
        melhoria_percentual = (melhoria_absoluta / baseline_mean) * 100 if baseline_mean > 0 else 0
        
        comparacao[metrica] = {
            'baseline': baseline_mean,
            'pso': pso_mean,
            'melhoria_absoluta': melhoria_absoluta,
            'melhoria_percentual': melhoria_percentual
        }
        
        print(f"\n{metrica.upper()}:")
        print(f"  Baseline: {baseline_mean:.4f}")
        print(f"  PSO-SciBERT: {pso_mean:.4f}")
        print(f"  Melhoria: {melhoria_absoluta:+.4f} ({melhoria_percentual:+.1f}%)")
    
    baseline_std = metricas_baseline.get('balanced_accuracy_std', 0)
    pso_std = metricas_pso.get('balanced_accuracy_std', 0)
    reducao_variabilidade = ((baseline_std - pso_std) / baseline_std) * 100 if baseline_std > 0 else 0
    
    print(f"\nVARIABILIDADE (Desvio Padrão):")
    print(f"  Baseline: {baseline_std:.4f}")
    print(f"  PSO-SciBERT: {pso_std:.4f}")
    print(f"  Redução: {reducao_variabilidade:.1f}%")
    
    comparacao['variabilidade'] = {
        'baseline_std': baseline_std,
        'pso_std': pso_std,
        'reducao_percentual': reducao_variabilidade
    }
    
    return comparacao

def gerar_tabela_resultados(
    resultados_todos_dominios: Dict
) -> pd.DataFrame:
    
    dados_tabela = []
    
    for dominio, resultado in resultados_todos_dominios.items():
        linha = {
            'Domínio': dominio.replace('_', ' ').title(),
            'Bal. Acc. Baseline': f"{resultado['baseline']['balanced_accuracy_mean']:.3f}",
            'Bal. Acc. PSO': f"{resultado['pso']['balanced_accuracy_mean']:.3f}",
            'Melhoria (%)': f"{resultado['comparacao']['balanced_accuracy']['melhoria_percentual']:.1f}",
            'F1 Baseline': f"{resultado['baseline']['f1_macro_mean']:.3f}",
            'F1 PSO': f"{resultado['pso']['f1_macro_mean']:.3f}",
            'MCC Baseline': f"{resultado['baseline'].get('mcc_mean', 0):.3f}",
            'MCC PSO': f"{resultado['pso'].get('mcc_mean', 0):.3f}"
        }
        dados_tabela.append(linha)
    
    df = pd.DataFrame(dados_tabela)
    return df

def exibir_matriz_confusao(
    matriz: np.ndarray,
    labels_names: Dict[int, str] = None,
    titulo: str = "Matriz de Confusão"
):
    print(f"\n{titulo}")
    print("=" * 50)
    
    if labels_names:
        nomes = [labels_names[i] for i in sorted(labels_names.keys())]
        max_len = max(len(nome) for nome in nomes)
        
        print(" " * (max_len + 2), end="")
        for nome in nomes:
            print(f"{nome[:10]:>12}", end="")
        print()
        
        for i, nome in enumerate(nomes):
            print(f"{nome[:max_len]:{max_len}} ", end="")
            for j in range(len(nomes)):
                valor = matriz[i, j]
                print(f"{valor:>12.3f}", end="")
            print()
    else:
        print(matriz)

def calcular_tempo_execucao(
    inicio: float,
    fim: float,
    num_amostras: int = None
) -> Dict:
    
    tempo_total = fim - inicio
    horas = int(tempo_total // 3600)
    minutos = int((tempo_total % 3600) // 60)
    segundos = tempo_total % 60
    
    resultado = {
        'tempo_total_segundos': tempo_total,
        'tempo_formatado': f"{horas}h {minutos}m {segundos:.1f}s"
    }
    
    if num_amostras:
        resultado['tempo_por_amostra'] = tempo_total / num_amostras
    
    return resultado