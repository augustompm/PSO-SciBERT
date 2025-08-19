#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from typing import Tuple, List
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

def aplicar_random_oversampling(
    textos: List[str], 
    labels: np.ndarray, 
    seed: int = None
) -> Tuple[List[str], np.ndarray]:
    
    indices = np.arange(len(textos))
    indices_reshaped = indices.reshape(-1, 1)
    
    ros = RandomOverSampler(random_state=seed)
    indices_balanceados, labels_balanceados = ros.fit_resample(indices_reshaped, labels)
    
    indices_balanceados = indices_balanceados.flatten()
    textos_balanceados = [textos[i] for i in indices_balanceados]
    
    return textos_balanceados, labels_balanceados

def aplicar_random_undersampling(
    textos: List[str], 
    labels: np.ndarray, 
    seed: int = None
) -> Tuple[List[str], np.ndarray]:
    
    indices = np.arange(len(textos))
    indices_reshaped = indices.reshape(-1, 1)
    
    rus = RandomUnderSampler(random_state=seed)
    indices_balanceados, labels_balanceados = rus.fit_resample(indices_reshaped, labels)
    
    indices_balanceados = indices_balanceados.flatten()
    textos_balanceados = [textos[i] for i in indices_balanceados]
    
    return textos_balanceados, labels_balanceados

def aplicar_smote(
    textos: List[str], 
    labels: np.ndarray, 
    vetores_features: np.ndarray,
    seed: int = None
) -> Tuple[List[str], np.ndarray]:
    
    smote = SMOTE(random_state=seed, k_neighbors=min(5, min(np.bincount(labels)) - 1))
    
    vetores_balanceados, labels_balanceados = smote.fit_resample(vetores_features, labels)
    
    num_sinteticos = len(labels_balanceados) - len(labels)
    textos_originais = textos.copy()
    
    for i in range(num_sinteticos):
        classe_sintetica = labels_balanceados[len(labels) + i]
        indices_classe = [j for j, l in enumerate(labels) if l == classe_sintetica]
        texto_base = textos[np.random.choice(indices_classe)]
        textos_originais.append(texto_base)
    
    return textos_originais, labels_balanceados

def aplicar_estrategia_combinada(
    textos: List[str], 
    labels: np.ndarray,
    vetores_features: np.ndarray = None,
    seed: int = None
) -> Tuple[List[str], np.ndarray]:
    
    if seed is not None:
        np.random.seed(seed)
    
    indices = np.arange(len(textos))
    indices_reshaped = indices.reshape(-1, 1)
    
    # Para multi-class, usar 'auto' ou dict específico
    rus = RandomUnderSampler(sampling_strategy='auto', random_state=seed)
    indices_sub, labels_sub = rus.fit_resample(indices_reshaped, labels)
    indices_sub = indices_sub.flatten()
    textos_sub = [textos[i] for i in indices_sub]
    
    ros = RandomOverSampler(random_state=seed)
    indices_sub_reshaped = np.arange(len(textos_sub)).reshape(-1, 1)
    indices_final, labels_final = ros.fit_resample(indices_sub_reshaped, labels_sub)
    indices_final = indices_final.flatten()
    textos_final = [textos_sub[i] for i in indices_final]
    
    return textos_final, labels_final

def balancear_dataset(
    textos: List[str],
    labels: np.ndarray,
    estrategia: str = "nenhuma",
    vetores_features: np.ndarray = None,
    seed: int = None
) -> Tuple[List[str], np.ndarray]:
    
    print(f"Aplicando estratégia: {estrategia}")
    print(f"Distribuição original: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    if estrategia == "nenhuma":
        textos_balanceados, labels_balanceados = textos, labels
    
    elif estrategia == "random_oversampling":
        textos_balanceados, labels_balanceados = aplicar_random_oversampling(
            textos, labels, seed
        )
    
    elif estrategia == "random_undersampling":
        textos_balanceados, labels_balanceados = aplicar_random_undersampling(
            textos, labels, seed
        )
    
    elif estrategia == "smote":
        if vetores_features is None:
            print("SMOTE requer vetores de features, usando Random Oversampling")
            textos_balanceados, labels_balanceados = aplicar_random_oversampling(
                textos, labels, seed
            )
        else:
            textos_balanceados, labels_balanceados = aplicar_smote(
                textos, labels, vetores_features, seed
            )
    
    elif estrategia == "combinada":
        textos_balanceados, labels_balanceados = aplicar_estrategia_combinada(
            textos, labels, vetores_features, seed
        )
    
    elif estrategia == "oversampling_aleatorio":
        textos_balanceados, labels_balanceados = aplicar_random_oversampling(
            textos, labels, seed
        )
    
    else:
        raise ValueError(f"Estratégia desconhecida: {estrategia}")
    
    print(f"Distribuição balanceada: {dict(zip(*np.unique(labels_balanceados, return_counts=True)))}")
    print(f"Total amostras: {len(labels)} -> {len(labels_balanceados)}")
    
    return textos_balanceados, labels_balanceados

def verificar_proporcoes(labels_original: np.ndarray, labels_balanceado: np.ndarray) -> dict:
    prop_original = np.bincount(labels_original) / len(labels_original)
    prop_balanceado = np.bincount(labels_balanceado) / len(labels_balanceado)
    
    mudancas = {}
    for i in range(len(prop_original)):
        if i < len(prop_balanceado):
            mudanca = (prop_balanceado[i] - prop_original[i]) * 100
            mudancas[i] = {
                'original': prop_original[i],
                'balanceado': prop_balanceado[i],
                'mudanca_percentual': mudanca
            }
    
    return mudancas