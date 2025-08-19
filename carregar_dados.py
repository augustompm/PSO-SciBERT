#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import torch
from configuracao import ConfiguracaoGeral

def carregar_dataset_amclima() -> pd.DataFrame:
    config = ConfiguracaoGeral()
    caminho = Path(config.caminho_dados)
    
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {caminho}")
    
    dados = pd.read_csv(caminho, sep=';', encoding='utf-8-sig')
    
    if len(dados) != 700:
        raise ValueError(f"Esperado 700 artigos, encontrado {len(dados)}")
    
    return dados

def preparar_textos(dados: pd.DataFrame) -> List[str]:
    if 'combined_text' not in dados.columns:
        raise ValueError("Coluna 'combined_text' não encontrada")
    
    textos = dados['combined_text'].tolist()
    
    for texto in textos:
        if pd.isna(texto) or len(texto) < 50:
            raise ValueError("Texto inválido ou muito curto encontrado")
    
    return textos

def preparar_labels(dados: pd.DataFrame, coluna_label: str) -> Tuple[np.ndarray, Dict[int, str]]:
    if coluna_label not in dados.columns:
        raise ValueError(f"Coluna {coluna_label} não encontrada")
    
    labels = dados[coluna_label].values
    
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels.astype(str))
    
    label_para_nome = {i: str(classe) for i, classe in enumerate(encoder.classes_)}
    
    return labels_encoded, label_para_nome

def criar_divisao_estratificada(
    textos: List[str], 
    labels: np.ndarray, 
    n_splits: int = 5, 
    seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    divisoes = []
    for treino_idx, val_idx in skf.split(textos, labels):
        divisoes.append((treino_idx, val_idx))
    
    return divisoes

def verificar_distribuicao_classes(labels: np.ndarray, nome_dominio: str) -> Dict[int, int]:
    unique, counts = np.unique(labels, return_counts=True)
    distribuicao = dict(zip(unique, counts))
    
    print(f"\nDistribuição {nome_dominio}:")
    total = len(labels)
    for classe, quantidade in sorted(distribuicao.items()):
        percentual = (quantidade / total) * 100
        print(f"  Classe {classe}: {quantidade} ({percentual:.1f}%)")
    
    return distribuicao

def preparar_dados_completos() -> Dict:
    dados = carregar_dataset_amclima()
    textos = preparar_textos(dados)
    
    labels_tipo, mapa_tipo = preparar_labels(dados, 'disaster_type_id')
    labels_fase, mapa_fase = preparar_labels(dados, 'management_phase_id')
    labels_paradigma, mapa_paradigma = preparar_labels(dados, 'ml_paradigm_id')
    
    verificar_distribuicao_classes(labels_tipo, "Tipo de Desastre")
    verificar_distribuicao_classes(labels_fase, "Fase de Gestão")
    verificar_distribuicao_classes(labels_paradigma, "Paradigma ML")
    
    return {
        'textos': textos,
        'labels': {
            'tipo_desastre': labels_tipo,
            'fase_gestao': labels_fase,
            'paradigma_ml': labels_paradigma
        },
        'mapas': {
            'tipo_desastre': mapa_tipo,
            'fase_gestao': mapa_fase,
            'paradigma_ml': mapa_paradigma
        },
        'dados_originais': dados
    }

def criar_dataset_torch(textos: List[str], labels: np.ndarray) -> torch.utils.data.Dataset:
    class DatasetTexto(torch.utils.data.Dataset):
        def __init__(self, textos, labels):
            self.textos = textos
            self.labels = labels
        
        def __len__(self):
            return len(self.textos)
        
        def __getitem__(self, idx):
            return {
                'texto': self.textos[idx],
                'label': self.labels[idx]
            }
    
    return DatasetTexto(textos, labels)

if __name__ == "__main__":
    print("Carregando dados AMCLIMA-BR...")
    dados_completos = preparar_dados_completos()
    
    print(f"\nTotal de artigos: {len(dados_completos['textos'])}")
    print(f"Dimensões classificadas: {list(dados_completos['labels'].keys())}")
    
    for dominio in dados_completos['labels']:
        num_classes = len(np.unique(dados_completos['labels'][dominio]))
        print(f"{dominio}: {num_classes} classes")