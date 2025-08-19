#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List, Dict, Tuple
import torch

@dataclass
class ConfiguracaoGeral:
    seed: int = 42
    dispositivo: str = "cuda" if torch.cuda.is_available() else "cpu"
    caminho_dados: str = "data/amclima-br.csv"
    caminho_checkpoints: str = "checkpoints/"
    caminho_resultados: str = "resultados/"
    modelo_base: str = "allenai/scibert_scivocab_uncased"

@dataclass
class ConfiguracaoPSO:
    num_particulas: int = 12
    max_geracoes: int = 30
    inercia_inicial: float = 0.9
    inercia_final: float = 0.4
    c1: float = 2.0
    c2: float = 2.0
    early_stopping_paciencia: int = 5
    early_stopping_min_melhoria: float = 0.0025
    num_folds_pso: int = 2

@dataclass
class EspacoBusca:
    taxa_aprendizado: List[float] = None
    dropout: List[float] = None
    camadas_descongeladas: List[int] = None
    estrategia_balanceamento: List[str] = None
    gamma_focal_loss: List[float] = None
    weight_decay: List[float] = None
    cabecas_atencao: List[int] = None
    scheduler: List[str] = None
    
    def __post_init__(self):
        self.taxa_aprendizado = [2e-5, 2.5e-5, 3e-5, 4e-5, 5e-5]
        self.dropout = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
        self.camadas_descongeladas = [4, 5, 6, 7]
        self.estrategia_balanceamento = ["nenhuma", "random_oversampling", "smote", "combinada"]
        self.gamma_focal_loss = [1.5, 2.0, 2.5]
        self.weight_decay = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4]
        self.cabecas_atencao = [4, 8, 12]
        self.scheduler = ["cosine_warmup", "one_cycle", "cyclic"]

@dataclass
class ConfiguracaoTreinamento:
    batch_size: int = 12
    gradient_accumulation_steps: int = 2
    num_epochs: int = 10
    mixed_precision: bool = True
    num_folds_avaliacao: int = 5
    max_seq_length: int = 512
    
@dataclass
class HiperparametrosOtimizados:
    """Hyperparameters found through PSO optimization for different domains."""
    tipo_desastre: Dict = None
    fase_gestao: Dict = None
    paradigma_ml: Dict = None
    
    def __post_init__(self):
        # Default optimized values - users can modify or find their own via PSO
        self.tipo_desastre = {
            "taxa_aprendizado": 2.5e-5,
            "dropout": 0.35,
            "camadas_descongeladas": 4,
            "estrategia_balanceamento": "combinada",
            "gamma_focal_loss": 2.0,
            "weight_decay": 5e-5,
            "scheduler": "one_cycle",
            "cabecas_atencao": 12
        }
        
        self.fase_gestao = {
            "taxa_aprendizado": 4e-5,
            "dropout": 0.30,
            "camadas_descongeladas": 7,
            "estrategia_balanceamento": "random_oversampling",
            "gamma_focal_loss": 1.5,
            "weight_decay": 2e-5,
            "scheduler": "cyclic",
            "cabecas_atencao": 4
        }
        
        self.paradigma_ml = {
            "taxa_aprendizado": 3e-5,
            "dropout": 0.40,
            "camadas_descongeladas": 6,
            "estrategia_balanceamento": "random_oversampling",
            "gamma_focal_loss": 2.5,
            "weight_decay": 1e-4,
            "scheduler": "cosine_warmup",
            "cabecas_atencao": 8
        }