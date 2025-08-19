#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Optional, Tuple
import numpy as np
from configuracao import ConfiguracaoGeral, ConfiguracaoTreinamento

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()

class ModeloSciBERT(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.3,
        camadas_descongeladas: int = 7,
        gamma_focal_loss: float = 0.0,
        cabecas_atencao: int = 12
    ):
        super().__init__()
        config_geral = ConfiguracaoGeral()
        
        self.bert = AutoModel.from_pretrained(config_geral.modelo_base)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
        if gamma_focal_loss > 0:
            self.loss_fn = FocalLoss(gamma=gamma_focal_loss)
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        
        self.congelar_camadas(12 - camadas_descongeladas)
    
    def congelar_camadas(self, num_camadas: int):
        if num_camadas > 0:
            for i, layer in enumerate(self.bert.encoder.layer):
                if i < num_camadas:
                    for param in layer.parameters():
                        param.requires_grad = False
    
    def forward(
        self, 
        input_ids: torch.Tensor = None, 
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        result = {'logits': logits}
        
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            result['loss'] = loss
        
        return result

class TokenizadorSciBERT:
    def __init__(self, max_length: int = 512):
        config_geral = ConfiguracaoGeral()
        self.tokenizer = AutoTokenizer.from_pretrained(config_geral.modelo_base)
        self.max_length = max_length
    
    def tokenizar_textos(
        self,
        textos: List[str],
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        
        return self.tokenizer(
            textos,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors=return_tensors
        )
    
    def tokenizar_batch(
        self,
        textos: List[str],
        dispositivo: str = "cpu"
    ) -> Dict[str, torch.Tensor]:
        
        tokens = self.tokenizar_textos(textos)
        
        if dispositivo != "cpu":
            tokens = {k: v.to(dispositivo) for k, v in tokens.items()}
        
        return tokens

def criar_modelo_para_dominio(
    dominio: str,
    num_classes: int,
    hiperparametros: Dict
) -> ModeloSciBERT:
    
    modelo = ModeloSciBERT(
        num_classes=num_classes,
        dropout=hiperparametros.get('dropout', 0.3),
        camadas_descongeladas=hiperparametros.get('camadas_descongeladas', 7),
        gamma_focal_loss=hiperparametros.get('gamma_focal_loss', 0.0),
        cabecas_atencao=hiperparametros.get('cabecas_atencao', 12)
    )
    
    print(f"Modelo criado para {dominio}:")
    print(f"  Classes: {num_classes}")
    print(f"  Dropout: {hiperparametros.get('dropout', 0.3)}")
    print(f"  Camadas descongeladas: {hiperparametros.get('camadas_descongeladas', 7)}")
    print(f"  Gamma focal loss: {hiperparametros.get('gamma_focal_loss', 0.0)}")
    print(f"  Cabeças de atenção: {hiperparametros.get('cabecas_atencao', 12)}")
    print(f"  Scheduler: {hiperparametros.get('scheduler', 'cosine_warmup')}")
    
    return modelo

def calcular_parametros_treinaveis(modelo: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in modelo.parameters())
    treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    congelados = total - treinaveis
    
    return {
        'total': total,
        'treinaveis': treinaveis,
        'congelados': congelados,
        'percentual_treinavel': (treinaveis / total) * 100
    }

def salvar_modelo(modelo: ModeloSciBERT, caminho: str, metadados: Dict = None):
    checkpoint = {
        'model_state_dict': modelo.state_dict(),
        'num_classes': modelo.classifier.out_features,
        'dropout': modelo.dropout.p,
        'camadas_descongeladas': metadados.get('camadas_descongeladas', 7) if metadados else 7,
        'metadados': metadados
    }
    torch.save(checkpoint, caminho)
    print(f"Modelo salvo em: {caminho}")

def carregar_modelo(caminho: str, dispositivo: str = "cpu") -> Tuple[ModeloSciBERT, Dict]:
    checkpoint = torch.load(caminho, map_location=dispositivo)
    
    modelo = ModeloSciBERT(
        num_classes=checkpoint['num_classes'],
        dropout=checkpoint['dropout']
    )
    modelo.load_state_dict(checkpoint['model_state_dict'])
    modelo = modelo.to(dispositivo)
    
    return modelo, checkpoint.get('metadados', {})