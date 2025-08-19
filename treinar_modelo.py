#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
import numpy as np
from typing import Dict, List, Tuple
import time
from tqdm import tqdm
import gc
import random

from configuracao import ConfiguracaoGeral, ConfiguracaoTreinamento
from carregar_dados import criar_dataset_torch
from modelo_scibert import ModeloSciBERT, TokenizadorSciBERT, criar_modelo_para_dominio
from balanceamento import balancear_dataset
from metricas import calcular_metricas_completas, calcular_metricas_cv

class DatasetTextoTorch(Dataset):
    def __init__(self, textos: List[str], labels: np.ndarray, tokenizador: TokenizadorSciBERT):
        self.textos = textos
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.tokenizador = tokenizador
    
    def __len__(self):
        return len(self.textos)
    
    def __getitem__(self, idx):
        return self.textos[idx], self.labels[idx]

def collate_fn(batch, tokenizador, dispositivo):
    textos, labels = zip(*batch)
    tokens = tokenizador.tokenizar_batch(list(textos), dispositivo)
    labels = torch.stack(list(labels)).to(dispositivo)
    return tokens, labels

def treinar_epoca(
    modelo: ModeloSciBERT,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler,
    dispositivo: str,
    gradient_accumulation_steps: int = 1,
    mixed_precision: bool = True
) -> float:
    
    modelo.train()
    total_loss = 0
    num_batches = 0
    
    if mixed_precision and dispositivo != "cpu":
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
    
    for i, (tokens, labels) in enumerate(tqdm(dataloader, desc="Treinando")):
        if mixed_precision and dispositivo != "cpu":
            with autocast():
                outputs = modelo(**tokens, labels=labels)
                loss = outputs['loss'] / gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
        else:
            outputs = modelo(**tokens, labels=labels)
            loss = outputs['loss'] / gradient_accumulation_steps
            loss.backward()
            
            if (i + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
    
    return total_loss / num_batches

def avaliar_modelo(
    modelo: ModeloSciBERT,
    dataloader: DataLoader,
    dispositivo: str
) -> Tuple[np.ndarray, np.ndarray]:
    
    modelo.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for tokens, labels in tqdm(dataloader, desc="Avaliando"):
            outputs = modelo(**tokens)
            predictions = torch.argmax(outputs['logits'], dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_labels), np.array(all_predictions)

def treinar_modelo_completo(
    textos_treino: List[str],
    labels_treino: np.ndarray,
    textos_val: List[str],
    labels_val: np.ndarray,
    dominio: str,
    hiperparametros: Dict,
    config_treino: ConfiguracaoTreinamento = None
) -> Dict:
    
    config_geral = ConfiguracaoGeral()
    config_treino = config_treino or ConfiguracaoTreinamento()
    
    dispositivo = torch.device(config_geral.dispositivo)
    print(f"Dispositivo: {dispositivo}")
    
    textos_balanceados, labels_balanceados = balancear_dataset(
        textos_treino,
        labels_treino,
        estrategia=hiperparametros.get('estrategia_balanceamento', 'nenhuma'),
        seed=config_geral.seed
    )
    
    num_classes = len(np.unique(labels_treino))
    modelo = criar_modelo_para_dominio(dominio, num_classes, hiperparametros)
    modelo = modelo.to(dispositivo)
    
    tokenizador = TokenizadorSciBERT(
        max_length=hiperparametros.get('max_seq_length', config_treino.max_seq_length)
    )
    
    dataset_treino = DatasetTextoTorch(textos_balanceados, labels_balanceados, tokenizador)
    dataset_val = DatasetTextoTorch(textos_val, labels_val, tokenizador)
    
    dataloader_treino = DataLoader(
        dataset_treino,
        batch_size=config_treino.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizador, dispositivo)
    )
    
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=config_treino.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizador, dispositivo)
    )
    
    optimizer = optim.AdamW(
        modelo.parameters(),
        lr=hiperparametros.get('taxa_aprendizado', 2e-5),
        weight_decay=hiperparametros.get('weight_decay', 0.01)
    )
    
    num_training_steps = len(dataloader_treino) * config_treino.num_epochs
    tipo_scheduler = hiperparametros.get('scheduler', 'cosine_warmup')
    
    if tipo_scheduler == 'cosine_warmup':
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_training_steps * 0.1),
            num_training_steps=num_training_steps
        )
    elif tipo_scheduler == 'one_cycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=hiperparametros.get('taxa_aprendizado', 2e-5),
            total_steps=num_training_steps
        )
    elif tipo_scheduler == 'cyclic':
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=hiperparametros.get('taxa_aprendizado', 2e-5) / 10,
            max_lr=hiperparametros.get('taxa_aprendizado', 2e-5),
            step_size_up=num_training_steps // 4
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_training_steps * 0.1),
            num_training_steps=num_training_steps
        )
    
    print(f"\nTreinando modelo para {dominio}")
    print(f"Épocas: {config_treino.num_epochs}")
    print(f"Amostras treino: {len(dataset_treino)}")
    print(f"Amostras validação: {len(dataset_val)}")
    
    melhor_bal_acc = 0
    melhor_epoca = 0
    
    for epoca in range(config_treino.num_epochs):
        print(f"\nÉpoca {epoca + 1}/{config_treino.num_epochs}")
        
        loss_medio = treinar_epoca(
            modelo,
            dataloader_treino,
            optimizer,
            scheduler,
            dispositivo,
            config_treino.gradient_accumulation_steps,
            config_treino.mixed_precision
        )
        
        print(f"Loss médio: {loss_medio:.4f}")
        
        y_true, y_pred = avaliar_modelo(modelo, dataloader_val, dispositivo)
        metricas = calcular_metricas_completas(y_true, y_pred)
        
        print(f"Balanced Accuracy: {metricas['balanced_accuracy']:.4f}")
        print(f"F1 Macro: {metricas['f1_macro']:.4f}")
        
        if metricas['balanced_accuracy'] > melhor_bal_acc:
            melhor_bal_acc = metricas['balanced_accuracy']
            melhor_epoca = epoca + 1
    
    print(f"\nMelhor resultado: {melhor_bal_acc:.4f} na época {melhor_epoca}")
    
    y_true_final, y_pred_final = avaliar_modelo(modelo, dataloader_val, dispositivo)
    metricas_finais = calcular_metricas_completas(y_true_final, y_pred_final)
    
    del modelo
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return metricas_finais

def treinar_com_validacao_cruzada(
    dados: Dict,
    dominio: str,
    hiperparametros: Dict,
    n_folds: int = 5,
    verbose: bool = True
) -> Dict:
    
    config_geral = ConfiguracaoGeral()
    
    textos = dados['textos']
    labels = dados['labels'][dominio]
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config_geral.seed)
    
    resultados_folds = []
    
    for fold, (treino_idx, val_idx) in enumerate(skf.split(textos, labels)):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Fold {fold + 1}/{n_folds}")
            print(f"{'='*50}")
        
        textos_treino = [textos[i] for i in treino_idx]
        labels_treino = labels[treino_idx]
        
        textos_val = [textos[i] for i in val_idx]
        labels_val = labels[val_idx]
        
        resultado_fold = treinar_modelo_completo(
            textos_treino,
            labels_treino,
            textos_val,
            labels_val,
            dominio,
            hiperparametros
        )
        
        resultados_folds.append(resultado_fold)
        
        if not verbose:
            break
    
    metricas_agregadas = calcular_metricas_cv(resultados_folds)
    
    if verbose:
        print(f"\n{'='*50}")
        print("Resultados Finais - Validação Cruzada")
        print(f"{'='*50}")
        print(f"Balanced Accuracy: {metricas_agregadas['balanced_accuracy_mean']:.4f} ± {metricas_agregadas['balanced_accuracy_std']:.4f}")
        print(f"F1 Macro: {metricas_agregadas['f1_macro_mean']:.4f} ± {metricas_agregadas['f1_macro_std']:.4f}")
        print(f"MCC: {metricas_agregadas['mcc_mean']:.4f} ± {metricas_agregadas['mcc_std']:.4f}")
    
    return metricas_agregadas