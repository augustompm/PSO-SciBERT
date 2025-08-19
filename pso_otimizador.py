#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from typing import Dict, List, Callable, Tuple, Any
from dataclasses import dataclass
import random
from configuracao import ConfiguracaoPSO, EspacoBusca

@dataclass
class Particula:
    posicao: Dict[str, Any]
    velocidade: Dict[str, float]
    melhor_posicao: Dict[str, Any]
    melhor_fitness: float
    fitness_atual: float

class PSO:
    def __init__(
        self,
        espaco_busca: EspacoBusca,
        funcao_objetivo: Callable,
        config: ConfiguracaoPSO = None
    ):
        self.espaco_busca = espaco_busca
        self.funcao_objetivo = funcao_objetivo
        self.config = config or ConfiguracaoPSO()
        
        self.parametros = self._extrair_parametros()
        self.particulas = []
        self.melhor_global_posicao = None
        self.melhor_global_fitness = -float('inf')
        self.historico_fitness = []
        self.geracao_atual = 0
    
    def _extrair_parametros(self) -> Dict[str, List]:
        parametros = {}
        for nome, valor in vars(self.espaco_busca).items():
            if valor is not None and isinstance(valor, list):
                parametros[nome] = valor
        return parametros
    
    def _inicializar_posicao(self) -> Dict[str, Any]:
        posicao = {}
        for param, valores in self.parametros.items():
            posicao[param] = random.choice(valores)
        return posicao
    
    def _inicializar_velocidade(self) -> Dict[str, float]:
        velocidade = {}
        for param in self.parametros:
            velocidade[param] = np.random.uniform(-1, 1)
        return velocidade
    
    def inicializar_enxame(self):
        print(f"Inicializando enxame com {self.config.num_particulas} partículas...")
        
        for i in range(self.config.num_particulas):
            posicao = self._inicializar_posicao()
            velocidade = self._inicializar_velocidade()
            
            particula = Particula(
                posicao=posicao,
                velocidade=velocidade,
                melhor_posicao=posicao.copy(),
                melhor_fitness=-float('inf'),
                fitness_atual=-float('inf')
            )
            
            self.particulas.append(particula)
            print(f"  Partícula {i+1} inicializada")
    
    def _mapear_continuo_para_discreto(self, valor_continuo: float, valores_discretos: List) -> Any:
        indice = int(valor_continuo * len(valores_discretos)) % len(valores_discretos)
        return valores_discretos[indice]
    
    def _atualizar_velocidade(self, particula: Particula, inercia: float):
        r1 = np.random.random()
        r2 = np.random.random()
        
        for param in self.parametros:
            valores_discretos = self.parametros[param]
            
            pos_atual_idx = valores_discretos.index(particula.posicao[param])
            melhor_local_idx = valores_discretos.index(particula.melhor_posicao[param])
            melhor_global_idx = valores_discretos.index(self.melhor_global_posicao[param])
            
            pos_normalizada = pos_atual_idx / len(valores_discretos)
            melhor_local_norm = melhor_local_idx / len(valores_discretos)
            melhor_global_norm = melhor_global_idx / len(valores_discretos)
            
            componente_inercia = inercia * particula.velocidade[param]
            componente_cognitivo = self.config.c1 * r1 * (melhor_local_norm - pos_normalizada)
            componente_social = self.config.c2 * r2 * (melhor_global_norm - pos_normalizada)
            
            particula.velocidade[param] = (
                componente_inercia + 
                componente_cognitivo + 
                componente_social
            )
            
            particula.velocidade[param] = np.clip(particula.velocidade[param], -1, 1)
    
    def _atualizar_posicao(self, particula: Particula):
        for param in self.parametros:
            valores_discretos = self.parametros[param]
            pos_atual_idx = valores_discretos.index(particula.posicao[param])
            
            pos_continua = (pos_atual_idx / len(valores_discretos)) + particula.velocidade[param]
            pos_continua = np.clip(pos_continua, 0, 0.999)
            
            particula.posicao[param] = self._mapear_continuo_para_discreto(
                pos_continua, valores_discretos
            )
    
    def avaliar_particula(self, particula: Particula) -> float:
        fitness = self.funcao_objetivo(particula.posicao)
        particula.fitness_atual = fitness
        
        if fitness > particula.melhor_fitness:
            particula.melhor_fitness = fitness
            particula.melhor_posicao = particula.posicao.copy()
        
        if fitness > self.melhor_global_fitness:
            self.melhor_global_fitness = fitness
            self.melhor_global_posicao = particula.posicao.copy()
        
        return fitness
    
    def verificar_convergencia(self) -> bool:
        if len(self.historico_fitness) < self.config.early_stopping_paciencia:
            return False
        
        ultimas_geracoes = self.historico_fitness[-self.config.early_stopping_paciencia:]
        
        for i in range(1, len(ultimas_geracoes)):
            melhoria = ultimas_geracoes[i] - ultimas_geracoes[i-1]
            if melhoria >= self.config.early_stopping_min_melhoria:
                return False
        
        print(f"Convergência detectada após {self.geracao_atual} gerações")
        return True
    
    def otimizar(self) -> Dict[str, Any]:
        self.inicializar_enxame()
        
        print("\nIniciando otimização PSO...")
        print(f"Gerações máximas: {self.config.max_geracoes}")
        print(f"Early stopping: {self.config.early_stopping_paciencia} gerações")
        print(f"Melhoria mínima: {self.config.early_stopping_min_melhoria:.4f}")
        
        print("\nAvaliando população inicial...")
        for i, particula in enumerate(self.particulas):
            fitness = self.avaliar_particula(particula)
            print(f"  Partícula {i+1}: fitness = {fitness:.4f}")
            
        print(f"\nMelhor fitness inicial: {self.melhor_global_fitness:.4f}")
        
        for geracao in range(self.config.max_geracoes):
            self.geracao_atual = geracao + 1
            
            inercia = self.config.inercia_inicial - (
                (self.config.inercia_inicial - self.config.inercia_final) * 
                (geracao / self.config.max_geracoes)
            )
            
            print(f"\nGeração {self.geracao_atual}/{self.config.max_geracoes}")
            print(f"Inércia atual: {inercia:.3f}")
            
            fitness_geracao = []
            
            for i, particula in enumerate(self.particulas):
                self._atualizar_velocidade(particula, inercia)
                self._atualizar_posicao(particula)
                
                fitness = self.avaliar_particula(particula)
                fitness_geracao.append(fitness)
                
                print(f"  Partícula {i+1}: fitness = {fitness:.4f}")
            
            melhor_fitness_geracao = max(fitness_geracao)
            media_fitness_geracao = np.mean(fitness_geracao)
            
            self.historico_fitness.append(melhor_fitness_geracao)
            
            print(f"Melhor fitness da geração: {melhor_fitness_geracao:.4f}")
            print(f"Média fitness da geração: {media_fitness_geracao:.4f}")
            print(f"Melhor global até agora: {self.melhor_global_fitness:.4f}")
            
            if self.verificar_convergencia():
                print("Early stopping ativado - convergência atingida")
                break
            
            if (geracao + 1) % 5 == 0:
                self._salvar_checkpoint(geracao + 1)
        
        print("\nOtimização concluída!")
        print(f"Melhor fitness global: {self.melhor_global_fitness:.4f}")
        print(f"Gerações executadas: {self.geracao_atual}")
        
        return {
            'melhor_posicao': self.melhor_global_posicao,
            'melhor_fitness': self.melhor_global_fitness,
            'historico': self.historico_fitness,
            'geracoes': self.geracao_atual
        }
    
    def _salvar_checkpoint(self, geracao: int):
        checkpoint = {
            'geracao': geracao,
            'melhor_global_posicao': self.melhor_global_posicao,
            'melhor_global_fitness': self.melhor_global_fitness,
            'historico_fitness': self.historico_fitness,
            'particulas': [
                {
                    'posicao': p.posicao,
                    'melhor_posicao': p.melhor_posicao,
                    'melhor_fitness': p.melhor_fitness
                }
                for p in self.particulas
            ]
        }
        
        import pickle
        caminho = f"checkpoints/pso_checkpoint_gen_{geracao}.pkl"
        
        import os
        os.makedirs("checkpoints", exist_ok=True)
        
        with open(caminho, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"  Checkpoint salvo: {caminho}")

def criar_funcao_objetivo_scibert(
    dominio: str,
    dados_treino: Dict,
    config_treino: Dict
) -> Callable:
    
    def funcao_objetivo(hiperparametros: Dict) -> float:
        from treinar_modelo import treinar_com_validacao_cruzada
        
        print(f"\nAvaliando configuração para {dominio}:")
        for param, valor in hiperparametros.items():
            print(f"  {param}: {valor}")
        
        resultado = treinar_com_validacao_cruzada(
            dados=dados_treino,
            dominio=dominio,
            hiperparametros=hiperparametros,
            n_folds=2,
            verbose=False
        )
        
        fitness = resultado['balanced_accuracy_mean']
        print(f"  Fitness (Balanced Accuracy): {fitness:.4f}")
        
        return fitness
    
    return funcao_objetivo