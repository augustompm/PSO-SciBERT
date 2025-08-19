# PSO-SciBERT

<p align="center">
  <img src="images/logo.png" alt="PSO-SciBERT Logo">
</p>

Particle Swarm Optimization for Scientific Text Classification using SciBERT.

## Overview

Implementation of the PSO-SciBERT model for automatic classification of scientific articles on climate disasters. This repository contains the code and dataset presented in the SBPO 2025 paper.

## Dataset

The AMCLIMA-BR dataset (`data/amclima-br.csv`) contains 700 scientific articles classified across three dimensions:
- Disaster type (geological, hydrological, meteorological, climatological)
- Management phase (preparation, prevention, response, recovery)
- Machine learning paradigm (supervised, unsupervised, multi-task, reinforcement)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Train models with PSO optimization:
```bash
python main.py
```

## Project Structure

```
PSO-SciBERT/
  data/
    amclima-br.csv          # Main dataset
  images/
  main.py                   # Main execution script
  configuracao.py           # Configuration parameters
  pso_otimizador.py         # PSO implementation
  modelo_scibert.py         # SciBERT model
  treinar_modelo.py         # Training routines
  carregar_dados.py         # Data loading utilities
  balanceamento.py          # Class balancing strategies
  metricas.py               # Evaluation metrics
  requirements.txt          # Dependencies
  LICENSE                   # MIT License
  README.md                 # This file
```

## Citation

If you use this code or dataset, please cite:

```
Mendonça, A.M.P., Sousa, F.P., Coelho, I.M., Ferro, M. (2025). 
PSO-SciBERT: Otimização por Enxame de Partículas para Classificação 
Multimodal de Artigos Científicos sobre Eventos Climáticos Extremos. 
In: Simpósio Brasileiro de Pesquisa Operacional (SBPO 2025).
```

## License

MIT License. See LICENSE file for details.

## Authors

- Augusto Magalhães Pinto de Mendonça (UFF)
- Filipe Pessôa Sousa (UERJ)
- Igor Machado Coelho (UFF)
- Mariza Ferro (UFF)

## Contact

For questions about the implementation, please open an issue on GitHub.