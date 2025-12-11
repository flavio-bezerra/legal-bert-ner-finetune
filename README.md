# ⚖️ Legal Risk Extractor: Auditoria Contratual com Legal-BERT & MLOps

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Hugging Face](https://img.shields.io/badge/🤗%20Transformers-Legal--BERT-yellow)
![MLflow](https://img.shields.io/badge/MLflow-Managed-blue)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-orange)

> **Auditoria automatizada de riscos em contratos de alta volumetria usando NLP Especializado (NER), com custo 100x menor que LLMs Generativos.**

---

## 🎯 Visão Executiva & Contexto de Negócio

Em cenários de M&A (Fusões e Aquisições) ou Auditoria de Compliance, empresas precisam revisar milhares de contratos legados para identificar riscos ocultos, como **Cláusulas de Multa**, **Foro de Eleição** ou **Datas de Término Automático**.

**Por que não usar GPT-4?**
Embora LLMs sejam poderosos, processar 1 milhão de páginas via API gera:

1.  **Custo Proibitivo:** Alto custo por token, tornando inviável para varreduras massivas.
2.  **Latência:** APIs generativas são lentas; este modelo roda em milissegundos localmente.
3.  **Privacidade/Compliance:** Envio de dados confidenciais para APIs externas pode violar NDAs rigorosos.

**A Solução:** Este projeto implementa um modelo **Legal-BERT (110M parâmetros)** fine-tuned, otimizado para extração de entidades jurídicas (NER) em contratos longos, oferecendo velocidade, privacidade e custo marginal zero.

---

## 🏗️ Arquitetura Técnica

### 1. Superando o Limite de 512 Tokens (Robust Sliding Windows)

Contratos reais excedem o limite de contexto do BERT.

- **Estratégia:** Implementação de Janelas Deslizantes (`stride=128`, `max_length=512`).
- **Agregação:** Um pipeline de pós-processamento reconstrói as entidades que foram "cortadas" na divisa das janelas, garantindo integridade semântica.

### 2. Otimização de Hiperparâmetros (Bayesian Search)

Em vez de "chutar" learning rates, utilizei **Optuna** para realizar uma busca Bayesiana, maximizando o F1-Score no conjunto de validação e encontrando a convergência ideal para o dataset jurídico.

### 3. Engenharia de Dados Robusta & Correção de Viés

- **Prevenção de Data Leakage:** Split de dados realizado por _Document ID_, garantindo que trechos do mesmo contrato nunca apareçam simultaneamente no treino e teste.
- **Tratamento de Labels:** Inicialização explícita de tokens "Outside" ('O') e **Negative Downsampling** para lidar com o desbalanceamento extremo (99% do texto jurídico não é entidade de interesse).

### 4. Bônus: Quantização (INT8)

O modelo final inclui uma etapa de **Quantização Dinâmica**, reduzindo o tamanho do modelo em 4x e acelerando a inferência na CPU, ideal para deploy em ambientes com recursos limitados (Edge/Serverless).

---

## ⚙️ MLOps Pipeline

O ciclo de vida do modelo foi gerenciado utilizando **MLflow**:

- **Tracking:** Log automático de métricas (Loss, Precision, Recall, F1).
- **Artifacts:** Salvamento auditável de gráficos de diagnóstico (Curvas de Aprendizado, Matriz de Confusão).
- **Model Registry:** Versionamento de modelos com status (Staging -> Production).

---

## 📊 Resultados

O modelo final atingiu performance competitiva para varredura automatizada:

| Métrica      | Valor (Validação)\* | Significado                                                       |
| :----------- | :------------------ | :---------------------------------------------------------------- |
| **F1-Score** | **~90%+**           | Média harmônica (Equilíbrio entre precisão e cobertura).          |
| Precision    | Alta                | Quando o modelo aponta um risco, ele é fidedigno.                 |
| Recall       | Alta                | O modelo captura a vasta maioria das cláusulas críticas de risco. |

_(Resultados aproximados dependentes da rodada de otimização Bayesiana)._

---

## 🚀 Como Usar

### Instalação

```bash
pip install transformers datasets seqeval accelerate evaluate torch pandas mlflow optuna
```

### Inferência (Simulação de Produção)

O notebook inclui um wrapper `predict_long_contract` que abstrai a complexidade do janelamento.

```python
from transformers import pipeline

# Carregar modelo treinado
model_path = "./legal-bert-ner-production/final_model"
pipe = pipeline("token-classification", model=model_path, tokenizer=model_path, aggregation_strategy="simple")

texto_contrato = """
SECTION 10. GOVERNING LAW. This Agreement shall be governed by the laws of the State of Sao Paulo.
"""

resultado = pipe(texto_contrato)
print(resultado)
# Output esperado: [{'entity_group': 'Governing Law', 'word': 'State of Sao Paulo', ...}]
```

---

## 📂 Estrutura do Projeto

```text
.
├── legal_ner_finetune_final_polished.ipynb  # Notebook Principal (End-to-End)
├── mlruns/                                  # Logs de experimentação do MLflow
├── legal-bert-ner-production/               # Artifacts do modelo final salvo
└── README.md                                # Documentação do Projeto
```

---

_Desenvolvido como case de Engenharia de Machine Learning focado em NLP Jurídico._
