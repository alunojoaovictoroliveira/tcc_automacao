# 🤖 RELATÓRIO DE DIAGNÓSTICO VIA MACHINE LEARNING (v5)
**Bancada de Ensaios Automazoom — Sistema Inteligente de Automação**  
*Modelo Utilizado:* Isolation Forest (Scikit-Learn) | *Data:* 27/11/2026 10:45:03  
*Bucket InfluxDB:* `DADOS_AUTOMAZOOM` | *Organização:* `SENAI_TTAUT`

---

## 1. RESUMO EXECUTIVO E DIAGNÓSTICO DO MODELO DE IA

| Parâmetro de Avaliação | Resultado Obtido pelo Algoritmo |
| :--- | :--- |
| **Diagnóstico do Modelo ML** | **NOMINAL: COMPORTAMENTO RECONHECIDO PELO MODELO DE ML** |
| **Nível de Severidade** | **NORMAL** |
| **Taxa de Anomalia Detectada** | **`1.6%`** (`9/552` pontos) |
| **Score Médio de Anomalia (ML)**| `-0.3958` |
| **Duração da Batelada Avaliada**| `637.6 s` (`552` amostras) |
| **Variação no Tempo $t_s$** | `38.3 s` (`+1.0%` em relação ao baseline) |
| **Variação de Esforço (RPM)** | `717.6 RPM` (`+0.0%` em relação ao baseline) |

### 🔍 Parecer Técnico Automatizado (Machine Learning)
> **O modelo de ML validou a batelada atual com apenas 1.6% de discrepância. O padrão de funcionamento corresponde com alta fidelidade ao baseline de treinamento.**

---

## 2. POR QUE O MACHINE LEARNING ADAPTA-SE AUTOMATICAMENTE?

A principal vantagem deste módulo de **Machine Learning Não Supervisionado (Isolation Forest)** é a **generalização adaptativa**:

1. **Independência de Limiares Fixos:** O modelo não utiliza regras manuais fixas (como `if rpm > 2000`). Ele constrói uma estrutura de decisão multivariada que mapeia a distribuição normal das variáveis (`velocidade_real`, `pressao_pn2594`, `nivel`, `erro_nivel`).
2. **Re-treinamento Simplificado (`.fit()`):** Se as características físicas da bancada mudarem (ex.: novo Setpoint de nível, substituição da bomba ou alteração de tubulação), **basta executar uma nova batelada padrão**. O script executará o treinamento (`fit`) sobre os novos dados sem que uma única linha de código precise ser alterada.

---

## 3. COMPARATIVO DAS GRANDEZAS FÍSICAS (DESVIOS DO BASELINE)

| Grandeza Medida | Batelada Padrão (Treinamento ML) | Batelada Atual (Inferência ML) | Desvio Relativo (%) |
| :--- | :---: | :---: | :---: |
| **Tempo de Estabilização ($t_s$)** | `38.0 s` | `38.3 s` | **`+1.0%`** |
| **Rotação Média da Bomba** | `717.6 RPM` | `717.6 RPM` | **`+0.0%`** |
| **Esforço Acumulado (Integral RPM)** | `449006 RPM·s` | `448465 RPM·s` | **`-0.1%`** |
| **Pressão no Recalque (PN2594)** | `0.08 bar` | `0.08 bar` | **`-1.1%`** |
| **Erro Acumulado (IAE)** | `1486.9 mm·s` | `1519.1 mm·s` | **`+2.2%`** |
| **Temperatura Média do Fluido** | `24.2 °C` | `24.2 °C` | **`+0.2%`** |

---

## 4. RECOMENDAÇÕES DE MANUTENÇÃO E RECALIBRAÇÃO

- [ ] **Validação do Modelo:** Caso altere o setpoint ou características hidráulicas da bancada, execute o ensaio nominal e execute novamente `python3 gerador_relatorio_influxdb-v5.py` para recalibrar o modelo de ML.
- [ ] **Inspeção de Linha:** Se a taxa de anomalia for $> 25\%$, verificar pontos de atrito mecânico ou vazamento na tubulação de recalque.
- [ ] **Checagem de Sensores:** Confirmar a calibração do sensor de pressão `ifm PN2594` e do transmissor de nível `LT3023`.

---
*Relatório gerado automaticamente por algoritmo de Machine Learning (Isolation Forest v5) — TCC SENAI.*
