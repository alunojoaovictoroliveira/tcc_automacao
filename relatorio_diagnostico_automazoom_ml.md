# 🤖 RELATÓRIO DE DIAGNÓSTICO VIA MACHINE LEARNING (v5)
**Bancada de Ensaios Automazoom — Sistema Inteligente de Automação**  
*Modelo Utilizado:* Isolation Forest (Scikit-Learn) | *Data:* 26/11/2026 21:46:25  
*Bucket InfluxDB:* `DADOS_AUTOMAZOOM` | *Organização:* `SENAI_TTAUT`

---

## 1. RESUMO EXECUTIVO E DIAGNÓSTICO DO MODELO DE IA

| Parâmetro de Avaliação | Resultado Obtido pelo Algoritmo |
| :--- | :--- |
| **Diagnóstico do Modelo ML** | **ALERTA CRÍTICO: ANOMALIA DETECTADA VIA MACHINE LEARNING** |
| **Nível de Severidade** | **ALTA** |
| **Taxa de Anomalia Detectada** | **`77.2%`** (`475/615` pontos) |
| **Score Médio de Anomalia (ML)**| `-0.5936` |
| **Duração da Batelada Avaliada**| `614.0 s` (`615` amostras) |
| **Variação no Tempo $t_s$** | `28.5 s` (`+103.6%` em relação ao baseline) |
| **Variação de Esforço (RPM)** | `2380.0 RPM` (`+28.6%` em relação ao baseline) |

### 🔍 Parecer Técnico Automatizado (Machine Learning)
> **O modelo de Machine Learning (Isolation Forest) classificou 77.2% dos pontos temporais da batelada como anômalos (475/615 amostras). Foi identificado um desvio estatístico severo na assinatura hidráulica do processo.**

---

## 2. POR QUE O MACHINE LEARNING ADAPTA-SE AUTOMATICAMENTE?

A principal vantagem deste módulo de **Machine Learning Não Supervisionado (Isolation Forest)** é a **generalização adaptativa**:

1. **Independência de Limiares Fixos:** O modelo não utiliza regras manuais fixas (como `if rpm > 2000`). Ele constrói uma estrutura de decisão multivariada que mapeia a distribuição normal das variáveis (`velocidade_real`, `pressao_pn2594`, `nivel`, `erro_nivel`).
2. **Re-treinamento Simplificado (`.fit()`):** Se as características físicas da bancada mudarem (ex.: novo Setpoint de nível, substituição da bomba ou alteração de tubulação), **basta executar uma nova batelada padrão**. O script executará o treinamento (`fit`) sobre os novos dados sem que uma única linha de código precise ser alterada.

---

## 3. COMPARATIVO DAS GRANDEZAS FÍSICAS (DESVIOS DO BASELINE)

| Grandeza Medida                      | Batelada Padrão (Treinamento ML) | Batelada Atual (Inferência ML) | Desvio Relativo (%) |
| :----------------------------------- | :------------------------------: | :----------------------------: | :-----------------: |
| **Tempo de Estabilização ($t_s$)**   |             `14.0 s`             |            `28.5 s`            |    **`+103.6%`**    |
| **Rotação Média da Bomba**           |           `1850.2 RPM`           |          `2380.0 RPM`          |    **`+28.6%`**     |
| **Esforço Acumulado (Integral RPM)** |         `1135999 RPM·s`          |        `1461315 RPM·s`         |    **`+28.6%`**     |
| **Pressão no Recalque (PN2594)**     |            `1.25 bar`            |           `0.98 bar`           |    **`-21.6%`**     |
| **Erro Acumulado (IAE)**             |          `1048.3 mm·s`           |         `1742.1 mm·s`          |    **`+66.2%`**     |
| **Temperatura Média do Fluido**      |            `23.8 °C`             |           `24.8 °C`            |     **`+4.2%`**     |

---

## 4. RECOMENDAÇÕES DE MANUTENÇÃO E RECALIBRAÇÃO

- [ ] **Validação do Modelo:** Caso altere o setpoint ou características hidráulicas da bancada, execute o ensaio nominal e execute novamente `python3 gerador_relatorio_influxdb-v5.py` para recalibrar o modelo de ML.
- [ ] **Inspeção de Linha:** Se a taxa de anomalia for $> 25\%$, verificar pontos de atrito mecânico ou vazamento na tubulação de recalque.
- [ ] **Checagem de Sensores:** Confirmar a calibração do sensor de pressão `ifm PN2594` e do transmissor de nível `LT3023`.

---
*Relatório gerado automaticamente por algoritmo de Machine Learning (Isolation Forest v5) — TCC SENAI.*
