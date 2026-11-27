#!/usr/bin/env python3
"""
SISTEMA DE DIAGNÓSTICO INTELIGENTE VIA MACHINE LEARNING (ISOLATION FOREST) - VERSÃO 5
===================================================================================
Bancada de Ensaios Automazoom — TCC SENAI Automação Industrial

Novidades da Versão 5:
- Adicionado tratamento e logs de depuração detalhados (status HTTP, contagem de pontos
  e resposta do servidor) na consulta ao InfluxDB v2 para diagnosticar falhas de conexão.
- Preserva todas as melhorias da v4 (isolar última batelada contínua, tratamento de CSV anotado,
  delta t real nas integrais, descoupling térmico).
"""

import io
import os
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 1. CONFIGURAÇÕES E PARÂMETROS DO INFLUXDB
# ==============================================================================

INFLUX_URL = os.getenv("INFLUX_URL", "http://influxdb_clp:8086")
INFLUX_ORG = os.getenv("INFLUX_ORG", "SENAI_TTAUT")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "DADOS_AUTOMAZOOM")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "TOKEN_TCC_SENAI_AUTOMACAO")

# Variáveis para o modelo de Machine Learning (Vetor de Características)
FEATURE_COLUMNS = ["velocidade_real", "pressao_pn2594", "nivel", "erro_nivel"]

# Compatibilidade universal para cálculo trapezoidal de integrais (NumPy 1.x e 2.x)
trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))


# ==============================================================================
# 2. SEPARAÇÃO DINÂMICA DE BATELADAS E CONSULTA AO INFLUXDB
# ==============================================================================
def isolar_ultima_batelada_continua(df, max_intervalo_segundos=5):
    """
    Identifica e extrai a última batelada completa (Regime Transitório + Permanente),
    detectando a pausa entre ensaios sem cortar os primeiros segundos do processo.
    """
    if df.empty or "_time" not in df.columns:
        return df

    df = df.sort_values("_time").reset_index(drop=True)
    deltas = df["_time"].diff().dt.total_seconds()

    # Identifica pontos de pausa maiores que 5 segundos entre registros
    pontos_de_corte = df.index[deltas > max_intervalo_segundos].tolist()

    if pontos_de_corte:
        ultimo_inicio = pontos_de_corte[-1]
        df_ultima_batelada = df.iloc[ultimo_inicio:].reset_index(drop=True)
    else:
        df_ultima_batelada = df

    return df_ultima_batelada


def buscar_dados_influx(tipo_ensaio="padrao", horas_atras=24):
    """
    Consulta o InfluxDB v2 via Flux Query. Exibe mensagens detalhadas no console sobre o
    status HTTP da resposta e a contagem de pontos carregados para facilidade de depuração.
    """
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -{horas_atras}h)
      |> filter(fn: (r) => r["_measurement"] == "dado" or r["equipamento"] == "automazoom")
      |> filter(fn: (r) => r["tipo_ensaio"] == "{tipo_ensaio}")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> sort(columns: ["_time"])
    '''
    headers = {
        "Authorization": f"Token {INFLUX_TOKEN}",
        "Content-Type": "application/vnd.flux",
        "Accept": "application/csv"
    }
    
    try:
        url = f"{INFLUX_URL}/api/v2/query?org={INFLUX_ORG}"
        response = requests.post(url, headers=headers, data=query, timeout=5)
        
        if response.status_code == 200 and len(response.text.strip()) > 0:
            # comment="#" descarta linhas de cabeçalho do InfluxDB (#datatype, #group, #default)
            df = pd.read_csv(io.StringIO(response.text), comment="#")
            if not df.empty and "_time" in df.columns:
                df["_time"] = pd.to_datetime(df["_time"], format="ISO8601")
                
                # Preenchimento temporal para tratar desencontros de milissegundos do S7
                df = df.ffill().bfill()
                
                # Garante cálculo do erro de nível
                if "setpoint_nivel" in df.columns and "nivel" in df.columns:
                    df["erro_nivel"] = np.abs(df["nivel"] - df["setpoint_nivel"])
                elif "nivel" in df.columns:
                    df["erro_nivel"] = np.abs(df["nivel"] - 250.0)
                
                df_isolado = isolar_ultima_batelada_continua(df)
                print(f"[OK] {len(df_isolado)} pontos reais carregados para '{tipo_ensaio}' (última batelada contínua de {len(df)} pontos no total).")
                return df_isolado
            else:
                print(f"[INFO] Resposta HTTP 200 recebida, mas o CSV está vazio ou sem coluna '_time' para '{tipo_ensaio}'.")
        else:
            print(f"[INFO] Sem dados reais para '{tipo_ensaio}' (HTTP {response.status_code}). Resposta: {response.text[:120]}")
    except Exception as e:
        print(f"[AVISO] Não foi possível conectar ao InfluxDB ({e}). Utilizando gerador de dados sintéticos para testes de ML.")

    return gerar_dados_sinteticos(tipo_ensaio)


def gerar_dados_sinteticos(tipo_ensaio="padrao", pontos=615):
    """
    Gera dados sintéticos realistas de ~615s (15s de regime transitório + 600s de permanente).
    """
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=pontos, freq="1s")
    t = np.linspace(0, pontos, pontos)
    setpoint = 250.0

    if tipo_ensaio == "padrao":
        ts_val = 14.0
        rpm_base = 1850.0 + 15.0 * np.sin(t / 20.0) + np.random.normal(0, 5, pontos)
        pressao_base = 1.25 + 0.02 * np.sin(t / 15.0) + np.random.normal(0, 0.005, pontos)
        nivel_base = setpoint * (1 - np.exp(-t / 4.0) * np.cos(t / 3.0)) + np.random.normal(0, 0.5, pontos)
        nivel_base[t > 15] = setpoint + np.random.normal(0, 0.8, np.sum(t > 15))
        temp_base = 23.5 + 0.001 * t + np.random.normal(0, 0.05, pontos)
    else:
        ts_val = 28.5
        rpm_base = 2380.0 + 35.0 * np.sin(t / 10.0) + np.random.normal(0, 10, pontos)
        pressao_base = 0.98 + 0.03 * np.cos(t / 12.0) + np.random.normal(0, 0.008, pontos)
        nivel_base = setpoint * (1 - np.exp(-t / 8.0) * np.cos(t / 2.0)) + np.random.normal(0, 1.2, pontos)
        nivel_base[t > 30] = setpoint + np.random.normal(0, 1.1, np.sum(t > 30))
        temp_base = 24.2 + 0.002 * t + np.random.normal(0, 0.05, pontos)

    df = pd.DataFrame({
        "_time": timestamps,
        "nivel": nivel_base,
        "setpoint_nivel": setpoint,
        "pressao_pn2594": pressao_base,
        "velocidade_real": rpm_base,
        "velocidade_comandada": rpm_base * 1.01,
        "temperatura": temp_base,
        "tempo_estabilizacao": ts_val,
        "tipo_ensaio": tipo_ensaio
    })
    df["erro_nivel"] = np.abs(df["nivel"] - df["setpoint_nivel"])
    return df


# ==============================================================================
# 3. MOTOR DE MACHINE LEARNING (ISOLATION FOREST)
# ==============================================================================
def treinar_e_avaliar_ml(df_padrao, df_atual):
    """
    Treina o modelo Isolation Forest no baseline e avalia o grau de anomalia da batelada atual.
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_padrao[FEATURE_COLUMNS])
    X_test = scaler.transform(df_atual[FEATURE_COLUMNS])

    model = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
    model.fit(X_train)

    predicoes = model.predict(X_test)
    scores = model.score_samples(X_test)

    pontos_anomalos = int(np.sum(predicoes == -1))
    total_pontos = len(predicoes)
    pct_anomalia = (pontos_anomalos / total_pontos) * 100.0 if total_pontos > 0 else 0.0
    score_medio = float(np.mean(scores))

    if pct_anomalia >= 25.0:
        status = "ALERTA CRÍTICO: ANOMALIA DETECTADA VIA MACHINE LEARNING"
        severidade = "ALTA"
        parecer = (
            f"O modelo de Machine Learning (Isolation Forest) classificou {pct_anomalia:.1f}% "
            f"dos pontos temporais da batelada como anômalos ({pontos_anomalos}/{total_pontos} amostras). "
            "Foi identificado um desvio estatístico severo na assinatura hidráulica do processo."
        )
    elif pct_anomalia >= 5.0:
        status = "ATENÇÃO: DESVIO OPERACIONAL DETECTADO VIA MACHINE LEARNING"
        severidade = "MÉDIA"
        parecer = (
            f"O algoritmo de ML detectou {pct_anomalia:.1f}% de amostras fora do padrão de treinamento. "
            "Existe uma perturbação moderada que afeta o comportamento nominal do sistema."
        )
    else:
        status = "NOMINAL: COMPORTAMENTO RECONHECIDO PELO MODELO DE ML"
        severidade = "NORMAL"
        parecer = (
            f"O modelo de ML validou a batelada atual com apenas {pct_anomalia:.1f}% de discrepância. "
            "O padrão de funcionamento corresponde com alta fidelidade ao baseline de treinamento."
        )

    return {
        "model": model,
        "scaler": scaler,
        "predicoes": predicoes,
        "scores": scores,
        "pontos_anomalos": pontos_anomalos,
        "total_pontos": total_pontos,
        "pct_anomalia": pct_anomalia,
        "score_medio": score_medio,
        "status": status,
        "severidade": severidade,
        "parecer": parecer
    }


# ==============================================================================
# 4. EXTRAÇÃO DE INDICADORES HIDRÁULICOS COMPLEMENTARES
# ==============================================================================
def safe_pct_diff(val_atual, val_padrao):
    """Calcula a variação percentual com proteção contra divisão por zero."""
    if val_padrao == 0 or np.isnan(val_padrao):
        return 0.0
    return ((val_atual - val_padrao) / val_padrao) * 100.0


def calcular_indicadores_hidraulicos(df):
    """Calcula indicadores estatísticos e hidráulicos utilizando a amostragem de tempo real."""
    if df.empty or "_time" not in df.columns:
        return {}

    t_seg = (df["_time"] - df["_time"].iloc[0]).dt.total_seconds().to_numpy()

    rpm_medio = float(df["velocidade_real"].mean())
    rpm_pico = float(df["velocidade_real"].max())
    esforco_acumulado = float(trapz_fn(df["velocidade_real"], x=t_seg))
    pressao_media = float(df["pressao_pn2594"].mean())
    
    setpoint = df["setpoint_nivel"].iloc[-1] if "setpoint_nivel" in df.columns else 250.0
    erro_abs = np.abs(df["nivel"] - setpoint)
    iae = float(trapz_fn(erro_abs, x=t_seg))
    
    ts = float(df["tempo_estabilizacao"].max()) if "tempo_estabilizacao" in df.columns else 0.0
    temp_media = float(df["temperatura"].mean())

    return {
        "rpm_medio": rpm_medio,
        "rpm_pico": rpm_pico,
        "esforco_acumulado": esforco_acumulado,
        "pressao_media": pressao_media,
        "iae": iae,
        "ts": ts,
        "temp_media": temp_media,
        "duracao_segundos": float(t_seg[-1]) if len(t_seg) > 0 else 0.0,
        "total_amostras": len(df)
    }


# ==============================================================================
# 5. GERAÇÃO DO RELATÓRIO MARKDOWN COM INFERÊNCIAS DE ML
# ==============================================================================
def gerar_relatorio_ml(m_p, m_a, ml_res, filepath="relatorio_diagnostico_automazoom_ml.md"):
    data_hora = pd.Timestamp.now().strftime("%d/%m/%Y %H:%M:%S")
    dev_rpm = safe_pct_diff(m_a["rpm_medio"], m_p["rpm_medio"])
    dev_pressao = safe_pct_diff(m_a["pressao_media"], m_p["pressao_media"])
    dev_ts = safe_pct_diff(m_a["ts"], m_p["ts"])
    dev_esforco = safe_pct_diff(m_a["esforco_acumulado"], m_p["esforco_acumulado"])
    dev_iae = safe_pct_diff(m_a["iae"], m_p["iae"])
    dev_temp = safe_pct_diff(m_a["temp_media"], m_p["temp_media"])

    md_content = f"""# 🤖 RELATÓRIO DE DIAGNÓSTICO VIA MACHINE LEARNING (v5)
**Bancada de Ensaios Automazoom — Sistema Inteligente de Automação**  
*Modelo Utilizado:* Isolation Forest (Scikit-Learn) | *Data:* {data_hora}  
*Bucket InfluxDB:* `{INFLUX_BUCKET}` | *Organização:* `{INFLUX_ORG}`

---

## 1. RESUMO EXECUTIVO E DIAGNÓSTICO DO MODELO DE IA

| Parâmetro de Avaliação | Resultado Obtido pelo Algoritmo |
| :--- | :--- |
| **Diagnóstico do Modelo ML** | **{ml_res['status']}** |
| **Nível de Severidade** | **{ml_res['severidade']}** |
| **Taxa de Anomalia Detectada** | **`{ml_res['pct_anomalia']:.1f}%`** (`{ml_res['pontos_anomalos']}/{ml_res['total_pontos']}` pontos) |
| **Score Médio de Anomalia (ML)**| `{ml_res['score_medio']:.4f}` |
| **Duração da Batelada Avaliada**| `{m_a['duracao_segundos']:.1f} s` (`{m_a['total_amostras']}` amostras) |
| **Variação no Tempo $t_s$** | `{m_a['ts']:.1f} s` (`{dev_ts:+.1f}%` em relação ao baseline) |
| **Variação de Esforço (RPM)** | `{m_a['rpm_medio']:.1f} RPM` (`{dev_rpm:+.1f}%` em relação ao baseline) |

### 🔍 Parecer Técnico Automatizado (Machine Learning)
> **{ml_res['parecer']}**

---

## 2. POR QUE O MACHINE LEARNING ADAPTA-SE AUTOMATICAMENTE?

A principal vantagem deste módulo de **Machine Learning Não Supervisionado (Isolation Forest)** é a **generalização adaptativa**:

1. **Independência de Limiares Fixos:** O modelo não utiliza regras manuais fixas (como `if rpm > 2000`). Ele constrói uma estrutura de decisão multivariada que mapeia a distribuição normal das variáveis (`velocidade_real`, `pressao_pn2594`, `nivel`, `erro_nivel`).
2. **Re-treinamento Simplificado (`.fit()`):** Se as características físicas da bancada mudarem (ex.: novo Setpoint de nível, substituição da bomba ou alteração de tubulação), **basta executar uma nova batelada padrão**. O script executará o treinamento (`fit`) sobre os novos dados sem que uma única linha de código precise ser alterada.

---

## 3. COMPARATIVO DAS GRANDEZAS FÍSICAS (DESVIOS DO BASELINE)

| Grandeza Medida | Batelada Padrão (Treinamento ML) | Batelada Atual (Inferência ML) | Desvio Relativo (%) |
| :--- | :---: | :---: | :---: |
| **Tempo de Estabilização ($t_s$)** | `{m_p['ts']:.1f} s` | `{m_a['ts']:.1f} s` | **`{dev_ts:+.1f}%`** |
| **Rotação Média da Bomba** | `{m_p['rpm_medio']:.1f} RPM` | `{m_a['rpm_medio']:.1f} RPM` | **`{dev_rpm:+.1f}%`** |
| **Esforço Acumulado (Integral RPM)** | `{m_p['esforco_acumulado']:.0f} RPM·s` | `{m_a['esforco_acumulado']:.0f} RPM·s` | **`{dev_esforco:+.1f}%`** |
| **Pressão no Recalque (PN2594)** | `{m_p['pressao_media']:.2f} bar` | `{m_a['pressao_media']:.2f} bar` | **`{dev_pressao:+.1f}%`** |
| **Erro Acumulado (IAE)** | `{m_p['iae']:.1f} mm·s` | `{m_a['iae']:.1f} mm·s` | **`{dev_iae:+.1f}%`** |
| **Temperatura Média do Fluido** | `{m_p['temp_media']:.1f} °C` | `{m_a['temp_media']:.1f} °C` | **`{dev_temp:+.1f}%`** |

---

## 4. RECOMENDAÇÕES DE MANUTENÇÃO E RECALIBRAÇÃO

- [ ] **Validação do Modelo:** Caso altere o setpoint ou características hidráulicas da bancada, execute o ensaio nominal e execute novamente `python3 gerador_relatorio_influxdb-v5.py` para recalibrar o modelo de ML.
- [ ] **Inspeção de Linha:** Se a taxa de anomalia for $> 25\\%$, verificar pontos de atrito mecânico ou vazamento na tubulação de recalque.
- [ ] **Checagem de Sensores:** Confirmar a calibração do sensor de pressão `ifm PN2594` e do transmissor de nível `LT3023`.

---
*Relatório gerado automaticamente por algoritmo de Machine Learning (Isolation Forest v5) — TCC SENAI.*
"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[SUCESSO] Relatório de ML gerado com êxito em: {filepath}")
    return filepath


# ==============================================================================
# 6. EXECUÇÃO PRINCIPAL
# ==============================================================================
def main():
    print("=================================================================")
    print("  INICIANDO SISTEMA DE DIAGNÓSTICO VIA MACHINE LEARNING (ISOLATION FOREST v5)")
    print("=================================================================")

    print("[1/4] Coletando dados da Batelada Padrão (Baseline)...")
    df_padrao = buscar_dados_influx(tipo_ensaio="padrao")

    print("[2/4] Coletando dados da Batelada Atual (Teste)...")
    df_atual = buscar_dados_influx(tipo_ensaio="perturbacao")

    print("[3/4] Treinando modelo de Machine Learning e avaliando anomalias...")
    ml_res = treinar_e_avaliar_ml(df_padrao, df_atual)
    ind_padrao = calcular_indicadores_hidraulicos(df_padrao)
    ind_atual = calcular_indicadores_hidraulicos(df_atual)

    print("[4/4] Gerando relatório diagnóstico baseado em ML...")
    gerar_relatorio_ml(ind_padrao, ind_atual, ml_res)

    print("\n-----------------------------------------------------------------")
    print(f" RESULTADO ML: {ml_res['status']}")
    print(f" PONTOS ANÔMALOS DETECTADOS PELO MODELO: {ml_res['pct_anomalia']:.1f}% ({ml_res['pontos_anomalos']}/{ml_res['total_pontos']})")
    print("-----------------------------------------------------------------\n")

if __name__ == "__main__":
    main()
