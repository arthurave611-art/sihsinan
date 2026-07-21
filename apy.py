import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import unicodedata
import re
import os
import io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import mapclassify
import libpysal
from esda.moran import Moran, Moran_Local
from statsmodels.tsa.seasonal import STL
import pymannkendall as mk

st.set_page_config(layout="wide")
st.title("Análise Epidemiológica Integrada — Mapas, Espacial (Moran/LISA) e Séries Temporais")

# ----------------------------------------------------------------------
# ARQUIVOS DE REFERÊNCIA (devem estar no repositório, junto com este app.py)
# ----------------------------------------------------------------------
ARQUIVO_POPULACAO = "populacao_ibge.csv"   # projeção de população IBGE/TabNet
ARQUIVO_MAPA = "mapa.json"                  # GeoJSON de municípios (usado só nos módulos espaciais)

PERMUTACOES = 999    # nº de simulações de Monte Carlo p/ pseudo p-valor (Moran e LISA)
ALPHA = 0.05         # limiar de significância (LISA e Mann-Kendall)
PERIODO_SAZONAL = 12  # dados mensais -> 1 ciclo sazonal = 12 meses

if "etapa" not in st.session_state:
    st.session_state.etapa = "Upload"


# ----------------------------------------------------------------------
# TABELAS DE REFERÊNCIA — código IBGE (2 primeiros dígitos) -> UF/Região
# (usadas para permitir agregação por Estado/Região sem depender do mapa.json)
# ----------------------------------------------------------------------
SIGLA_POR_CODIGO = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP", "17": "TO",
    "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB", "26": "PE", "27": "AL",
    "28": "SE", "29": "BA",
    "31": "MG", "32": "ES", "33": "RJ", "35": "SP",
    "41": "PR", "42": "SC", "43": "RS",
    "50": "MS", "51": "MT", "52": "GO", "53": "DF",
}

NOME_UF_POR_CODIGO = {
    "11": "RONDONIA", "12": "ACRE", "13": "AMAZONAS", "14": "RORAIMA", "15": "PARA",
    "16": "AMAPA", "17": "TOCANTINS",
    "21": "MARANHAO", "22": "PIAUI", "23": "CEARA", "24": "RIO GRANDE DO NORTE",
    "25": "PARAIBA", "26": "PERNAMBUCO", "27": "ALAGOAS", "28": "SERGIPE", "29": "BAHIA",
    "31": "MINAS GERAIS", "32": "ESPIRITO SANTO", "33": "RIO DE JANEIRO", "35": "SAO PAULO",
    "41": "PARANA", "42": "SANTA CATARINA", "43": "RIO GRANDE DO SUL",
    "50": "MATO GROSSO DO SUL", "51": "MATO GROSSO", "52": "GOIAS", "53": "DISTRITO FEDERAL",
}

REGIAO_POR_CODIGO = {
    "11": "NORTE", "12": "NORTE", "13": "NORTE", "14": "NORTE", "15": "NORTE",
    "16": "NORTE", "17": "NORTE",
    "21": "NORDESTE", "22": "NORDESTE", "23": "NORDESTE", "24": "NORDESTE",
    "25": "NORDESTE", "26": "NORDESTE", "27": "NORDESTE", "28": "NORDESTE", "29": "NORDESTE",
    "31": "SUDESTE", "32": "SUDESTE", "33": "SUDESTE", "35": "SUDESTE",
    "41": "SUL", "42": "SUL", "43": "SUL",
    "50": "CENTRO-OESTE", "51": "CENTRO-OESTE", "52": "CENTRO-OESTE", "53": "CENTRO-OESTE",
}

MESES_PT = {
    "JAN": 1, "FEV": 2, "MAR": 3, "ABR": 4, "MAI": 5, "JUN": 6,
    "JUL": 7, "AGO": 8, "SET": 9, "OUT": 10, "NOV": 11, "DEZ": 12,
}
MESES_PT_INV = {v: k.capitalize() for k, v in MESES_PT.items()}


# ----------------------------------------------------------------------
# FUNÇÕES AUXILIARES — limpeza e normalização (comuns a todos os módulos)
# ----------------------------------------------------------------------
def limpar_codigo(serie: pd.Series) -> pd.Series:
    """Normaliza códigos de município para os 6 primeiros dígitos (IBGE sem DV).
    Funciona tanto para códigos de 6 dígitos (SIH/TabNet) quanto de 7 dígitos
    com dígito verificador (CD_MUN do mapa)."""
    s = serie.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.str.extract(r"(\d+)")[0]

    def _seis_digitos(x):
        if pd.isna(x):
            return np.nan
        x = str(x)
        return x[:6] if len(x) >= 7 else x.zfill(6)

    return s.map(_seis_digitos)


def extrair_nome_municipio(valor_bruto) -> str:
    """Remove o código IBGE colado no nome do município (ex.: '110001 ALTA
    FLORESTA D'OESTE' -> 'ALTA FLORESTA D'OESTE'), para casar com o nome de
    município no arquivo de população."""
    s = str(valor_bruto).strip()
    s = re.sub(r"^\d+\s+", "", s)
    return s


def normalizar_texto(valor) -> str:
    """Remove acentos, espaços extras, prefixos hierárquicos do TabNet
    (".. ", "Região ") e padroniza caixa, para casar nomes de Região/UF/País
    vindos de fontes diferentes (mapa/código IBGE x TabNet população)."""
    if pd.isna(valor):
        return ""
    s = str(valor).strip().upper()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"^\.+\s*", "", s)
    s = re.sub(r"^REGIAO\s+", "", s)
    return s.strip()


def taxa_bruta(casos: pd.Series, pop: pd.Series, base: int) -> pd.Series:
    casos = casos.astype(float)
    pop = pop.astype(float)
    out = pd.Series(np.nan, index=casos.index)
    mask = pop > 0
    out[mask] = (casos[mask] / pop[mask]) * base
    return out


def taxa_bayesiana(casos: pd.Series, pop: pd.Series, base: int) -> pd.Series:
    """
    Suavização Bayesiana Empírica Global (Empirical Bayes) — validada
    numericamente contra a implementação de referência da biblioteca `esda`
    (mesma lógica usada no GeoDa para taxas em áreas pequenas):

        r_i    = O_i / P_i
        m      = sum(O_i) / sum(P_i)
        s2     = [sum(P_i*(r_i-m)^2)/sum(P_i)] - m/mean(P_i)
        w_i    = s2 / (s2 + m/P_i)
        r_i^EB = w_i * r_i + (1 - w_i) * m
    """
    casos = casos.astype(float)
    pop = pop.astype(float)
    mask = pop > 0

    r = pd.Series(0.0, index=casos.index)
    r[mask] = casos[mask] / pop[mask]

    total_pop = pop[mask].sum()
    total_casos = casos[mask].sum()

    if total_pop <= 0 or mask.sum() == 0:
        return pd.Series(np.nan, index=casos.index)

    m = total_casos / total_pop
    media_pop = pop[mask].mean()

    s2 = (pop[mask] * (r[mask] - m) ** 2).sum() / total_pop - (m / media_pop)
    if s2 < 0 or pd.isna(s2):
        s2 = 0.0

    w = pd.Series(0.0, index=casos.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        w[mask] = s2 / (s2 + (m / pop[mask]))
    w[mask] = w[mask].clip(0, 1)

    r_eb = pd.Series(m, index=casos.index)
    r_eb[mask] = w[mask] * r[mask] + (1 - w[mask]) * m

    return r_eb * base


# ----------------------------------------------------------------------
# FUNÇÕES AUXILIARES — posição/tamanho customizável da legenda dos mapas
# (usado nos módulos Mapas Coropléticos e Espacial/LISA, para o usuário
# poder mover a legenda quando ela tampa alguma área do mapa)
# ----------------------------------------------------------------------
OPCOES_POSICAO_LEGENDA = {
    "Inferior esquerda": "lower left",
    "Inferior direita": "lower right",
    "Superior esquerda": "upper left",
    "Superior direita": "upper right",
    "Centro esquerda": "center left",
    "Centro direita": "center right",
    "Superior centro": "upper center",
    "Inferior centro": "lower center",
    "Fora do mapa (direita)": "center left",
    "Fora do mapa (abaixo)": "upper center",
}

ANCORAS_LEGENDA = {
    "Inferior esquerda": (0.02, 0.02),
    "Inferior direita": (0.98, 0.02),
    "Superior esquerda": (0.02, 0.98),
    "Superior direita": (0.98, 0.98),
    "Centro esquerda": (0.02, 0.5),
    "Centro direita": (0.98, 0.5),
    "Superior centro": (0.5, 0.98),
    "Inferior centro": (0.5, 0.02),
    "Fora do mapa (direita)": (1.15, 0.5),
    "Fora do mapa (abaixo)": (0.5, -0.15),
}


def controles_posicao_legenda(chave_prefixo: str, indice_padrao: int = 0):
    """Renderiza na sidebar os controles de posição/tamanho/ajuste fino da
    legenda e devolve (posicao_label, fontsize, deslocamento_x, deslocamento_y).
    `chave_prefixo` evita colisão de key entre os módulos Mapa e Espacial."""
    st.subheader("Legenda")
    posicao_legenda = st.selectbox(
        "Posição da legenda:",
        list(OPCOES_POSICAO_LEGENDA.keys()),
        index=indice_padrao,
        help=(
            "Se a legenda estiver tampando alguma área do mapa, troque a posição "
            "aqui — as opções 'Fora do mapa' colocam a legenda totalmente fora da "
            "área do mapa (à direita ou embaixo)."
        ),
        key=f"{chave_prefixo}_posicao_legenda",
    )
    fonte_legenda = st.slider(
        "Tamanho da fonte da legenda:", 6, 16, 9, key=f"{chave_prefixo}_fonte_legenda"
    )
    ajuste_fino = st.checkbox(
        "Ajuste fino de posição (x/y)", key=f"{chave_prefixo}_ajuste_fino_legenda"
    )
    deslocamento_x = deslocamento_y = 0.0
    if ajuste_fino:
        col_lx, col_ly = st.columns(2)
        deslocamento_x = col_lx.slider(
            "Deslocamento X:", -0.5, 0.5, 0.0, step=0.02, key=f"{chave_prefixo}_desloc_x_legenda"
        )
        deslocamento_y = col_ly.slider(
            "Deslocamento Y:", -0.5, 0.5, 0.0, step=0.02, key=f"{chave_prefixo}_desloc_y_legenda"
        )
    return posicao_legenda, fonte_legenda, deslocamento_x, deslocamento_y


def kwargs_legenda(posicao_label: str, fontsize: int, deslocamento_x: float = 0.0, deslocamento_y: float = 0.0) -> dict:
    """Monta os kwargs de ax.legend(...) a partir da posição escolhida pelo
    usuário, permitindo mover/redimensionar a legenda livremente."""
    ancora_x, ancora_y = ANCORAS_LEGENDA[posicao_label]
    return {
        "loc": OPCOES_POSICAO_LEGENDA[posicao_label],
        "bbox_to_anchor": (ancora_x + deslocamento_x, ancora_y + deslocamento_y),
        "fontsize": fontsize,
        "framealpha": 0.9,
        "borderaxespad": 0.0,
    }


# ----------------------------------------------------------------------
# FUNÇÕES AUXILIARES — colunas de competência (Ano/Mês) e de ano puro
# ----------------------------------------------------------------------
def parse_coluna_data(nome_coluna: str):
    """Tenta interpretar um nome de coluna como Ano/Mês ou Mês/Ano
    (ex.: '2008/Jan', 'Jan/2008', '2008-01') e devolve um pd.Timestamp
    no primeiro dia do mês, ou None se a coluna não for uma competência."""
    s = normalizar_texto(nome_coluna)
    s = s.replace("-", "/").replace(".", "/").replace(" ", "/")
    partes = [p for p in s.split("/") if p != ""]
    if len(partes) != 2:
        return None
    a, b = partes
    if a.isdigit() and len(a) == 4 and b in MESES_PT:
        return pd.Timestamp(year=int(a), month=MESES_PT[b], day=1)
    if b.isdigit() and len(b) == 4 and a in MESES_PT:
        return pd.Timestamp(year=int(b), month=MESES_PT[a], day=1)
    if a.isdigit() and len(a) == 4 and b.isdigit() and 1 <= int(b) <= 12:
        return pd.Timestamp(year=int(a), month=int(b), day=1)
    if b.isdigit() and len(b) == 4 and a.isdigit() and 1 <= int(a) <= 12:
        return pd.Timestamp(year=int(b), month=int(a), day=1)
    return None


def _campo_e_cabecalho_valido(campo: str) -> bool:
    """Um campo do cabeçalho é válido se for um ano puro (ex.: '2008', usado
    no arquivo de população ou em exports antigos do SIH) OU uma competência
    Ano/Mês (ex.: '2014/Jul', 'Jul/2014') — sem isso, um cabeçalho todo em
    Ano/Mês nunca seria reconhecido como cabeçalho e a detecção cairia por
    engano numa linha de dados."""
    if campo.isdigit():
        return True
    return parse_coluna_data(campo) is not None


def detectar_linha_cabecalho(conteudo_bytes: bytes, sep: str, encoding: str) -> int:
    """Encontra automaticamente a linha onde começa a tabela de dados
    (ex: 'Município;2019;2020;...' ou 'Município;2019/Jan;2019/Fev;...'),
    não importa quantas linhas de título ou de filtro (ex: 'Unidade da
    Federação: Tocantins') o TabNet tenha colocado antes."""
    texto = conteudo_bytes.decode(encoding, errors="replace")
    linhas = texto.splitlines()
    for i, linha in enumerate(linhas):
        campos = [c.strip().strip('"') for c in linha.split(sep)]
        validos = sum(1 for c in campos if _campo_e_cabecalho_valido(c))
        if validos >= 2:
            return i
    return 0


def ler_csv(arquivo, sep, skiprows, encoding):
    return pd.read_csv(arquivo, encoding=encoding, skiprows=skiprows, sep=sep)


def colunas_de_ano(df: pd.DataFrame):
    """Colunas que são só um ano puro (ex.: população do IBGE: '2008', '2009'...)."""
    return [c for c in df.columns if str(c).strip().isdigit()]


def identificar_colunas_data(df: pd.DataFrame, fonte: str = "SIH"):
    """Varre as colunas do CSV (SIH ou SINAN) e devolve (colunas_data, granularidade_mensal):

    - colunas_data: {coluna_original: Timestamp} para cada competência encontrada.
    - granularidade_mensal: True se o arquivo realmente tem competência Ano/Mês
      (ex.: '2014/Jul'); False se o arquivo só tem colunas de ANO PURO — nesse
      caso cada ano é tratado como 1 ponto anual (31/12), suficiente para os
      módulos de Mapa e Espacial (Moran/LISA), mas não para Séries Temporais
      mensais (STL/Mann-Kendall/Índice Sazonal).

    `fonte` controla o que é tentado:
    - "SIH": tenta Ano/Mês primeiro (ex.: '2008/Jan'); se não achar nenhuma,
      cai para ano puro (formato TabNet mais antigo).
    - "SINAN": o TabNet do SINAN só exporta por Mês (agrupando todos os anos
      do período, sem granularidade real) ou por Ano puro — nunca por Ano/Mês
      de verdade. Por isso, para SINAN, nem se tenta reconhecer competência
      mensal; só colunas de ano puro são aceitas.
    """
    colunas_data = {}

    if fonte == "SIH":
        for col in df.columns:
            ts = parse_coluna_data(col)
            if ts is not None:
                colunas_data[col] = ts

        if colunas_data:
            return colunas_data, True

    for col in df.columns:
        texto = str(col).strip()
        if texto.isdigit() and len(texto) == 4:
            colunas_data[col] = pd.Timestamp(year=int(texto), month=12, day=31)

    return colunas_data, False


def ano_mais_proximo(ano_alvo: str, colunas_ano: list):
    """Se o ano exato não existir na tabela de população (projeção),
    usa o ano disponível mais próximo."""
    if str(ano_alvo) in [str(c) for c in colunas_ano]:
        return str(ano_alvo)
    alvo = int(ano_alvo)
    disponiveis = [int(c) for c in colunas_ano]
    mais_proximo = min(disponiveis, key=lambda x: abs(x - alvo))
    return str(mais_proximo)


def detectar_encoding(conteudo_bytes: bytes) -> str:
    """Detecta se o arquivo está em UTF-8 ou Latin-1. Tenta UTF-8 (com/sem
    BOM) de forma estrita primeiro; só cai para Latin-1 se o UTF-8 realmente
    falhar. Ler um arquivo com o encoding errado corrompe nomes com í/ô de
    forma silenciosa (ex.: Piauí, Rondônia, Espírito Santo, Paraíba)."""
    try:
        conteudo_bytes.decode("utf-8-sig")
        return "utf-8-sig"
    except UnicodeDecodeError:
        return "latin-1"


@st.cache_data
def carregar_populacao(encoding_manual: str | None = None):
    """Carrega o CSV de população (padrão TabNet/IBGE) direto do repositório,
    sem precisar de upload — é um dado de referência que não muda por consulta."""
    with open(ARQUIVO_POPULACAO, "rb") as f:
        conteudo = f.read()
    encoding_usado = encoding_manual or detectar_encoding(conteudo)
    skip = detectar_linha_cabecalho(conteudo, sep=";", encoding=encoding_usado)
    df = ler_csv(io.BytesIO(conteudo), sep=";", skiprows=skip, encoding=encoding_usado)
    df.columns = df.columns.str.strip()
    return df, encoding_usado


@st.cache_data
def carregar_mapa():
    """Carrega o mapa.json (GeoJSON de municípios) do repositório — usado
    apenas nos módulos espaciais (Mapas Coropléticos e Moran/LISA)."""
    gdf = gpd.read_file(ARQUIVO_MAPA, encoding="utf-8")
    if not gdf.geometry.is_valid.all():
        gdf["geometry"] = gdf.geometry.make_valid()
    return gdf


def agregar_para_anual(df_sih: pd.DataFrame, colunas_data: dict):
    """Agrega as colunas de competência (Ano/Mês, ou ano puro) em totais
    ANUAIS por município, para uso nos módulos espaciais — Mapas Coropléticos
    e Moran/LISA — que trabalham em nível de 1 observação por município/ano,
    diferente do módulo de Séries Temporais (que usa a granularidade mensal
    original, quando disponível)."""
    col_municipio = df_sih.columns[0]
    colunas_base = [c for c in [col_municipio, "id_join", "codigo_uf"] if c in df_sih.columns]
    df_anual = df_sih[colunas_base].copy()

    anos = sorted({ts.year for ts in colunas_data.values()})
    for ano in anos:
        cols_do_ano = [c for c, ts in colunas_data.items() if ts.year == ano]
        valores = df_sih[cols_do_ano].apply(
            lambda col: pd.to_numeric(col.astype(str).replace("-", "0"), errors="coerce")
        ).fillna(0)
        df_anual[str(ano)] = valores.sum(axis=1)

    return df_anual, [str(a) for a in anos]


# ----------------------------------------------------------------------
# FUNÇÕES AUXILIARES — agregação de série temporal (País / Região / Estado / Município)
# ----------------------------------------------------------------------
def montar_serie_casos(df_subset: pd.DataFrame, colunas_data: dict) -> pd.Series:
    """Soma os casos de todos os municípios do subconjunto, competência a competência."""
    valores = {}
    for col, ts in colunas_data.items():
        serie_col = pd.to_numeric(
            df_subset[col].astype(str).replace("-", "0"), errors="coerce"
        ).fillna(0)
        valores[ts] = serie_col.sum()
    return pd.Series(valores).sort_index()


def populacao_no_ano(df_pop, col_local_pop, nome_normalizado, ano, anos_pop, nivel=None):
    """População de um País/Região/Estado/Município no ano mais próximo
    disponível. Para 'País': se o arquivo não tiver uma linha explícita de
    'Brasil', cai para a soma das 27 UFs como aproximação do total nacional."""
    ano_usado = ano_mais_proximo(str(ano), anos_pop)
    linha = df_pop[df_pop[col_local_pop].map(normalizar_texto) == nome_normalizado]
    if not linha.empty:
        return pd.to_numeric(linha.iloc[0][ano_usado], errors="coerce")

    if nivel == "País":
        nomes_uf_normalizados = {normalizar_texto(n) for n in NOME_UF_POR_CODIGO.values()}
        linhas_uf = df_pop[df_pop[col_local_pop].map(normalizar_texto).isin(nomes_uf_normalizados)]
        if not linhas_uf.empty:
            return pd.to_numeric(linhas_uf[ano_usado], errors="coerce").sum()

    return np.nan


def nomes_disponiveis_populacao(df_pop, col_local_pop, limite=80):
    """Lista (para diagnóstico) os nomes de localidade encontrados na
    primeira coluna do arquivo de população."""
    nomes = sorted(df_pop[col_local_pop].dropna().astype(str).unique().tolist())
    return nomes[:limite]


def aplicar_taxa(serie_casos: pd.Series, df_pop, col_local_pop, nome_normalizado,
                  anos_pop, base_taxa, nivel=None) -> pd.Series:
    """Converte a série de casos absolutos em taxa por `base_taxa` habitantes,
    usando a população do ano de cada observação."""
    valores = {}
    for ts, casos in serie_casos.items():
        pop = populacao_no_ano(df_pop, col_local_pop, nome_normalizado, ts.year, anos_pop, nivel=nivel)
        valores[ts] = (casos / pop) * base_taxa if pop and pop > 0 else np.nan
    return pd.Series(valores)


# ----------------------------------------------------------------------
# FUNÇÕES AUXILIARES — estatística de séries temporais (STL / Mann-Kendall)
# ----------------------------------------------------------------------
def rodar_stl(serie: pd.Series, periodo: int = PERIODO_SAZONAL):
    """Decompõe a série em Tendência + Sazonalidade + Resíduo (loess/STL) e
    calcula a Força da Tendência (F_T) e a Força da Sazonalidade (F_S),
    conforme Hyndman & Athanasopoulos:
        F_T = max(0, 1 - Var(R) / Var(T + R))
        F_S = max(0, 1 - Var(R) / Var(S + R))
    """
    stl = STL(serie, period=periodo, robust=True)
    resultado = stl.fit()

    var_resid = np.var(resultado.resid)
    var_tend_resid = np.var(resultado.trend + resultado.resid)
    var_saz_resid = np.var(resultado.seasonal + resultado.resid)

    f_tendencia = max(0.0, 1 - var_resid / var_tend_resid) if var_tend_resid > 0 else 0.0
    f_sazonalidade = max(0.0, 1 - var_resid / var_saz_resid) if var_saz_resid > 0 else 0.0

    return resultado, f_tendencia, f_sazonalidade


TRADUCAO_TENDENCIA_MK = {
    "increasing": "crescente",
    "decreasing": "decrescente",
    "no trend": "sem tendência",
}


def rodar_mann_kendall(serie: pd.Series):
    """Teste de Mann-Kendall modificado por Hamed e Rao (1998): não
    paramétrico e com correção para autocorrelação serial."""
    return mk.hamed_rao_modification_test(serie.values)


def formatar_pvalor(p: float) -> str:
    """Formata o valor-p de forma legível (notação científica explicada)."""
    if p < 0.0001:
        return f"{p:.2e} (extremamente pequeno — evidência muito forte contra H0)"
    return f"{p:.4f}"


def calcular_indice_sazonal_padronizado(serie: pd.Series) -> pd.DataFrame:
    """Índice Sazonal Padronizado (ISP): cada observação mensal é dividida
    pela média geral de TODA a série e multiplicada por 100."""
    media_geral = serie.mean()
    isp = (serie / media_geral) * 100
    df = pd.DataFrame({"data": serie.index, "valor": serie.values, "isp": isp.values})
    df["mes_num"] = df["data"].dt.month
    df["ano"] = df["data"].dt.year
    return df


def interpretar_indice_sazonal(df_isp: pd.DataFrame) -> pd.DataFrame:
    """Para cada mês, aplica a regra de bolso:
      - 1º quartil > 100%  -> efeito sazonal POSITIVO (significativo)
      - 3º quartil < 100%  -> efeito sazonal NEGATIVO (significativo)
      - caso contrário     -> sem efeito sazonal claro nesse mês
    """
    linhas = []
    for mes_num in range(1, 13):
        valores = df_isp.loc[df_isp["mes_num"] == mes_num, "isp"]
        if valores.empty:
            continue
        q1, mediana, q3 = valores.quantile([0.25, 0.5, 0.75])
        media = valores.mean()
        if q1 > 100:
            interpretacao = "Efeito sazonal positivo (significativo)"
        elif q3 < 100:
            interpretacao = "Efeito sazonal negativo (significativo)"
        else:
            interpretacao = "Sem efeito sazonal claro"
        linhas.append({
            "Mês": MESES_PT_INV[mes_num],
            "Média ISP (%)": round(media, 1),
            "Mediana ISP (%)": round(mediana, 1),
            "1º Quartil (%)": round(q1, 1),
            "3º Quartil (%)": round(q3, 1),
            "n (anos)": int(valores.shape[0]),
            "Interpretação": interpretacao,
        })
    return pd.DataFrame(linhas)


# ----------------------------------------------------------------------
# FUNÇÕES AUXILIARES — estatística espacial (Moran / LISA)
# ----------------------------------------------------------------------
def construir_pesos_espaciais(gdf: gpd.GeoDataFrame, tipo: str, k: int = 5):
    """Constrói a matriz de vizinhança (W) para o subconjunto de polígonos
    já filtrado (Brasil inteiro ou só um Estado). Ilhas (polígonos sem
    vizinho) são silenciadas, não geram erro."""
    gdf = gdf.reset_index(drop=True)
    if tipo == "Queen":
        w = libpysal.weights.Queen.from_dataframe(gdf, use_index=False, silence_warnings=True)
    elif tipo == "Rook":
        w = libpysal.weights.Rook.from_dataframe(gdf, use_index=False, silence_warnings=True)
    else:  # KNN
        w = libpysal.weights.KNN.from_dataframe(gdf, k=k, use_index=False)
    w.transform = "r"  # padronização por linha (row-standardized), padrão GeoDa
    return w


def rotulo_forca_moran(i: float) -> str:
    forca = "fraca" if abs(i) < 0.3 else ("moderada" if abs(i) < 0.6 else "forte")
    sinal = "positiva" if i > 0 else "negativa"
    return f"{sinal} {forca}"


def classificar_quadrante_lisa(moran_local: Moran_Local, alpha: float = ALPHA) -> pd.Series:
    """Traduz Moran_Local.q (1=HH, 2=LH, 3=LL, 4=HL) + significância (p_sim)
    em rótulos de cluster, igual à convenção usada no GeoDa/artigos."""
    mapa_quadrante = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
    rotulos = pd.Series(
        [mapa_quadrante[q] for q in moran_local.q], index=range(len(moran_local.q))
    )
    significativo = moran_local.p_sim < alpha
    rotulos[~significativo] = "ns"
    return rotulos


CORES_CLUSTER = {
    "HH": "#d7191c",   # vermelho — Alto-Alto (Cluster Quente)
    "LL": "#2c7bb6",   # azul — Baixo-Baixo (Cluster Frio)
    "HL": "#fdae61",   # laranja — Alto-Baixo (Outlier)
    "LH": "#abd9e9",   # azul claro — Baixo-Alto (Outlier)
    "ns": "#d9d9d9",   # cinza — Não significativo
}
NOMES_CLUSTER = {
    "HH": "Alto-Alto (Cluster Quente)",
    "LL": "Baixo-Baixo (Cluster Frio)",
    "HL": "Alto-Baixo (Outlier)",
    "LH": "Baixo-Alto (Outlier)",
    "ns": "Não significativo",
}


# ----------------------------------------------------------------------
# ETAPA 1: UPLOAD (SIH: Ano/Mês ou ano puro | SINAN: só ano puro)
# ----------------------------------------------------------------------
if st.session_state.etapa == "Upload":
    st.header("Passo 1: Upload e Conferência")

    if not os.path.exists(ARQUIVO_POPULACAO):
        st.error(
            f"Não encontrei '{ARQUIVO_POPULACAO}' no repositório. "
            "Salve o CSV de projeção de população do IBGE/TabNet com esse nome, "
            "junto do mapa.json."
        )
        st.stop()
    if not os.path.exists(ARQUIVO_MAPA):
        st.error(f"Não encontrei '{ARQUIVO_MAPA}' no repositório.")
        st.stop()

    st.subheader("📄 CSV de casos (SIH ou SINAN) por município/competência")

    fonte_dados = st.radio(
        "Fonte dos dados:",
        ["SIH", "SINAN"],
        horizontal=True,
        help=(
            "SIH: aceita competência Ano/Mês (ex.: '2008/Jan') OU ano puro. "
            "SINAN: o TabNet do SINAN só exporta por Mês (agrupando todos os anos "
            "do período junto, sem separar por ano) ou por Ano puro — nunca por "
            "Ano/Mês de verdade. Por isso, para SINAN, exporte por ANO e o "
            "arquivo é aceito apenas nesse formato (sem Séries Temporais)."
        ),
        key="fonte_dados",
    )

    if fonte_dados == "SIH":
        st.caption(
            "Suba a tabela por **mês + ano** (competência), ex.: '2008/Jan', 'Jan/2008' "
            "ou '2008/01' — igual já é feito nas Séries Temporais. Esses mesmos dados "
            "mensais são agregados automaticamente por ANO para os módulos de Mapa "
            "Coroplético e Espacial (Moran/LISA), que trabalham só em nível anual. "
            "Também funciona com exportações já filtradas por Estado/Região no TabNet, "
            "e com arquivos antigos que só tenham colunas de ano puro (nesse caso, "
            "apenas os módulos Mapa e Espacial ficam disponíveis)."
        )
    else:
        st.caption(
            "Suba a tabela exportada por **ANO** (ex.: '2016', '2017'...), no formato "
            "'Casos confirmados por Município de residência e Ano Diagnóstico' do "
            "TabNet do SINAN. Como o SINAN não exporta Ano/Mês de verdade, apenas os "
            "módulos de **Mapas Coropléticos** e **Espacial (Moran/LISA)** ficam "
            "disponíveis — **Séries Temporais** não se aplica a esse tipo de exportação."
        )

    with st.expander("Opções de leitura (mude só se o arquivo vier de outra fonte)", expanded=False):
        sih_sep = st.selectbox("Separador:", [";", ",", "\t"], index=0, key="sih_sep")
        sih_enc = st.selectbox("Encoding:", ["latin-1", "utf-8"], index=0, key="sih_enc")
        pop_enc_override = st.selectbox(
            "Forçar encoding da população (só mude se o diagnóstico mostrar nomes quebrados):",
            ["Automático", "utf-8-sig", "latin-1"], index=0, key="pop_enc_override",
        )
    csv_sih = st.file_uploader(f"Suba o CSV do {fonte_dados} (TabNet)", type=["csv"], key="csv_sih")

    if csv_sih:
        try:
            conteudo_sih = csv_sih.getvalue()
            skip_detectado = detectar_linha_cabecalho(conteudo_sih, sih_sep, sih_enc)
            df_sih = ler_csv(io.BytesIO(conteudo_sih), sih_sep, skip_detectado, sih_enc)
            df_sih.columns = df_sih.columns.str.strip()

            enc_manual = None if pop_enc_override == "Automático" else pop_enc_override
            df_pop, pop_encoding_usado = carregar_populacao(enc_manual)
            st.caption(f"✅ População lida com encoding: **{pop_encoding_usado}**")
            gdf = carregar_mapa()

            colunas_data, granularidade_mensal = identificar_colunas_data(df_sih, fonte=fonte_dados)
            if not colunas_data:
                msg_formato = (
                    "competência (Ano/Mês) nem de ano puro"
                    if fonte_dados == "SIH" else "de ano puro"
                )
                st.error(
                    f"Não encontrei nenhuma coluna {msg_formato} neste arquivo. "
                    "Confira o separador/encoding nas opções de leitura acima "
                    + ("(para SINAN, exporte por Ano, não por Mês)." if fonte_dados == "SINAN" else ".")
                )
                st.stop()

            datas_ordenadas = sorted(colunas_data.values())
            if granularidade_mensal:
                st.success(
                    f"Arquivo carregado! Cabeçalho detectado na linha {skip_detectado + 1}. "
                    f"{len(colunas_data)} competências mensais encontradas, de "
                    f"{datas_ordenadas[0]:%m/%Y} a {datas_ordenadas[-1]:%m/%Y}."
                )
            else:
                st.warning(
                    f"Arquivo carregado! Cabeçalho detectado na linha {skip_detectado + 1}. "
                    f"Encontrei apenas colunas de **ano puro** ({datas_ordenadas[0].year} a "
                    f"{datas_ordenadas[-1].year}), sem granularidade mensal. Por isso, o módulo "
                    "de **Séries Temporais** (STL/Mann-Kendall/Índice Sazonal) não ficará "
                    "disponível — apenas **Mapas Coropléticos** e **Espacial (Moran/LISA)**."
                )

            df_sih["id_join"] = limpar_codigo(df_sih[df_sih.columns[0]])
            df_sih["codigo_uf"] = df_sih["id_join"].str[:2]
            nao_mapeados = int((~df_sih["codigo_uf"].isin(SIGLA_POR_CODIGO)).sum())
            if nao_mapeados > 0:
                st.info(
                    f"{nao_mapeados} linha(s) não foram reconhecidas como município "
                    "válido (ex.: 'MUNICÍPIO IGNORADO') e ficarão de fora das agregações "
                    "por Estado/Região, mas continuam entrando no total do País."
                )

            gdf_check = gdf.copy()
            gdf_check["id_join"] = limpar_codigo(gdf_check["CD_MUN"])
            ufs_no_arquivo = sorted(
                gdf_check.merge(df_sih[["id_join"]], on="id_join", how="inner")["NM_UF"].dropna().unique().tolist()
            )

            df_sih_anual, anos_anuais = agregar_para_anual(df_sih, colunas_data)

            c1, c2 = st.columns(2)
            with c1:
                st.caption("SIH (casos por competência) — enviado agora")
                st.dataframe(df_sih.head())
            with c2:
                st.caption("População (IBGE) — carregada do repositório")
                st.dataframe(df_pop.head(8))

            if st.button("Confirmar e Prosseguir ➡️"):
                st.session_state.df_sih = df_sih
                st.session_state.df_sih_anual = df_sih_anual
                st.session_state.anos_anuais = anos_anuais
                st.session_state.colunas_data = colunas_data
                st.session_state.granularidade_mensal = granularidade_mensal
                st.session_state.df_pop = df_pop
                st.session_state.pop_encoding_usado = pop_encoding_usado
                st.session_state.gdf = gdf
                st.session_state.ufs_no_arquivo = ufs_no_arquivo
                st.session_state.etapa = "VisaoGeral"
                st.rerun()
        except Exception as e:
            st.error(f"Erro ao carregar o arquivo: {e}")
    else:
        st.info("Suba o CSV do SIH para continuar. A população do IBGE já está carregada automaticamente.")


# ----------------------------------------------------------------------
# ETAPA 2: VISÃO GERAL (tabela do SIH + tabela da população processada)
# ----------------------------------------------------------------------
elif st.session_state.etapa == "VisaoGeral":
    if not all(k in st.session_state for k in ("df_sih", "df_pop", "gdf", "colunas_data")):
        st.warning("Sua sessão foi reiniciada e o arquivo do SIH precisa ser enviado novamente.")
        st.session_state.etapa = "Upload"
        st.rerun()

    df_sih = st.session_state.df_sih
    df_sih_anual = st.session_state.df_sih_anual
    df_pop = st.session_state.df_pop
    granularidade_mensal = st.session_state.granularidade_mensal

    st.header("Passo 2: Visão Geral dos Dados Carregados")

    st.subheader("📄 Tabela do SIH (por competência mês/ano)")
    st.caption(
        "Esta é a tabela enviada no upload, uma linha por município e uma coluna "
        "por competência. É a base usada integralmente pelo módulo de Séries "
        "Temporais, e agregada por ano para os módulos de Mapa e Espacial."
    )
    st.dataframe(df_sih.drop(columns=["id_join", "codigo_uf"], errors="ignore"), use_container_width=True)

    st.subheader("🧮 Tabela do SIH agregada por ano (base dos módulos espaciais)")
    st.caption("Soma dos casos de cada município dentro de cada ano civil.")
    st.dataframe(df_sih_anual.drop(columns=["id_join", "codigo_uf"], errors="ignore"), use_container_width=True)

    st.subheader("👥 Tabela de População IBGE (processada do repositório)")
    st.caption(
        "Carregada automaticamente de `populacao_ibge.csv`, sem necessidade de upload — "
        "exibida aqui só para conferir que o encoding e o cabeçalho foram processados "
        f"corretamente (encoding usado: **{st.session_state.get('pop_encoding_usado', '?')}**)."
    )
    st.dataframe(df_pop, use_container_width=True)

    st.divider()
    st.subheader("Para onde ir agora?")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🗺️ Mapas Coropléticos", use_container_width=True):
            st.session_state.etapa = "Mapa"
            st.rerun()
    with col2:
        if st.button("🌐 Espacial (Moran / LISA)", use_container_width=True):
            st.session_state.etapa = "Espacial"
            st.rerun()
    with col3:
        if granularidade_mensal:
            if st.button("📈 Séries Temporais", use_container_width=True):
                st.session_state.etapa = "Temporal"
                st.rerun()
        else:
            st.button("📈 Séries Temporais (indisponível)", use_container_width=True, disabled=True)
            st.caption("Precisa de dados mensais (Ano/Mês); este arquivo só tem colunas de ano puro.")

    if st.button("⬅️ Trocar arquivo do SIH"):
        st.session_state.etapa = "Upload"
        st.rerun()


# ----------------------------------------------------------------------
# ETAPA 3: MAPAS COROPLÉTICOS (sempre por ANO)
# ----------------------------------------------------------------------
elif st.session_state.etapa == "Mapa":
    if not all(k in st.session_state for k in ("df_sih_anual", "df_pop", "gdf")):
        st.warning("Sua sessão foi reiniciada e o arquivo do SIH precisa ser enviado novamente.")
        st.session_state.etapa = "Upload"
        st.rerun()

    df_sih = st.session_state.df_sih_anual   # tabela agregada por ano
    df_pop = st.session_state.df_pop
    gdf = st.session_state.gdf
    anos_anuais = st.session_state.anos_anuais

    COL_MUN_SIH = df_sih.columns[0]
    COL_LOCAL_POP = df_pop.columns[0]
    COL_REGIAO_MAPA = "NM_REGIAO" if "NM_REGIAO" in gdf.columns else gdf.columns[0]
    COL_UF_MAPA = "NM_UF" if "NM_UF" in gdf.columns else gdf.columns[1]

    with st.sidebar:
        st.header("⚙️ Configurações — Mapa Coroplético")
        if st.button("⬅️ Voltar à Visão Geral"):
            st.session_state.etapa = "VisaoGeral"
            st.rerun()

        st.subheader("Escala geográfica")
        ufs_no_arquivo = st.session_state.get("ufs_no_arquivo", [])
        if 0 < len(ufs_no_arquivo) < 27:
            st.caption(f"O arquivo enviado parece conter apenas: {', '.join(ufs_no_arquivo)}.")
            default_escala = "Estado(s) específico(s)"
        else:
            default_escala = "O Brasil inteiro"
        opcoes_escala = ["O Brasil inteiro", "Região(ões) específica(s)", "Estado(s) específico(s)"]
        tipo_filtro = st.radio("O mapa deve mostrar:", opcoes_escala, index=opcoes_escala.index(default_escala))
        regioes, estados = [], []
        if tipo_filtro == "Região(ões) específica(s)":
            regioes = st.multiselect("Escolha a(s) Região(ões):", sorted(gdf[COL_REGIAO_MAPA].dropna().unique()))
        elif tipo_filtro == "Estado(s) específico(s)":
            opcoes_uf = sorted(gdf[COL_UF_MAPA].dropna().unique())
            padrao_uf = [u for u in ufs_no_arquivo if u in opcoes_uf]
            estados = st.multiselect("Escolha o(s) Estado(s):", opcoes_uf, default=padrao_uf)

        st.subheader("Nível de agregação do mapa")
        nivel_agregacao = st.radio(
            "Desenhar o mapa por:",
            ["Município", "Estado (UF)", "Região"],
            help=(
                "Município: mais detalhado, mas usa a população do estado como "
                "aproximação (pois só temos população por Estado/Região). "
                "Estado ou Região: menos detalhado, mas com população exata."
            ),
        )

        st.subheader("Dados")
        with st.expander("🔍 Diagnóstico da população carregada (clique se algo parecer errado)", expanded=False):
            st.write(f"Encoding usado na leitura: **{st.session_state.get('pop_encoding_usado', '?')}**")
            st.write(f"Total de linhas carregadas do CSV de população: {len(df_pop)}")
            st.write(f"Nome da 1ª coluna (local): `{COL_LOCAL_POP}`")
            st.write("Valores únicos da coluna de local (bruto, antes de normalizar):")
            st.code(repr(sorted(df_pop[COL_LOCAL_POP].dropna().unique().tolist())))
            st.write("Mesmos valores, depois de normalizados:")
            st.code(repr(sorted(set(df_pop[COL_LOCAL_POP].dropna().map(normalizar_texto)))))
            st.write("Nomes de UF do mapa, depois de normalizados:")
            st.code(repr(sorted(set(gdf[COL_UF_MAPA].dropna().map(normalizar_texto)))))

        ano = st.selectbox("Ano:", anos_anuais)

        anos_pop = colunas_de_ano(df_pop)
        ano_pop_usado = ano_mais_proximo(ano, anos_pop)
        if str(ano_pop_usado) != str(ano):
            st.info(f"Não há projeção de população para {ano}; usando {ano_pop_usado} (ano mais próximo disponível).")

        st.subheader("Métrica epidemiológica")
        metrica = st.radio(
            "Métrica:",
            ["Quantidade Absoluta", "Taxa Bruta", "Taxa Bayesiana"],
            help=(
                "Absoluta: número de casos, sem considerar população. "
                "Taxa Bruta: casos/população, direta — instável em áreas pequenas. "
                "Taxa Bayesiana: mesma taxa, mas suavizada estatisticamente para "
                "corrigir oscilações artificiais em áreas com população pequena."
            ),
        )
        base_taxa = st.selectbox("Taxa por quantos habitantes:", [1000, 10000, 100000], index=2,
                                  disabled=(metrica == "Quantidade Absoluta"))

        st.subheader("Classificação das cores")
        modo_classes = st.radio(
            "Como dividir as faixas de cor:",
            ["Quantis automáticos", "Agrupamento Natural", "Personalizado"],
            help=(
                "Quantis automáticos: cada faixa tem a mesma quantidade de áreas — bom "
                "para ver a distribuição geral. "
                "Agrupamento Natural (método Jenks): agrupa áreas com valores parecidos, "
                "criando faixas mais fiéis aos dados reais. "
                "Personalizado: você define os limites de cada faixa manualmente."
            ),
        )
        n_classes = st.slider("Nº de faixas de cor:", 3, 9, 5, disabled=(modo_classes == "Personalizado"))
        rupturas_manual = st.text_input(
            "Limites das faixas (separados por vírgula, ex: 0,10,50,100,500):",
            disabled=(modo_classes != "Personalizado")
        )
        destacar_zero = st.checkbox(
            "Tratar 'sem casos' (valor 0) como cor separada", value=True,
            help=(
                "Recomendado para doenças raras: se a maioria dos municípios tem 0 casos, "
                "os quantis colapsam e o mapa fica quase todo de uma cor só. Isso separa os "
                "municípios com 0 (cor cinza clara fixa) e aplica a escala de cores apenas "
                "onde há casos de fato, deixando a variação visível."
            ),
        )

        st.subheader("Aparência")
        titulo_mapa = st.text_input("Título do mapa:", value=f"{metrica} — {ano}")
        paleta = st.selectbox("Paleta de cores:", ["OrRd", "YlGnBu", "PuBu", "YlOrRd", "RdPu", "viridis", "magma"])
        espessura_borda = st.slider("Espessura da borda:", 0.0, 2.0, 0.2, step=0.1)
        largura = st.slider("Largura do gráfico (pol.):", 4, 20, 10)
        altura = st.slider("Altura do gráfico (pol.):", 4, 20, 8)

        posicao_legenda, fonte_legenda, desloc_x_legenda, desloc_y_legenda = controles_posicao_legenda(
            "mapa", indice_padrao=0
        )

    gerar = st.button("🗺️ Gerar Mapa")

    if gerar:
        try:
            # ---------- Filtro geográfico ----------
            mapa_plot = gdf.copy()
            if tipo_filtro == "Região(ões) específica(s)" and regioes:
                mapa_plot = mapa_plot[mapa_plot[COL_REGIAO_MAPA].isin(regioes)]
            elif tipo_filtro == "Estado(s) específico(s)" and estados:
                mapa_plot = mapa_plot[mapa_plot[COL_UF_MAPA].isin(estados)]

            if mapa_plot.empty:
                st.error("Nenhuma área corresponde ao filtro selecionado.")
                st.stop()

            # ---------- Junção dos casos (sempre em nível de município) ----------
            df_sih_local = df_sih.copy()
            mapa_plot = mapa_plot.copy()

            mapa_plot["id_join"] = limpar_codigo(mapa_plot["CD_MUN"])

            mapa_dados = mapa_plot.merge(
                df_sih_local[["id_join", ano]], on="id_join", how="left"
            )
            mapa_dados["casos"] = pd.to_numeric(mapa_dados[ano], errors="coerce").fillna(0)

            # ---------- Dicionário de população por nome (Região/UF) ----------
            pop_lookup = dict(zip(
                df_pop[COL_LOCAL_POP].map(normalizar_texto),
                pd.to_numeric(df_pop[ano_pop_usado], errors="coerce")
            ))

            def buscar_populacao(nome):
                return pop_lookup.get(normalizar_texto(nome), np.nan)

            # ---------- Nível de agregação ----------
            if nivel_agregacao == "Município":
                mapa_dados["populacao"] = mapa_dados[COL_UF_MAPA].map(buscar_populacao)
                nome_col_area = "NM_MUN" if "NM_MUN" in mapa_dados.columns else "id_join"

            elif nivel_agregacao == "Estado (UF)":
                mapa_dados = mapa_dados.dissolve(by=COL_UF_MAPA, aggfunc={"casos": "sum"}).reset_index()
                mapa_dados["populacao"] = mapa_dados[COL_UF_MAPA].map(buscar_populacao)
                nome_col_area = COL_UF_MAPA

            else:  # Região
                mapa_dados = mapa_dados.dissolve(by=COL_REGIAO_MAPA, aggfunc={"casos": "sum"}).reset_index()
                mapa_dados["populacao"] = mapa_dados[COL_REGIAO_MAPA].map(buscar_populacao)
                nome_col_area = COL_REGIAO_MAPA

            areas_sem_pop = int(
                ((mapa_dados["populacao"].isna()) | (mapa_dados["populacao"] <= 0)).sum()
            )
            if areas_sem_pop > 0 and metrica != "Quantidade Absoluta":
                mask_sem_pop = (mapa_dados["populacao"].isna()) | (mapa_dados["populacao"] <= 0)
                coluna_diagnostico = COL_UF_MAPA if nivel_agregacao == "Município" else nome_col_area
                nomes_sem_pop = sorted(mapa_dados.loc[mask_sem_pop, coluna_diagnostico].dropna().unique().tolist())
                st.warning(
                    f"{areas_sem_pop} área(s) não tiveram população encontrada por nome. "
                    f"Estado(s)/Região(ões) envolvidos: {', '.join(nomes_sem_pop) if nomes_sem_pop else '(não identificado)'}."
                )

            # ---------- Cálculo da métrica ----------
            if metrica == "Quantidade Absoluta":
                mapa_dados["valor_mapa"] = mapa_dados["casos"]
                fmt_legenda = ",.0f"
            elif metrica == "Taxa Bruta":
                mapa_dados["valor_mapa"] = taxa_bruta(mapa_dados["casos"], mapa_dados["populacao"], base_taxa)
                fmt_legenda = ",.2f"
            else:  # Taxa Bayesiana
                mapa_dados["valor_mapa"] = taxa_bayesiana(mapa_dados["casos"], mapa_dados["populacao"], base_taxa)
                fmt_legenda = ",.2f"

            sem_dado = mapa_dados["valor_mapa"].isna()

            # ---------- Plot ----------
            fig, ax = plt.subplots(figsize=(largura, altura))
            dados_validos = mapa_dados.loc[~sem_dado, "valor_mapa"]

            def montar_scheme_kwargs(valores_para_classificar):
                if modo_classes == "Quantis automáticos":
                    return {"scheme": "Quantiles", "k": min(n_classes, max(1, valores_para_classificar.nunique()))}
                elif modo_classes == "Agrupamento Natural":
                    return {"scheme": "NaturalBreaks", "k": min(n_classes, max(1, valores_para_classificar.nunique()))}
                else:
                    try:
                        bins = sorted(float(x.strip()) for x in rupturas_manual.split(",") if x.strip() != "")
                        if len(bins) < 2:
                            raise ValueError("Informe pelo menos 2 limites.")
                        return {"scheme": "UserDefined", "classification_kwds": {"bins": bins}}
                    except Exception as e:
                        st.error(f"Limites personalizados inválidos: {e}")
                        st.stop()

            if destacar_zero:
                mask_sem_dado = mapa_dados["valor_mapa"].isna()
                mask_zero = (mapa_dados["valor_mapa"] == 0) & (~mask_sem_dado)
                mask_com_valor = (mapa_dados["valor_mapa"] > 0) & (~mask_sem_dado)

                dados_sem_info = mapa_dados[mask_sem_dado]
                dados_zero = mapa_dados[mask_zero]
                dados_com_valor = mapa_dados[mask_com_valor]

                if not dados_sem_info.empty:
                    dados_sem_info.plot(ax=ax, color="lightgrey", edgecolor="black",
                                         linewidth=espessura_borda, hatch="///")
                if not dados_zero.empty:
                    dados_zero.plot(ax=ax, color="#f2f2f2", edgecolor="black", linewidth=espessura_borda)

                if not dados_com_valor.empty:
                    plot_kwargs = dict(
                        column="valor_mapa", ax=ax, legend=True, cmap=paleta,
                        edgecolor="black", linewidth=espessura_borda,
                    )
                    plot_kwargs.update(montar_scheme_kwargs(dados_com_valor["valor_mapa"]))
                    dados_com_valor.plot(**plot_kwargs)
                else:
                    st.info("Nenhuma área com valor acima de 0 para colorir por faixas.")

                legenda_existente = ax.get_legend()
                handles = list(getattr(legenda_existente, "legend_handles", None)
                               or getattr(legenda_existente, "legendHandles", [])) if legenda_existente else []
                labels = [t.get_text() for t in legenda_existente.get_texts()] if legenda_existente else []
                if not dados_zero.empty:
                    handles.append(mpatches.Patch(facecolor="#f2f2f2", edgecolor="black", linewidth=0.5))
                    labels.append("Sem casos (0)")
                if not dados_sem_info.empty:
                    handles.append(mpatches.Patch(facecolor="lightgrey", edgecolor="black", hatch="///", linewidth=0.5))
                    labels.append("Sem dado")
                if handles:
                    ax.legend(
                        handles=handles, labels=labels,
                        title=metrica if metrica == "Quantidade Absoluta" else f"{metrica} (por {base_taxa:,} hab.)",
                        **kwargs_legenda(posicao_legenda, fonte_legenda, desloc_x_legenda, desloc_y_legenda),
                    )
            else:
                plot_kwargs = dict(
                    column="valor_mapa",
                    ax=ax,
                    legend=True,
                    cmap=paleta,
                    edgecolor="black",
                    linewidth=espessura_borda,
                    missing_kwds={"color": "lightgrey", "edgecolor": "black",
                                  "linewidth": espessura_borda, "hatch": "///", "label": "Sem dado"},
                )
                plot_kwargs.update(montar_scheme_kwargs(dados_validos))
                mapa_dados.plot(**plot_kwargs)

                legenda = ax.get_legend()
                if legenda is not None:
                    # Recria a legenda (que o geopandas posiciona automaticamente
                    # em "best") na posição/tamanho escolhidos pelo usuário.
                    handles = list(getattr(legenda, "legend_handles", None)
                                   or getattr(legenda, "legendHandles", []))
                    labels = [t.get_text() for t in legenda.get_texts()]
                    legenda.remove()
                    ax.legend(
                        handles=handles, labels=labels,
                        title=metrica if metrica == "Quantidade Absoluta" else f"{metrica} (por {base_taxa:,} hab.)",
                        **kwargs_legenda(posicao_legenda, fonte_legenda, desloc_x_legenda, desloc_y_legenda),
                    )

            ax.set_title(titulo_mapa, fontsize=14, fontweight="bold")
            ax.set_axis_off()

            # bbox_inches="tight" evita que a legenda seja cortada quando o
            # usuário escolhe uma posição "Fora do mapa" (fora da área dos eixos).
            st.pyplot(fig, bbox_inches="tight")

            with st.expander("📊 Estatísticas do mapa"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Áreas no mapa", len(mapa_dados))
                c2.metric("Áreas sem valor calculado", int(sem_dado.sum()))
                c3.metric("Áreas sem população encontrada", areas_sem_pop)
                if len(dados_validos) > 0:
                    st.write(
                        f"Mínimo: {dados_validos.min():{fmt_legenda}} | "
                        f"Máximo: {dados_validos.max():{fmt_legenda}} | "
                        f"Média: {dados_validos.mean():{fmt_legenda}}"
                    )
                colunas_tabela = [c for c in [nome_col_area, "casos", "populacao", "valor_mapa"] if c in mapa_dados.columns]
                st.dataframe(mapa_dados[colunas_tabela])

        except Exception as e:
            st.error(f"Erro ao gerar o mapa: {e}")


# ----------------------------------------------------------------------
# ETAPA 4: ANÁLISE ESPACIAL — MORAN & LISA (sempre por ANO)
# ----------------------------------------------------------------------
elif st.session_state.etapa == "Espacial":
    if not all(k in st.session_state for k in ("df_sih_anual", "df_pop", "gdf")):
        st.warning("Sua sessão foi reiniciada e o arquivo do SIH precisa ser enviado novamente.")
        st.session_state.etapa = "Upload"
        st.rerun()

    df_sih = st.session_state.df_sih_anual   # tabela agregada por ano
    df_pop = st.session_state.df_pop
    gdf = st.session_state.gdf
    anos_anuais = st.session_state.anos_anuais

    COL_UF_MAPA = "NM_UF" if "NM_UF" in gdf.columns else gdf.columns[1]
    COL_MUN_MAPA = "NM_MUN" if "NM_MUN" in gdf.columns else gdf.columns[0]
    COL_LOCAL_POP = df_pop.columns[0]

    with st.sidebar:
        st.header("⚙️ Configurações — Espacial")

        if st.button("⬅️ Voltar à Visão Geral"):
            st.session_state.etapa = "VisaoGeral"
            st.rerun()

        st.subheader("Escolha o Tipo de Análise")
        tipo_analise = st.radio(
            "Teste espacial:",
            ["Índice de Moran", "LISA"],
        )

        st.subheader("Configurações Gerais")
        ano = st.selectbox("Ano:", anos_anuais)

        anos_pop = colunas_de_ano(df_pop)
        ano_pop_usado = ano_mais_proximo(ano, anos_pop)
        if str(ano_pop_usado) != str(ano):
            st.info(f"Não há projeção de população para {ano}; usando {ano_pop_usado} (ano mais próximo).")

        opcoes_metrica = ["Quantidade Absoluta", "Taxa Bruta"]
        st.caption(
            "Índice de Moran e LISA não funcionam com Taxa Bayesiana aqui — "
            "a maioria dos artigos usa Taxa Bruta (ou Quantidade Absoluta) "
            "nos testes de autocorrelação."
        )
        metrica = st.selectbox("Escolha o Tipo de Métrica:", opcoes_metrica)
        base_taxa = st.selectbox("Taxa por quantos habitantes:", [1000, 10000, 100000], index=2,
                                  disabled=(metrica == "Quantidade Absoluta"))

        st.subheader("Configuração do Filtro de Observação")
        opcoes_uf = sorted(gdf[COL_UF_MAPA].dropna().unique())
        filtro_obs = st.selectbox("Filtro Observacional:", ["Brasil"] + opcoes_uf)
        if filtro_obs != "Brasil":
            st.caption(
                "Filtrando por Estado: só os municípios desse Estado entram na análise "
                "(municípios vizinhos de outros Estados não são considerados)."
            )

        st.subheader("Configuração para os Testes de Autocorrelação")
        tipo_vizinhanca = st.selectbox("Tipo de Vizinhança para os Testes Espaciais:", ["KNN", "Queen", "Rook"])
        k_vizinhos = st.slider("Número de Vizinhos Mais Próximos (K) para os Testes:", 1, 20, 5,
                                disabled=(tipo_vizinhanca != "KNN"))

        posicao_legenda_lisa, fonte_legenda_lisa, desloc_x_legenda_lisa, desloc_y_legenda_lisa = (
            controles_posicao_legenda("espacial", indice_padrao=0)
        )

        executar = st.button(f"▶️ Executar {tipo_analise}")

    with st.expander("🔍 Diagnóstico da população carregada (clique se algo parecer errado)", expanded=False):
        st.write(f"Encoding usado na leitura: **{st.session_state.get('pop_encoding_usado', '?')}**")
        st.write(f"Total de linhas carregadas do CSV de população: {len(df_pop)}")

    if not executar:
        st.info("Configure os parâmetros na barra lateral e clique em Executar.")
        st.stop()

    try:
        # ---------- Junção casos + mapa (nível município) ----------
        mapa_local = gdf.copy()
        mapa_local["id_join"] = limpar_codigo(mapa_local["CD_MUN"])

        mapa_dados = mapa_local.merge(df_sih[["id_join", ano]], on="id_join", how="left")
        mapa_dados["casos"] = pd.to_numeric(mapa_dados[ano], errors="coerce").fillna(0)

        # ---------- Filtro de observação (Brasil / Estado) ----------
        if filtro_obs != "Brasil":
            mapa_dados = mapa_dados[mapa_dados[COL_UF_MAPA] == filtro_obs].copy()
            if mapa_dados.empty:
                st.error("Nenhum município encontrado para esse Estado.")
                st.stop()

        # ---------- População por município (aproximação via UF) ----------
        pop_lookup = dict(zip(
            df_pop[COL_LOCAL_POP].map(normalizar_texto),
            pd.to_numeric(df_pop[ano_pop_usado], errors="coerce")
        ))

        def buscar_populacao(nome):
            return pop_lookup.get(normalizar_texto(nome), np.nan)

        mapa_dados["populacao"] = mapa_dados[COL_UF_MAPA].map(buscar_populacao)

        areas_sem_pop = int(((mapa_dados["populacao"].isna()) | (mapa_dados["populacao"] <= 0)).sum())
        if areas_sem_pop > 0 and metrica != "Quantidade Absoluta":
            st.warning(f"{areas_sem_pop} município(s) não tiveram população encontrada por nome do Estado.")

        # ---------- Cálculo da métrica ----------
        if metrica == "Quantidade Absoluta":
            mapa_dados["valor"] = mapa_dados["casos"]
            fmt_legenda = ",.0f"
        else:  # Taxa Bruta
            mapa_dados["valor"] = taxa_bruta(mapa_dados["casos"], mapa_dados["populacao"], base_taxa)
            fmt_legenda = ",.2f"

        nome_col_area = COL_MUN_MAPA if COL_MUN_MAPA in mapa_dados.columns else "id_join"

        # ================================================================
        # ÍNDICE DE MORAN (global) e LISA (local) — base comum
        # ================================================================
        mapa_validos = mapa_dados.dropna(subset=["valor"]).reset_index(drop=True)
        n_excluidos_dado = len(mapa_dados) - len(mapa_validos)

        # ---------- Blindagem contra geometrias nulas/vazias/inválidas ----------
        geom_ok = (
            mapa_validos.geometry.notna()
            & ~mapa_validos.geometry.is_empty
            & mapa_validos.geometry.is_valid
        )
        n_excluidos_geom = int((~geom_ok).sum())
        if n_excluidos_geom > 0:
            mapa_validos = mapa_validos.loc[geom_ok].reset_index(drop=True)

        n_excluidos = n_excluidos_dado + n_excluidos_geom
        if n_excluidos_dado > 0:
            st.info(f"{n_excluidos_dado} município(s) ficaram de fora da análise por falta de dado (casos ou população).")
        if n_excluidos_geom > 0:
            st.info(f"{n_excluidos_geom} município(s) ficaram de fora da análise por geometria nula/vazia/inválida no mapa.")

        if len(mapa_validos) < 8:
            st.error("Poucos municípios com dado válido para calcular autocorrelação espacial (mínimo recomendado: 8).")
            st.stop()

        with st.spinner("Calculando matriz de vizinhança..."):
            w = construir_pesos_espaciais(mapa_validos, tipo_vizinhanca, k_vizinhos)

        valores = mapa_validos["valor"].to_numpy(dtype=float)

        # ---------------- ÍNDICE DE MORAN (global) ----------------
        if tipo_analise == "Índice de Moran":
            with st.spinner(f"Rodando {PERMUTACOES} simulações de Monte Carlo..."):
                moran = Moran(valores, w, permutations=PERMUTACOES)

            z = moran.z
            z_lag = libpysal.weights.lag_spatial(w, z)

            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(z, z_lag, s=12, color="grey", alpha=0.6)
            b, a = np.polyfit(z, z_lag, 1)
            xs = np.array([z.min(), z.max()])
            ax.plot(xs, a + b * xs, color="red", linewidth=1.5)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_xlabel("Valores Padronizados")
            ax.set_ylabel("Valores Espaciais Padronizados (lag)")
            ax.set_title(f"Gráfico de Dispersão de Moran — {metrica} ({ano})")
            st.pyplot(fig)

            c1, c2 = st.columns(2)
            c1.metric("Índice de Moran (I)", f"{moran.I:.4f}")
            c2.metric("P-Valor (pseudo, empírico)", f"{moran.p_sim:.4f}")

            if moran.p_sim >= ALPHA:
                st.info(
                    f"P-valor ≥ {ALPHA:.2f}: não há evidência de autocorrelação espacial "
                    "significativa — a distribuição pode ser considerada aleatória no mapa."
                )
            else:
                st.success(
                    f"P-valor < {ALPHA:.2f}: autocorrelação espacial **{rotulo_forca_moran(moran.I)}** "
                    f"(I = {moran.I:.4f})."
                )
            st.caption(
                f"O pseudo P-valor é calculado empiricamente com {PERMUTACOES} permutações de Monte Carlo, "
                f"então o menor valor possível é 1/{PERMUTACOES + 1} ≈ {1/(PERMUTACOES+1):.4f} — "
                "não estranhe se ele não ficar menor que isso."
            )

        # ---------------- LISA (local) ----------------
        else:
            with st.spinner(f"Rodando {PERMUTACOES} simulações de Monte Carlo por polígono..."):
                moran_loc = Moran_Local(valores, w, permutations=PERMUTACOES, seed=42)

            rotulos = classificar_quadrante_lisa(moran_loc, ALPHA)
            mapa_validos = mapa_validos.copy()
            mapa_validos["cluster"] = rotulos.values

            fig, ax = plt.subplots(figsize=(10, 8))

            if n_excluidos > 0:
                ids_validos = set(mapa_validos["id_join"])
                excluidos = mapa_dados[~mapa_dados["id_join"].isin(ids_validos)]
                excluidos_plotaveis = excluidos[
                    excluidos.geometry.notna()
                    & ~excluidos.geometry.is_empty
                    & excluidos.geometry.is_valid
                ]
                if not excluidos_plotaveis.empty:
                    excluidos_plotaveis.plot(ax=ax, color="white", edgecolor="black", linewidth=0.15, hatch="///")

            for cat in ["ns", "LH", "HL", "LL", "HH"]:
                subset = mapa_validos[mapa_validos["cluster"] == cat]
                if not subset.empty:
                    subset.plot(ax=ax, color=CORES_CLUSTER[cat], edgecolor="black", linewidth=0.15)

            handles = [mpatches.Patch(facecolor=CORES_CLUSTER[c], edgecolor="black", label=NOMES_CLUSTER[c])
                       for c in ["HH", "HL", "LH", "LL", "ns"]]
            ax.legend(
                handles=handles, title=metrica,
                **kwargs_legenda(posicao_legenda_lisa, fonte_legenda_lisa, desloc_x_legenda_lisa, desloc_y_legenda_lisa),
            )

            titulo = f"Clusters LISA — {metrica} ({ano})" + (f" — {filtro_obs}" if filtro_obs != "Brasil" else "")
            ax.set_title(titulo, fontsize=14, fontweight="bold")
            ax.set_axis_off()
            st.pyplot(fig, bbox_inches="tight")

            st.subheader("Distribuição de Clusters por Estado")
            tabela = (
                mapa_validos[mapa_validos["cluster"] != "ns"]
                .groupby([COL_UF_MAPA, "cluster"])
                .size()
                .unstack(fill_value=0)
                .reindex(columns=["HH", "LL", "HL", "LH"], fill_value=0)
                .rename(columns=NOMES_CLUSTER)
                .rename_axis("Estado")
                .reset_index()
            )
            st.dataframe(tabela, use_container_width=True)

            st.caption(
                "Como no Índice de Moran, isso continua sendo um ecológico avançado: os mesmos limites "
                "dos ecológicos básicos se aplicam — o cluster reflete onde há mais casos "
                "**detectados/notificados**, não necessariamente onde a doença é mais prevalente."
            )

    except Exception as e:
        st.error(f"Erro ao gerar a análise: {e}")


# ----------------------------------------------------------------------
# ETAPA 5: SÉRIES TEMPORAIS — STL / Mann-Kendall / Índice Sazonal / Tendência
# (exige granularidade mensal real; bloqueada se o arquivo só tiver ano puro)
# ----------------------------------------------------------------------
elif st.session_state.etapa == "Temporal":
    if not all(k in st.session_state for k in ("df_sih", "df_pop", "colunas_data")):
        st.warning("Sua sessão foi reiniciada e o arquivo do SIH precisa ser enviado novamente.")
        st.session_state.etapa = "Upload"
        st.rerun()

    if not st.session_state.get("granularidade_mensal", False):
        st.warning(
            "Este arquivo só tem colunas de ano puro (sem competência mês/ano), então o "
            "módulo de Séries Temporais não está disponível. Use Mapas Coropléticos ou "
            "Espacial (Moran/LISA) na Visão Geral."
        )
        if st.button("⬅️ Voltar à Visão Geral"):
            st.session_state.etapa = "VisaoGeral"
            st.rerun()
        st.stop()

    df_sih = st.session_state.df_sih          # tabela mensal original (não a anual)
    df_pop = st.session_state.df_pop
    colunas_data = st.session_state.colunas_data
    COL_LOCAL_POP = df_pop.columns[0]
    anos_pop = colunas_de_ano(df_pop)

    with st.sidebar:
        st.header("⚙️ Configurações — Séries Temporais")

        if st.button("⬅️ Voltar à Visão Geral"):
            st.session_state.etapa = "VisaoGeral"
            st.rerun()

        st.subheader("Escolha o Tipo de Análise")
        tipo_analise = st.radio(
            "Teste temporal:",
            ["Decomposição STL", "Tendência Mann-Kendall", "Análise do Índice Sazonal", "Tabela de Tendência"],
        )

        municipio_escolhido = None
        if tipo_analise == "Tabela de Tendência":
            st.subheader("📈 Tabela de Tendência")
            st.caption(
                "Essa análise não usa o filtro de Nível Observacional: ela calcula, de "
                "uma vez só, o início e o fim do período para Norte, Nordeste, Sudeste, "
                "Sul, Centro-Oeste e Brasil."
            )
            periodicidade = st.radio("Periodicidade da série:", ["anual", "mensal"])
        else:
            st.subheader("Nível Observacional")
            nivel = st.selectbox("Nível:", ["País", "Região", "Estado", "Município"])
            if nivel == "Região":
                regiao_escolhida = st.selectbox("Selecione a Região:", sorted(set(REGIAO_POR_CODIGO.values())))
            elif nivel in ("Estado", "Município"):
                uf_escolhida = st.selectbox("Selecione o Estado:", sorted(set(SIGLA_POR_CODIGO.values())))
                if nivel == "Município":
                    codigo_uf_mun = [c for c, s in SIGLA_POR_CODIGO.items() if s == uf_escolhida][0]
                    col_municipio = df_sih.columns[0]
                    df_uf_mun = df_sih[df_sih["codigo_uf"] == codigo_uf_mun]
                    opcoes_municipio = sorted(
                        v for v in df_uf_mun[col_municipio].dropna().astype(str).unique().tolist()
                        if "IGNORADO" not in v.upper() and "IGNORADA" not in v.upper()
                    )
                    if opcoes_municipio:
                        municipio_escolhido = st.selectbox("Selecione o Município:", opcoes_municipio)
                        st.caption(
                            "Linhas do tipo 'MUNICÍPIO IGNORADO' foram ocultadas dessa lista "
                            "(não representam um lugar específico), mas continuam entrando "
                            "normalmente nos totais de Estado/Região/País."
                        )
                    else:
                        st.warning("Nenhum município encontrado para esse Estado no arquivo do SIH.")

        st.subheader("Tipo da Série")
        tipo_serie = st.selectbox(
            "Como expressar os valores:",
            ["Número Absoluto", "Taxa por 1.000 hab.", "Taxa por 10.000 hab.", "Taxa por 100.000 hab."],
        )
        base_taxa = {"Número Absoluto": None, "Taxa por 1.000 hab.": 1000,
                      "Taxa por 10.000 hab.": 10000, "Taxa por 100.000 hab.": 100000}[tipo_serie]

        st.subheader("Período")
        datas_disponiveis = sorted(colunas_data.values())
        opcoes_periodo = [f"{d:%Y-%m}" for d in datas_disponiveis]
        mes_inicial = st.selectbox("Mês inicial (AAAA-MM):", opcoes_periodo, index=0)
        mes_final = st.selectbox("Mês final (AAAA-MM):", opcoes_periodo, index=len(opcoes_periodo) - 1)

        if tipo_analise != "Tabela de Tendência":
            st.subheader("Aparência")
            titulo_grafico = st.text_input("Título do Gráfico:", value="Série Temporal")
            rotulo_x = st.text_input("Rótulo do Eixo X:", value="Ano/Mês")
            rotulo_y = st.text_input("Rótulo do Eixo Y:", value=tipo_serie)
            largura_grafico = st.slider("Largura do Gráfico:", 5, 20, 10)
            altura_grafico = st.slider("Altura do Gráfico:", 5, 20, 8)
            col_y1, col_y2 = st.columns(2)
            with col_y1:
                y_min = st.text_input("Valor mínimo do eixo Y (deixe em branco para automático):", value="")
            with col_y2:
                y_max = st.text_input("Valor máximo do eixo Y (deixe em branco para automático):", value="")

        if tipo_analise == "Tabela de Tendência":
            executar = st.button("📊 Gerar Tabela de Tendência")
        else:
            executar = st.button(f"▶️ Executar {tipo_analise}")

    if not executar:
        st.info("Configure os parâmetros na barra lateral e clique em Executar.")
        st.stop()

    try:
        if tipo_analise != "Tabela de Tendência":
            # ---------- Filtro por nível observacional ----------
            if nivel == "País":
                df_subset = df_sih
                nome_pop_normalizado = normalizar_texto("Brasil")
            elif nivel == "Região":
                codigos_da_regiao = [c for c, r in REGIAO_POR_CODIGO.items() if r == regiao_escolhida]
                df_subset = df_sih[df_sih["codigo_uf"].isin(codigos_da_regiao)]
                nome_pop_normalizado = normalizar_texto(regiao_escolhida)
            elif nivel == "Estado":
                codigo_da_uf = [c for c, s in SIGLA_POR_CODIGO.items() if s == uf_escolhida][0]
                df_subset = df_sih[df_sih["codigo_uf"] == codigo_da_uf]
                nome_pop_normalizado = normalizar_texto(NOME_UF_POR_CODIGO[codigo_da_uf])
            else:  # Município
                if municipio_escolhido is None:
                    st.error("Selecione um município válido na barra lateral.")
                    st.stop()
                col_municipio = df_sih.columns[0]
                df_subset = df_sih[df_sih[col_municipio].astype(str) == str(municipio_escolhido)]
                nome_pop_normalizado = normalizar_texto(extrair_nome_municipio(municipio_escolhido))

            if df_subset.empty:
                st.error("Nenhum município encontrado para esse filtro.")
                st.stop()

            # ---------- Monta a série de casos absolutos ----------
            serie_casos = montar_serie_casos(df_subset, colunas_data)
            serie_casos = serie_casos.loc[pd.Timestamp(mes_inicial + "-01"):pd.Timestamp(mes_final + "-01")]

            if len(serie_casos) < 2 * PERIODO_SAZONAL:
                st.error(
                    f"Poucos meses no período selecionado ({len(serie_casos)}). "
                    f"São necessários pelo menos {2 * PERIODO_SAZONAL} meses (2 ciclos anuais) "
                    "para decompor Tendência/Sazonalidade/Resíduo com confiabilidade."
                )
                st.stop()

            # ---------- Aplica a métrica escolhida ----------
            if tipo_serie == "Número Absoluto":
                serie = serie_casos
            else:
                serie = aplicar_taxa(serie_casos, df_pop, COL_LOCAL_POP, nome_pop_normalizado,
                                      anos_pop, base_taxa, nivel=nivel)
                if serie.isna().all():
                    nome_buscado = {
                        "País": "Brasil",
                        "Região": regiao_escolhida if nivel == "Região" else None,
                        "Estado": uf_escolhida if nivel == "Estado" else None,
                        "Município": municipio_escolhido if nivel == "Município" else None,
                    }[nivel]
                    st.error(
                        f"Não encontrei população para '{nivel}: {nome_buscado}' "
                        "no arquivo de população. Confira o diagnóstico abaixo."
                    )
                    with st.expander("🔎 Diagnóstico do casamento de nomes de população", expanded=True):
                        st.write(f"Nome buscado (normalizado): `{nome_pop_normalizado}`")
                        st.write(
                            f"Nomes encontrados na coluna `{COL_LOCAL_POP}` do arquivo de "
                            "população (até 80, em ordem alfabética):"
                        )
                        st.write(nomes_disponiveis_populacao(df_pop, COL_LOCAL_POP))
                        if nivel == "País":
                            st.info(
                                "Para 'País' o app tenta somar a população das 27 UFs quando não "
                                "existe uma linha explícita de 'Brasil' no arquivo. Se mesmo assim "
                                "falhou, confira se os nomes de Estado no arquivo de população "
                                f"batem com: {sorted(NOME_UF_POR_CODIGO.values())}"
                            )
                    st.stop()
                serie = serie.dropna()

            # ---------- Alerta de possível competência incompleta no SIH ----------
            if mes_final == opcoes_periodo[-1] and len(serie) >= 4:
                janela_recente = serie.iloc[-7:-1] if len(serie) >= 7 else serie.iloc[:-1]
                media_recente = janela_recente.mean()
                ultimo_valor = serie.iloc[-1]
                if pd.notna(media_recente) and media_recente > 0 and ultimo_valor < 0.6 * media_recente:
                    st.warning(
                        f"⚠️ O último mês da série ({serie.index[-1]:%m/%Y} = {ultimo_valor:,.1f}) "
                        f"está bem abaixo da média dos meses anteriores ({media_recente:,.1f}). "
                        "Isso costuma ser sintoma de competência ainda incompleta no SIH/DATASUS "
                        "— hospitais enviam AIH com atraso, então os últimos meses sobem aos poucos "
                        "conforme os dados são consolidados. Se não for uma queda real, considere "
                        "escolher um 'Mês final' anterior na barra lateral (excluindo esse(s) "
                        "último(s) mês(es)) antes de rodar STL, Mann-Kendall ou Índice Sazonal, "
                        "para não distorcer o resultado."
                    )

            y_min_val = float(y_min) if y_min.strip() else None
            y_max_val = float(y_max) if y_max.strip() else None

        # ================================================================
        # DECOMPOSIÇÃO STL
        # ================================================================
        if tipo_analise == "Decomposição STL":
            with st.spinner("Rodando a decomposição STL..."):
                resultado, f_tendencia, f_sazonalidade = rodar_stl(serie)

            c1, c2 = st.columns(2)
            c1.metric("Força da Tendência (F_T)", f"{f_tendencia:.3f}")
            c2.metric("Força da Sazonalidade (F_S)", f"{f_sazonalidade:.3f}")
            if f_sazonalidade > 0.4:
                st.info("F_S > 0,4: indicativo de presença de sazonalidade significativa.")

            fig, axes = plt.subplots(4, 1, figsize=(largura_grafico, altura_grafico), sharex=True)
            fig.suptitle(titulo_grafico, fontsize=14, fontweight="bold")

            axes[0].plot(serie.index, serie.values, color="tab:blue")
            axes[0].set_title("Observado", fontsize=10)
            if y_min_val is not None or y_max_val is not None:
                axes[0].set_ylim(y_min_val, y_max_val)

            axes[1].plot(serie.index, resultado.trend, color="tab:red")
            axes[1].set_title("Tendência", fontsize=10)

            axes[2].plot(serie.index, resultado.seasonal, color="tab:green")
            axes[2].set_title("Sazonalidade", fontsize=10)

            axes[3].plot(serie.index, resultado.resid, color="tab:purple")
            axes[3].set_title("Resíduo", fontsize=10)
            axes[3].set_xlabel(rotulo_x)

            for ax in axes:
                ax.set_ylabel(rotulo_y, fontsize=8)
                ax.grid(alpha=0.3)

            st.pyplot(fig)

            with st.expander("📊 Valores da Série Temporal"):
                st.dataframe(
                    pd.DataFrame({
                        "Data": serie.index,
                        "Observado": serie.values,
                        "Tendência": resultado.trend.values,
                        "Sazonalidade": resultado.seasonal.values,
                        "Resíduo": resultado.resid.values,
                    }),
                    use_container_width=True,
                )

        # ================================================================
        # TENDÊNCIA MANN-KENDALL (modificado por Hamed e Rao, 1998)
        # ================================================================
        elif tipo_analise == "Tendência Mann-Kendall":
            with st.spinner("Calculando a tendência (STL, só para visualização)..."):
                resultado_stl, _, _ = rodar_stl(serie)

            with st.spinner("Rodando o Teste de Mann-Kendall modificado (Hamed e Rao, 1998)..."):
                resultado_mk = rodar_mann_kendall(serie)

            fig, ax = plt.subplots(figsize=(largura_grafico, altura_grafico))
            ax.plot(serie.index, serie.values, color="tab:blue", label="Série Original", linewidth=1)
            ax.plot(serie.index, resultado_stl.trend, color="tab:red", label="Tendência (STL)", linewidth=1.5)
            ax.set_title(titulo_grafico, fontsize=14, fontweight="bold")
            ax.set_xlabel(rotulo_x)
            ax.set_ylabel(rotulo_y)
            if y_min_val is not None or y_max_val is not None:
                ax.set_ylim(y_min_val, y_max_val)
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)

            tendencia_pt = TRADUCAO_TENDENCIA_MK.get(resultado_mk.trend, resultado_mk.trend)
            tabela_resultado = pd.DataFrame({
                "Métrica": ["Tendência", "h", "Valor-p", "Estatística Z", "Tau de Kendall", "Inclinação de Sen"],
                "Resultado": [
                    tendencia_pt, resultado_mk.h, formatar_pvalor(resultado_mk.p),
                    f"{resultado_mk.z:.4f}", f"{resultado_mk.Tau:.4f}", f"{resultado_mk.slope:.4f}",
                ],
            })
            st.subheader("Resultados do Teste Mann-Kendall Modificado")
            st.dataframe(tabela_resultado, use_container_width=True)

            if resultado_mk.p < ALPHA:
                st.success(
                    f"Valor-p = {formatar_pvalor(resultado_mk.p)} (< {ALPHA:.2f}): há uma tendência "
                    f"**{tendencia_pt}** estatisticamente significativa (inclinação de Sen = "
                    f"{resultado_mk.slope:.4f} por mês; anualizada ≈ {resultado_mk.slope * 12:.4f})."
                )
            else:
                st.info(
                    f"P-valor ≥ {ALPHA:.2f}: não há evidência de tendência significativa "
                    "— os demais resultados do teste não precisam ser interpretados."
                )

            variacao_pct = ((serie.iloc[-1] - serie.iloc[0]) / serie.iloc[0] * 100) if serie.iloc[0] != 0 else np.nan
            st.caption(
                f"Variação entre o primeiro ({serie.index[0]:%m/%Y}: {serie.iloc[0]:,.2f}) e o "
                f"último ({serie.index[-1]:%m/%Y}: {serie.iloc[-1]:,.2f}) dado do período: "
                f"{variacao_pct:,.1f}%."
            )
            st.caption(
                "Lembrete de metodologia: ao reportar esse resultado, especifique que é o "
                "teste de Mann-Kendall modificado por Hamed e Rao (1998), não paramétrico e "
                "com correção para autocorrelação serial."
            )

            with st.expander("📊 Valores da Série Temporal"):
                st.dataframe(
                    pd.DataFrame({"Data": serie.index, "Valor": serie.values}),
                    use_container_width=True,
                )

        # ================================================================
        # ÍNDICE SAZONAL PADRONIZADO (ISP)
        # ================================================================
        elif tipo_analise == "Análise do Índice Sazonal":
            df_isp = calcular_indice_sazonal_padronizado(serie)

            dados_por_mes = [df_isp.loc[df_isp["mes_num"] == m, "isp"].values for m in range(1, 13)]
            rotulos_meses = [MESES_PT_INV[m] for m in range(1, 13)]

            fig, ax = plt.subplots(figsize=(largura_grafico, altura_grafico))
            ax.boxplot(dados_por_mes)
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(rotulos_meses)
            ax.axhline(100, color="red", linestyle="--", linewidth=1, label="Média Mensal (100%)")
            ax.set_title(titulo_grafico, fontsize=14, fontweight="bold")
            ax.set_xlabel("Mês")
            ax.set_ylabel("Índice Sazonal Padronizado (%)")
            if y_min_val is not None or y_max_val is not None:
                ax.set_ylim(y_min_val, y_max_val)
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)

            st.caption(
                "Lembrete de metodologia: o Índice Sazonal Padronizado (ISP) é o valor de "
                "cada mês individual dividido pela média geral de toda a série, ×100. Assim "
                "como a decomposição STL, **não é um teste estatístico** — não gera P-valor. "
                "A leitura é visual/subjetiva a partir do boxplot: se o 1º quartil de um mês "
                "está acima da linha vermelha (100%), há efeito sazonal positivo; se o 3º "
                "quartil está abaixo, há efeito sazonal negativo."
            )

            st.subheader("Interpretação por Mês (regra de bolso: 1º/3º quartil vs. 100%)")
            tabela_interp = interpretar_indice_sazonal(df_isp)
            st.dataframe(tabela_interp, use_container_width=True)

            with st.expander("📊 Valores Individuais (ISP por mês/ano)"):
                st.dataframe(
                    df_isp[["data", "valor", "isp"]].rename(
                        columns={"data": "Data", "valor": "Valor Observado", "isp": "ISP (%)"}
                    ),
                    use_container_width=True,
                )

        # ================================================================
        # TABELA DE TENDÊNCIA (Norte/Nordeste/Sudeste/Sul/Centro-Oeste/Brasil)
        # ================================================================
        else:
            UNIDADES_TENDENCIA = ["Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste", "Brasil"]

            base_taxa_tabela = base_taxa if base_taxa is not None else 100000
            rotulo_taxa_tabela = (
                tipo_serie if base_taxa is not None else "Taxa por 100.000 hab. (padrão)"
            )

            inicio_ts = pd.Timestamp(mes_inicial + "-01")
            fim_ts = pd.Timestamp(mes_final + "-01")

            series_abs_tendencia = {}
            series_taxa_tendencia = {}

            with st.spinner("Calculando a Tabela de Tendência para todas as unidades..."):
                for unidade in UNIDADES_TENDENCIA:
                    if unidade == "Brasil":
                        df_unidade = df_sih
                        nivel_unidade = "País"
                    else:
                        codigos_unidade = [c for c, r in REGIAO_POR_CODIGO.items() if r == unidade.upper()]
                        df_unidade = df_sih[df_sih["codigo_uf"].isin(codigos_unidade)]
                        nivel_unidade = "Região"
                    nome_pop_unidade = normalizar_texto(unidade)

                    serie_abs_unidade = montar_serie_casos(df_unidade, colunas_data)
                    serie_abs_unidade = serie_abs_unidade.loc[inicio_ts:fim_ts]

                    if periodicidade == "anual":
                        anos_da_serie = serie_abs_unidade.index.year
                        serie_abs_unidade = serie_abs_unidade.groupby(anos_da_serie).sum()
                        serie_abs_unidade.index = pd.to_datetime(
                            serie_abs_unidade.index.astype(str) + "-12-31"
                        )

                    serie_taxa_unidade = aplicar_taxa(
                        serie_abs_unidade, df_pop, COL_LOCAL_POP, nome_pop_unidade,
                        anos_pop, base_taxa_tabela, nivel=nivel_unidade,
                    )

                    series_abs_tendencia[unidade] = serie_abs_unidade
                    series_taxa_tendencia[unidade] = serie_taxa_unidade

            linhas_tendencia = []
            for unidade in UNIDADES_TENDENCIA:
                s_abs = series_abs_tendencia[unidade]
                s_taxa = series_taxa_tendencia[unidade]
                linhas_tendencia.append({
                    "Unidade": unidade,
                    f"Início - absoluto - {mes_inicial}": s_abs.iloc[0] if len(s_abs) else np.nan,
                    f"Fim - absoluto - {mes_final}": s_abs.iloc[-1] if len(s_abs) else np.nan,
                    f"Início - taxa - {mes_inicial}": round(s_taxa.iloc[0], 2) if len(s_taxa) else np.nan,
                    f"Fim - taxa - {mes_final}": round(s_taxa.iloc[-1], 2) if len(s_taxa) else np.nan,
                })
            tabela_tendencia = pd.DataFrame(linhas_tendencia)

            st.subheader("📈 Tabela de Tendência")
            st.dataframe(tabela_tendencia, use_container_width=True)
            aviso_base_padrao = (
                "" if base_taxa is not None else
                " Como 'Número Absoluto' foi escolhido em 'Tipo da Série', a coluna de "
                "taxa usa por padrão a base de 100.000 hab. apenas como referência informativa."
            )
            st.caption(
                f"Periodicidade: **{periodicidade}**. Taxa calculada como "
                f"**{rotulo_taxa_tabela}**, com a população do ano mais próximo disponível "
                f"para cada unidade.{aviso_base_padrao}"
            )
            csv_tendencia = tabela_tendencia.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 Baixar CSV da Tabela de Tendência", csv_tendencia,
                "tabela_tendencia.csv", "text/csv",
            )

            datas_uniao = sorted(
                set().union(*[set(series_abs_tendencia[u].index) for u in UNIDADES_TENDENCIA])
            )
            df_series_utilizadas = pd.DataFrame({"Data": datas_uniao})
            for unidade in UNIDADES_TENDENCIA:
                df_series_utilizadas[f"{unidade} - absoluto"] = (
                    df_series_utilizadas["Data"].map(series_abs_tendencia[unidade].to_dict())
                )
                df_series_utilizadas[f"{unidade} - taxa"] = (
                    df_series_utilizadas["Data"].map(series_taxa_tendencia[unidade].round(2).to_dict())
                )

            st.subheader("🕒 Séries Temporais Utilizadas")
            st.dataframe(df_series_utilizadas, use_container_width=True)
            csv_series_utilizadas = df_series_utilizadas.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 Baixar CSV das Séries Utilizadas", csv_series_utilizadas,
                "series_utilizadas.csv", "text/csv",
            )

            st.subheader("📉 Gráficos das Séries de Taxa")

            if periodicidade == "mensal":
                figsize_grafico = (12, 5)
                tamanho_marcador = 4
                espessura_linha = 1.2
            else:
                figsize_grafico = (7, 4)
                tamanho_marcador = 6
                espessura_linha = 1.5

            for unidade in UNIDADES_TENDENCIA:
                s_taxa = series_taxa_tendencia[unidade]
                fig, ax = plt.subplots(figsize=figsize_grafico)
                ax.plot(
                    s_taxa.index, s_taxa.values,
                    marker="o", markersize=tamanho_marcador, linewidth=espessura_linha,
                    color="tab:blue",
                )
                ax.set_title(f"Série temporal de taxa: {unidade}")
                ax.set_xlabel("Data")
                ax.set_ylabel("Taxa")

                if periodicidade == "mensal":
                    ax.xaxis.set_major_locator(mdates.YearLocator())
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
                    fig.autofmt_xdate(rotation=0, ha="center")

                ax.grid(axis="y", alpha=0.3)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                fig.tight_layout()
                st.pyplot(fig)

    except Exception as e:
        st.error(f"Erro ao gerar a análise: {e}")
