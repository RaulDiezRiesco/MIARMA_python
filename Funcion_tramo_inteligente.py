"""
MÓDULO: Funcion_tramo_inteligente.py

Este módulo implementa la lógica para identificar y seleccionar de forma inteligente
el tramo más representativo de una serie temporal para su uso en modelos ARMA 
Integra validaciones estadísticas, pruebas de estacionariedad,
interpolación selectiva y visualización guiada del tramo seleccionado.

────────────────────────────────────────────────────────────────────────────
📌 FUNCIONALIDADES PRINCIPALES:

1. Selección jerárquica de tramos representativos:
   - `seleccionar_tramo_inteligente_para_arma()`: Punto de entrada principal. Aplica
     múltiples criterios (tests estadísticos, estructura PACF, estabilidad) para encontrar
     el tramo más apto.

2. Estrategias de búsqueda:
   - `_buscar_por_tests()`: Evalúa subseries mediante tests ADF y KPSS + score estructural.
   - `_buscar_por_score()`: Alternativa más laxa, basada solo en la estructura PACF.
   - `_buscar_fallback()`: Último recurso heurístico, basado en estabilidad de media y varianza.

3. Evaluación cuantitativa de tramos:
   - `calcular_score()`: Calcula un score combinando calidad estadística y estructura autoregresiva.
   - `adf_seguro()`, `kpss_seguro()`: Implementaciones robustas de tests de estacionariedad, tolerantes a errores.

4. Interpolación controlada:
   - `interpolar_tramos_cortos()`: Interpola tramos breves con NaNs, aplicando un umbral
     máximo y registrando la intervención.

5. Registro y trazabilidad de descartes:
   - `registrar_descarte()`: Centraliza y detalla el motivo de descarte de cada subserie.
   - Variables globales: `motivos_descartes`, `motivos_descartes_extendidos`.

6. Visualización y modelado:
   - `mostrar_resultados_modelado()`: Visualiza y documenta el tramo final seleccionado.
   - `ejecutar_modelado_arma()`: Función orquestadora que ejecuta todo el flujo de análisis.

────────────────────────────────────────────────────────────────────────────
RETORNO FINAL:
   - Tramo de serie escogido
"""

# ==============================================================
# 🧱 1. LIBRERÍAS ESTÁNDAR
# ==============================================================

import json
import warnings
from typing import Tuple, Optional, Union, List, Callable, Dict

# ==============================================================
# 📦 2. LIBRERÍAS DE TERCEROS
# ==============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copy import deepcopy
from collections import Counter
from scipy.stats import normaltest
from statsmodels.tsa.stattools import adfuller, kpss, pacf
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tools.sm_exceptions import InterpolationWarning

# ==============================================================
# 📦 3. IMPORTS INTERNOS (otros módulos del proyecto)
# ==============================================================

from Funciones_preanalisis import preguntar_si_no, plot_serie_inicial

# ==============================================================
# ⚙️ 4. CONFIGURACIÓN GLOBAL DESDE CONFIG.PY
# ==============================================================

# ── Parámetros de ventana
from config import (
    MIN_VENTANA, MAX_VENTANA, PASO, STEP_VENTANA,

    # Interpolación
    MAX_NULOS_INTERPOLABLES, METODO_INTERPOLACION_RELLENO,

    # Umbrales para evaluación de score
    MIN_LARGO_SUBSERIE, MAX_DELTA_MEAN, MAX_DELTA_VAR,

    # PACF
    NLAGS_SCORE_MIN, NLAGS_SCORE_MAX,
    NLAGS_PACF_MINIMO_EFECTIVO, PACF_METHOD,

    # Tests de estacionariedad
    NLAGS_KPSS_DEFAULT, FALLBACK_PVALUE_KPSS,
    AUTO_LAG_ADF_DEFAULT, FALLBACK_PVALUE_ADF,
    CRITERIO_ESTRUCTO_TEST,

    # Umbrales estadísticos generales
    PVALOR_ADF_UMBRAL, PVALOR_KPSS_UMBRAL, MIN_VARIANZA_ADMISIBLE,

    # Motivos de descarte estandarizados
    DESCARTE_SCORE_NAN, DESCARTE_SCORE_VAR_CERO, DESCARTE_SCORE_CORTA,
    DESCARTE_SCORE_ESTADISTICA, DESCARTE_SCORE_PACF_INSUF, DESCARTE_SCORE_PACF_ERROR,
    DESCARTE_SCORE_DIVISION_CERO, DESCARTE_SCORE_DESCONOCIDO,

    DESCARTE_FALLBACK_NAN, DESCARTE_FALLBACK_VAR_CERO, DESCARTE_FALLBACK_DIVISION_CERO,
    DESCARTE_TEST_NAN, DESCARTE_TEST_VAR_CERO, DESCARTE_TEST_PVALORES,
    DESCARTE_TEST_SCORE_CORTA, DESCARTE_TEST_SCORE_ESTADISTICA,
    DESCARTE_TEST_SCORE_PACF_INSUF, DESCARTE_TEST_SCORE_PACF_ERROR,
    DESCARTE_TEST_SCORE_DIVISION_CERO, DESCARTE_TEST_SCORE_DESCONOCIDO,

    # Configuración de salida
    NOMBRE_LOG_TRAMO_DEFAULT, INCLUIR_LOG_EXTENDIDO_TRAMO_MODELOS,
    NOMBRE_ARCHIVO_TRAMO, TITULO_GRAFICO_TRAMO,

    # Modo automático
    MODO_AUTO
)

# ==============================================================
# ⚠️ 5. SUPRIMIR WARNINGS DE INTERPOLACIÓN DE STATSMODELS
# ==============================================================

warnings.simplefilter("ignore", InterpolationWarning)

# ==============================================================
# 🧾 6. VARIABLES GLOBALES DE TRAZABILIDAD
# ==============================================================
motivos_descartes: List[str] = []
motivos_descartes_extendidos: List[str] = []

def extraer_subserie(
    serie: pd.Series,
    inicio: int,
    ventana: int,
    log_msg: Optional[Callable[[str], None]] = None
) -> pd.Series:
    
    """
    Extrae de forma segura una subserie de longitud fija desde una posición
    específica dentro de una serie temporal de acuerdo a los parámetros de la función, con validación de límites.

    La función previene errores comunes asociados a solicitudes fuera de rango:
    - Índices negativos o fuera de los límites de la serie.
    - Tamaños de ventana inválidos.
    - Tramos que se extienden más allá del final de la serie.

    Parámetros:
        serie (pd.Series): Serie temporal original desde la cual extraer el tramo.
        inicio (int): Índice inicial del tramo a extraer (basado en posición, no en índice real).
        ventana (int): Longitud del tramo deseado. Debe ser > 0 y respetar los límites de la serie.
        log_msg (Callable, opcional): Función para registrar mensajes o advertencias.

    Retorna:
        pd.Series: Tramo válido extraído. Si los parámetros no son válidos, retorna una Serie vacía.
    """

    # Validación 1: ventana no puede ser nula o negativa
    if ventana <= 0:
        log_msg(f"⚠️ Ventana inválida: {ventana}. Debe ser > 0.")
        return pd.Series(dtype=serie.dtype)

    # Validación 2: índice de inicio fuera del rango permitido
    if inicio < 0 or inicio >= len(serie):
        log_msg(f"⚠️ Índice de inicio fuera de rango: {inicio}")
        return pd.Series(dtype=serie.dtype)

    # Validación 3: tramo solicitado excede el final de la serie
    fin = inicio + ventana
    if fin > len(serie):
        log_msg(f"⚠️ Subserie excede el límite de la serie: {inicio} → {fin}")
        return pd.Series(dtype=serie.dtype)

    # Subserie válida: devolver segmento solicitado
    return serie.iloc[inicio:fin]

def kpss_seguro(
    serie: Union[pd.Series, np.ndarray],
    nlags: str = NLAGS_KPSS_DEFAULT,
    fallback_pvalue: float = FALLBACK_PVALUE_KPSS,
    log_msg: Optional[Callable[[str], None]] = None
) -> float:
    """
    Ejecuta la prueba de estacionariedad KPSS (Kwiatkowski–Phillips–Schmidt–Shin)
    , devolviendo un valor de respaldo si ocurre un error.

    Parámetros:
        serie (Union[pd.Series, np.ndarray]): Serie temporal a analizar.
        nlags (str): Estrategia de selección de rezagos ('auto', 'legacy', 'short', 'long').
        fallback_pvalue (float): Valor p por defecto a devolver si el test falla.
        log_msg (Callable, opcional): Función para registrar mensajes o errores.

    Retorna:
        float: Valor p del test KPSS o el valor de respaldo si ocurre un error.
    """
    try:
        # Intenta ejecutar el test KPSS sobre la serie
        _, p_value, _, _ = kpss(serie, nlags=nlags)
        return p_value

    except Exception as e:
        # Si falla, registra el error y retorna un valor p alternativo configurable
        log_msg(f"⚠️ KPSS falló con nlags='{nlags}': {e}")
        return fallback_pvalue

def adf_seguro(
    serie: Union[pd.Series, np.ndarray],
    autolag: str = AUTO_LAG_ADF_DEFAULT,
    fallback_pvalue: float = FALLBACK_PVALUE_ADF,
    log_msg: Optional[Callable[[str], None]] = None
) -> float:
    
    """
    Ejecuta la prueba ADF (Augmented Dickey-Fuller) de forma segura, con tolerancia a errores.

    Parámetros:
        serie (Union[pd.Series, np.ndarray]): Serie temporal a evaluar.
        autolag (str): Estrategia para seleccionar el número óptimo de rezagos
                       ('AIC', 'BIC', 't-stat', etc.).
        fallback_pvalue (float): Valor p a devolver si ocurre un error durante el test.
        log_msg (Callable, opcional): Función para registrar errores u observaciones.

    Retorna:
        float: Valor p obtenido del test ADF o el valor de respaldo si se produce una excepción.
    """

    try:
        # Intenta ejecutar la prueba ADF sobre la serie
        _, p_value, _, _, _, _ = adfuller(serie, autolag=autolag)
        return p_value

    except Exception as e:
        # Si falla, registra el error y devuelve el valor p de respaldo
        log_msg(f"⚠️ ADF falló con autolag='{autolag}': {e}")
        return fallback_pvalue

def calcular_score(
    subserie: pd.Series,
    serie: pd.Series,
    min_largo: int = MIN_LARGO_SUBSERIE,
    max_delta_mean: float = MAX_DELTA_MEAN,
    max_delta_var: float = MAX_DELTA_VAR,
    min_varianza: float = MIN_VARIANZA_ADMISIBLE,
    nlags_min: int = NLAGS_SCORE_MIN,
    nlags_max: int = NLAGS_SCORE_MAX,
    nlags_pacf_min: int = NLAGS_PACF_MINIMO_EFECTIVO,
    pacf_method: str = PACF_METHOD,
    log_msg: Optional[Callable[[str], None]] = None,
) -> float:
    """
    Evalúa la calidad estadística y estructural de una subserie temporal mediante un score compuesto.

    Este score ponderado combina:
        - Estructura autoregresiva (PACF).
        - Similitud estadística con la serie completa (media y varianza).
        - Tamaño relativo respecto a la serie original.

    Si la subserie no es apta por razones estadísticas o técnicas, se devuelve un código negativo
    que indica el motivo específico del descarte.

    Códigos de retorno negativos:
        - -1.0 → Subserie demasiado corta.
        - -2.0 → Diferencia excesiva en media/varianza, o varianza nula.
        - -3.0 → Subserie demasiado corta para PACF efectivo.
        - -4.0 → Error al calcular PACF (NaNs o excepción).
        - -5.0 → División por cero durante normalización del score.

    Parámetros:
        subserie (pd.Series): Tramo de serie a evaluar.
        serie (pd.Series): Serie original completa.
        min_largo (int): Longitud mínima para aceptar una subserie.
        max_delta_mean (float): Diferencia máxima aceptada en medias.
        max_delta_var (float): Diferencia máxima aceptada en varianza.
        min_varianza (float): Varianza mínima aceptada para el tramo.
        nlags_min (int): Mínimo número de lags para PACF.
        nlags_max (int): Máximo número de lags para PACF.
        nlags_pacf_min (int): Umbral mínimo efectivo de lags útiles.
        pacf_method (str): Método de cálculo de PACF (por ejemplo, 'ywunbiased').
        log_msg (Callable, opcional): Función de logging para registrar errores o eventos.

    Retorna:
        float:
            - Score positivo → subserie válida.
            - Score negativo → motivo de descarte codificado.
    """

    # Validación de longitud mínima
    if len(subserie) < min_largo:
        log_msg("🔴 Subserie demasiado corta.")
        return -1.0

    serie = serie.dropna()

    # Comparación de media y varianza con la serie completa
    var_sub = subserie.var()
    mean_sub = subserie.mean()
    var_full = serie.var()
    mean_full = serie.mean()

    delta_mean = abs(mean_full - mean_sub)
    delta_var = abs(var_full - var_sub)

    # Validación de diferencias y varianza mínima
    if delta_mean > max_delta_mean or delta_var > max_delta_var:
        return -2.0
    if var_sub < min_varianza:
        return -2.0

    # Validación de PACF: determinar número de lags
    long_sub = len(subserie)
    long_full = len(serie)
    tamano_relativo = long_sub / long_full if long_full else 0.0

    nlags = min(nlags_max, max(nlags_min, long_sub // 10))
    nlags = min(nlags, (long_sub - 1) // 2)

    if nlags < nlags_pacf_min:
        return -3.0

    if subserie.isnull().any():
        return -4.0

    # Cálculo del score estructural con PACF
    weights = np.linspace(1.0, 0.3, nlags)  # Decrece linealmente
    try:
        pacf_vals = np.abs(pacf(subserie, nlags=nlags, method=pacf_method)[1:nlags + 1])
        pacf_score = np.sum(weights * pacf_vals)
    except Exception:
        return -4.0

    #  Cálculo final del score normalizado
    denom = 1 + np.sqrt(var_sub) + delta_mean + delta_var
    if denom == 0:
        return -5.0

    score = (pacf_score * tamano_relativo) / denom
    return score

def registrar_descarte(
    motivo: str,
    inicio: Optional[int] = None,
    fin: Optional[int] = None,
    motivos: Optional[List[str]] = None,
    motivos_ext: Optional[List[str]] = None,
    log_msg: Optional[Callable[[str], None]] = None
) -> str:
    """
    Registra un motivo de descarte para un tramo de una serie temporal de acuerdo al cálculo del score y las pruebas estadísticas realizadas. 

    Este registro se puede almacenar en dos niveles:
    - 🟢 Motivo simple: solo el texto del motivo (ej. "VARIANZA_CERO")
    - 🔵 Motivo extendido: incluye el rango de índice si está disponible (ej. "100:150 VARIANZA_CERO")

    Las listas de descarte pueden ser globales (por defecto) o listas externas pasadas como argumento.

    Parámetros:
        motivo (str): Texto que describe el motivo del descarte.
        inicio (int, opcional): Índice de inicio del tramo descartado.
        fin (int, opcional): Índice final del tramo descartado.
        motivos (List[str], opcional): Lista en la que guardar motivos simples.
        motivos_ext (List[str], opcional): Lista en la que guardar motivos extendidos.
        log_msg (Callable, opcional): Función para log (no utilizada actualmente, pero se puede activar).

    Retorna:
        str: Motivo registrado.
    """
    # Se accede a las listas globales si no se pasan listas específicas
    global motivos_descartes, motivos_descartes_extendidos

    # Si hay información del tramo (inicio-fin), se añade al motivo para mayor detalle
    if inicio is not None and fin is not None:
        motivo_extendido = f"{inicio}:{fin} {motivo}"
    else:
        motivo_extendido = motivo

    # Se usan listas locales si se pasan, si no, las listas globales
    if motivos is None:
        motivos = motivos_descartes
    if motivos_ext is None:
        motivos_ext = motivos_descartes_extendidos

    # Añadir motivo simple y extendido a sus respectivas listas
    motivos.append(motivo)
    motivos_ext.append(motivo_extendido)

    return motivo_extendido

def _buscar_por_tests(
    serie: pd.Series,
    prop_ventana: float = MAX_VENTANA,
    prop_ventana_min: float = MIN_VENTANA,
    paso: float = PASO,
    step_ventana: float = STEP_VENTANA,
    pvalor_adf_umbral: float = PVALOR_ADF_UMBRAL,
    pvalor_kpss_umbral: float = PVALOR_KPSS_UMBRAL,
    min_varianza_admisible: float = MIN_VARIANZA_ADMISIBLE,
    criterio_estricto_tests: bool = CRITERIO_ESTRUCTO_TEST,
    log_msg: Optional[Callable[[str], None]] = None,

) -> Tuple[Optional[int], Optional[int], Optional[pd.Series], float, Optional[float], Optional[float], str]:
    """
    Busca el mejor tramo de una serie temporal utilizando pruebas estadísticas (ADF, KPSS)
    y evaluación estructural (PACF) Aplicando el enfoque de ventanas móviles.

    El objetivo es encontrar una subserie sin NaNs, con varianza aceptable, que además
    cumpla con criterios de estacionariedad y estructura autoregresiva significativa.


    Parámetros:
        serie (pd.Series): Serie temporal original.
        prop_ventana (float): Proporción inicial del tamaño de la ventana (ej. 0.3 = 30%).
        prop_ventana_min (float): Proporción mínima permitida para las ventanas.
        paso (float): Proporción del total de la serie a desplazar entre ventanas (ej. 0.01 = 1%).
        step_ventana (float): Proporción del total de la serie que se reducirá en cada iteración de ventana.
        pvalor_adf_umbral (float): Umbral máximo para aceptar estacionariedad según ADF.
        pvalor_kpss_umbral (float): Umbral máximo para aceptar estacionariedad según KPSS.
        min_varianza_admisible (float): Varianza mínima aceptada para considerar una subserie válida.
        criterio_estricto_tests (bool): Si True, aplica criterio estricto combinando ADF y KPSS.
        log_msg (Callable, opcional): Función para imprimir o registrar mensajes del proceso.

    Retorna:
        Tuple[
            Optional[int],       # Índice inicial del mejor tramo
            Optional[int],       # Índice final (exclusivo) del mejor tramo
            Optional[pd.Series], # Subserie seleccionada
            float,               # Score calculado del tramo
            Optional[float],     # p-valor ADF
            Optional[float],     # p-valor KPSS
            str                  # Método utilizado ("Test+score")
        ]
    """

    L = len(serie)

    # Convertimos proporciones en valores absolutos (en número de observaciones)
    paso = max(1, int(round(L * paso)))
    step_ventana = max(1, int(round(L * step_ventana)))
    min_ventana = max(5, int(round(L * prop_ventana_min)))
    ventana = max(min_ventana + 1, int(round(L * prop_ventana)))

    # Inicializamos variables para guardar el mejor tramo encontrado
    mejor_score = -np.inf
    mejor_subserie = None
    inicio_mejor = None
    fin_mejor = None
    adf_p_mejor = None
    kpss_p_mejor = None

    # Logging de configuración inicial

    log_msg("🔍 Buscando por TESTS (ADF + KPSS)...")
    log_msg(f"📐 Config ventana: inicial={ventana}, mínima={min_ventana}, paso={paso}, reducción={step_ventana}")

    # Búsqueda jerárquica: vamos reduciendo la ventana hasta el mínimo permitido
    while ventana >= min_ventana:
        for inicio in range(0, L - ventana, paso):
            fin = inicio + ventana
            subserie = extraer_subserie(serie, inicio, ventana, log_msg=log_msg)

            # Filtro 1: Si hay nulos, descartamos
            if subserie.isna().sum() > 0:
                registrar_descarte(DESCARTE_TEST_NAN, inicio, fin, log_msg=log_msg)
                continue

            # Filtro 2: Si tiene varianza muy baja, descartamos
            if subserie.std() < min_varianza_admisible:
                registrar_descarte(DESCARTE_TEST_VAR_CERO, inicio, fin, log_msg=log_msg)
                continue

            # Pruebas estadísticas ADF y KPSS
            adf_p = adf_seguro(subserie, log_msg=log_msg)
            kpss_p = kpss_seguro(subserie, log_msg=log_msg)

            # Evaluación de estacionariedad
            if criterio_estricto_tests:
                if adf_p >= pvalor_adf_umbral or kpss_p <= pvalor_kpss_umbral:
                    registrar_descarte(DESCARTE_TEST_PVALORES, inicio, fin, log_msg=log_msg)
                    continue
            else:
                if adf_p >= pvalor_adf_umbral and kpss_p <= pvalor_kpss_umbral:
                    registrar_descarte(DESCARTE_TEST_PVALORES, inicio, fin, log_msg=log_msg)
                    continue

            # Evaluamos la calidad estructural con un score PACF
            score = calcular_score(subserie, serie, log_msg=log_msg)

            # Si el score es negativo, lo descartamos con su causa
            if score < 0:
                if score == -1.0:
                    registrar_descarte(DESCARTE_TEST_SCORE_CORTA, inicio, fin, log_msg=log_msg)
                elif score == -2.0:
                    registrar_descarte(DESCARTE_TEST_SCORE_ESTADISTICA, inicio, fin, log_msg=log_msg)
                elif score == -3.0:
                    registrar_descarte(DESCARTE_TEST_SCORE_PACF_INSUF, inicio, fin, log_msg=log_msg)
                elif score == -4.0:
                    registrar_descarte(DESCARTE_TEST_SCORE_PACF_ERROR, inicio, fin, log_msg=log_msg)
                elif score == -5.0:
                    registrar_descarte(DESCARTE_TEST_SCORE_DIVISION_CERO, inicio, fin, log_msg=log_msg)
                else:
                    registrar_descarte(DESCARTE_TEST_SCORE_DESCONOCIDO, inicio, fin, log_msg=log_msg)
                continue

            # Guardamos el mejor tramo encontrado hasta ahora
            if score > mejor_score:
                mejor_score = score
                mejor_subserie = subserie
                inicio_mejor = inicio
                fin_mejor = fin
                adf_p_mejor = adf_p
                kpss_p_mejor = kpss_p

        # Si ya tenemos un tramo válido, no reducimos más la ventana
        if mejor_subserie is not None:
            log_msg(f"✅ Tramo TEST encontrado: {inicio_mejor}-{fin_mejor} | Score: {mejor_score:.2f}")
            break

        # Reducimos la ventana y continuamos iterando
        ventana -= step_ventana

    # Retornamos resultado final
    return inicio_mejor, fin_mejor, mejor_subserie, mejor_score, adf_p_mejor, kpss_p_mejor, "Test+score"

def _buscar_por_score(
    serie: pd.Series,
    prop_ventana: float = MAX_VENTANA,
    prop_ventana_min: float = MIN_VENTANA,
    paso: float = PASO,
    step_ventana: float = STEP_VENTANA,
    min_varianza_admisible: float = MIN_VARIANZA_ADMISIBLE,
    log_msg: Optional[Callable[[str], None]] = None
) -> Tuple[Optional[int], Optional[int], Optional[pd.Series], float, None, None, str]:
    """
    Busca un tramo de serie temporal en base únicamente al análisis estructural,
    utilizando un score basado en la función de autocorrelación parcial (PACF) ponderada.

    Este enfoque actúa como fallback cuando no se puede aplicar (o no superan) los tests
    de estacionariedad ADF y KPSS. Se recorren ventanas móviles sobre la serie original,
    evaluando subseries que no contengan valores nulos y que superen una varianza mínima.

    Se selecciona la subserie con mayor score como el mejor tramo candidato.

    Parámetros:
        serie (pd.Series): Serie temporal original sobre la que buscar el tramo.
        prop_ventana (float): Proporción inicial del tamaño de ventana respecto a la longitud total de la serie (ej. 0.3).
        prop_ventana_min (float): Proporción mínima aceptada para la ventana (ej. 0.05).
        paso (float): Proporción del tamaño total de la serie para desplazar la ventana (ej. 0.01 = 1% del total).
        step_ventana (float): Proporción del tamaño total de la serie para reducir la ventana en cada iteración (ej. 0.01).
        min_varianza_admisible (float): Varianza mínima aceptada para considerar una subserie válida.
        log_msg (Callable | None): Función para registrar mensajes durante el proceso. Si None, no se registra nada.

    Retorna:
        Tuple[
            Optional[int],        # Índice de inicio del tramo seleccionado
            Optional[int],        # Índice de fin (exclusivo) del tramo seleccionado
            Optional[pd.Series],  # Subserie seleccionada como mejor candidata
            float,                # Score PACF obtenido por la subserie
            None,                 # No aplica test ADF
            None,                 # No aplica test KPSS
            str                   # Método utilizado ("Score")
        ]
    """

    L = len(serie)

    # Convertimos proporciones en valores absolutos
    paso = max(1, int(round(L * paso)))
    step_ventana = max(1, int(round(L * step_ventana)))
    min_ventana = max(5, int(round(L * prop_ventana_min)))
    ventana = max(min_ventana + 1, int(round(L * prop_ventana)))

    # Inicializamos las variables del mejor tramo
    mejor_score = -np.inf
    mejor_subserie = None
    inicio_mejor = None
    fin_mejor = None

    # Log de inicio
    log_msg("🔁 Fallback parcial: buscando por SCORE (estructura PACF)...")

    # Búsqueda jerárquica descendente de ventana
    while ventana >= min_ventana:
        for inicio in range(0, L - ventana, paso):
            fin = inicio + ventana
            subserie = extraer_subserie(serie, inicio, ventana, log_msg=log_msg)

            # Descartamos si hay nulos
            if subserie.isna().sum() > 0:
                registrar_descarte(DESCARTE_SCORE_NAN, inicio, fin, log_msg=log_msg)
                continue

            # Descartamos si tiene varianza demasiado baja
            if subserie.std() < min_varianza_admisible:
                registrar_descarte(DESCARTE_SCORE_VAR_CERO, inicio, fin, log_msg=log_msg)
                continue

            # 📈 Calculamos el score estructural (PACF ponderado)
            score = calcular_score(subserie, serie, log_msg=log_msg)

            # Si el score es negativo, lo descartamos con su causa
            if score < 0:
                motivo = {
                    -1.0: DESCARTE_SCORE_CORTA,
                    -2.0: DESCARTE_SCORE_ESTADISTICA,
                    -3.0: DESCARTE_SCORE_PACF_INSUF,
                    -4.0: DESCARTE_SCORE_PACF_ERROR,
                    -5.0: DESCARTE_SCORE_DIVISION_CERO
                }.get(score, DESCARTE_SCORE_DESCONOCIDO)

                registrar_descarte(motivo, inicio, fin, log_msg=log_msg)
                continue

            # Si mejora el score, guardamos como mejor candidato
            if score >= mejor_score:
                mejor_score = score
                mejor_subserie = subserie
                inicio_mejor = inicio
                fin_mejor = fin

        # Si encontramos uno válido, paramos
        if mejor_subserie is not None:
            log_msg(f"✅ Tramo SCORE elegido: {inicio_mejor} → {fin_mejor} | Score: {mejor_score:.3f}")
            break

        # Reducimos tamaño de ventana y repetimos
        ventana -= step_ventana

    # Resultado final
    return inicio_mejor, fin_mejor, mejor_subserie, mejor_score, None, None, "Score"

def _buscar_fallback(
    serie: pd.Series,
    prop_ventana: float = MAX_VENTANA,
    prop_ventana_min: float = MIN_VENTANA,
    paso: float = PASO,  
    step_ventana: float = STEP_VENTANA, 
    min_varianza_admisible: float = MIN_VARIANZA_ADMISIBLE,
    log_msg: Optional[Callable[[str], None]] = None
) -> Tuple[Optional[int], Optional[int], Optional[pd.Series], float, None, None, str]:
    """
    Estrategia final de respaldo ("fallback") para seleccionar un tramo representativo
    de la serie temporal, si fallan las pruebas ADF, KPSS o estructura PACF.

    Selecciona el tramo cuya media y varianza estén más cercanas a la serie completa,
    sin aplicar tests estadísticos ni análisis de autocorrelación. Ideal como último recurso.

    Parámetros:
        serie (pd.Series): Serie temporal original.
        prop_ventana (float): Proporción máxima del tamaño de ventana (ej. 0.3 = 30%).
        prop_ventana_min (float): Proporción mínima aceptada del tamaño de ventana.
        paso (float): Proporción de desplazamiento entre subseries (ej. 0.01 = 1%).
        step_ventana (float): Proporción con la que se reduce progresivamente la ventana.
        min_varianza_admisible (float): Varianza mínima para considerar una subserie válida.
        log_msg (Callable, opcional): Función de log para registrar pasos e incidencias.

    Retorna:
        Tuple[
            Optional[int],       # Índice de inicio del mejor tramo
            Optional[int],       # Índice final (exclusivo)
            Optional[pd.Series], # Subserie seleccionada
            float,               # Score heurístico (más alto es mejor)
            None,                # No aplica ADF
            None,                # No aplica KPSS
            str                  # Método aplicado: "Fallback"
        ]
    """
    # Configuración de ventana
    L = len(serie)
    min_ventana = max(5, int(round(L * prop_ventana_min)))
    ventana = max(min_ventana + 1, int(round(L * prop_ventana)))

    media_total = serie.mean()
    var_total = serie.var()

    # Inicialización de variables para búsqueda
    mejor_score = -np.inf
    mejor_subserie = None
    inicio_mejor = None
    fin_mejor = None

    log_msg(" Fallback final: buscando tramo con media/varianza similar a la serie...")

    # Recorrido de ventanas decrecientes 
    while ventana >= min_ventana:
        desplazamiento = max(1, int(round(L * paso)))  # Convertir proporción a paso absoluto

        for inicio in range(0, L - ventana + 1, desplazamiento):
            fin = inicio + ventana
            subserie = extraer_subserie(serie, inicio, ventana, log_msg=log_msg)

            # Filtros
            if subserie.isna().sum() > 0:
                registrar_descarte(DESCARTE_FALLBACK_NAN, inicio, fin, log_msg=log_msg)
                continue

            var_sub = subserie.var()
            if var_sub < min_varianza_admisible:
                registrar_descarte(DESCARTE_FALLBACK_VAR_CERO, inicio, fin, log_msg=log_msg)
                continue

            # Heurística basada en similitud de media y varianza
            media_diff = abs(subserie.mean() - media_total)
            var_diff = abs(var_sub - var_total)
            denominador = media_diff + var_diff

            if denominador < 1e-10:
                registrar_descarte(DESCARTE_FALLBACK_DIVISION_CERO, inicio, fin, log_msg=log_msg)
                continue

            # Score más alto si el tramo es más parecido a la serie global
            score = (L / ventana) / (denominador + 1e-8)

            if score > mejor_score:
                mejor_score = score
                mejor_subserie = subserie
                inicio_mejor = inicio
                fin_mejor = fin

        # Si se encontró un buen tramo, se detiene la búsqueda
        if mejor_subserie is not None:
            log_msg(f"✅ Tramo Fallback elegido: {inicio_mejor}-{fin_mejor} | Score heurístico: {mejor_score:.2f}")
            break

        # Reducir ventana proporcionalmente
        ventana -= int(round(L * step_ventana))

    return inicio_mejor, fin_mejor, mejor_subserie, mejor_score, None, None, "Fallback"

def interpolar_tramos_cortos(
    serie: pd.Series,
    max_huecos: int = MAX_NULOS_INTERPOLABLES,
    metodo: str = METODO_INTERPOLACION_RELLENO,
    log_msg: Optional[Callable[[str], None]] = None
) -> pd.Series:
    """
    Interpola de forma controlada los tramos cortos de NaNs dentro de una serie temporal,
    siempre que la longitud del hueco no supere un umbral especificado.

    Conserva los extremos del hueco para una interpolación confiable (requiere valores
    previos y posteriores). Si existen NaNs pero ningún tramo es lo suficientemente
    corto para interpolar, se informa por log.

    Parámetros:
        serie (pd.Series): Serie temporal original a procesar.
        max_huecos (int): Tamaño máximo de un hueco (consecutivo) que puede ser interpolado.
        metodo (str): Método de interpolación a aplicar (por defecto, 'linear').
        log_msg (Callable, opcional): Función para registrar mensajes (compatible con log global).

    Retorna:
        pd.Series: Serie con interpolaciones aplicadas en huecos válidos.
    """
    tramos_interpolados_log = []
    serie_interpolada = serie.copy()
    is_na = serie.isna().tolist()

    if not any(is_na):
        if log_msg:
            log_msg("ℹ️ La serie no contiene valores nulos. No se aplica interpolación.")
        return serie_interpolada

    start = None
    for i, es_nan in enumerate(is_na):
        if es_nan and start is None:
            start = i
        elif not es_nan and start is not None:
            fin = i
            longitud = fin - start

            if longitud <= max_huecos and start > 0 and fin < len(serie):
                tramo = serie_interpolada.iloc[start - 1:fin + 1]
                interpolado = tramo.interpolate(method=metodo, limit_direction="both")
                serie_interpolada.iloc[start:fin] = interpolado.iloc[1:-1]

                tramos_interpolados_log.append({
                    "inicio": start,
                    "fin": fin - 1,
                    "longitud": longitud,
                    "metodo": metodo
                })
            start = None

    # Caso especial: hueco al final de la serie
    if start is not None:
        fin = len(serie)
        longitud = fin - start
        if longitud <= max_huecos and start > 0:
            tramo = serie_interpolada.iloc[start - 1:]
            interpolado = tramo.interpolate(method=metodo, limit_direction="both")
            serie_interpolada.iloc[start:] = interpolado.iloc[1:]

            tramos_interpolados_log.append({
                "inicio": start,
                "fin": fin - 1,
                "longitud": longitud,
                "metodo": metodo
            })

    # Registro final
    if log_msg:
        if tramos_interpolados_log:
            log_msg(f"🩹 Interpolados {len(tramos_interpolados_log)} tramo(s) con ≤ {max_huecos} NaNs.")
        else:
            log_msg(f"⚠️ Se encontraron NaNs pero ningún tramo era interpolable (>{max_huecos} consecutivos).")

    return serie_interpolada, tramos_interpolados_log

def seleccionar_tramo_inteligente_para_arma(
    serie: pd.Series,
    prop_ventana: float = MAX_VENTANA,
    prop_ventana_min: float = MIN_VENTANA,
    paso: int = PASO,
    step_ventana: int = STEP_VENTANA,
    max_huecos_interpolables: int = MAX_NULOS_INTERPOLABLES,
    metodo_interpolacion: str = METODO_INTERPOLACION_RELLENO,
    criterio_estricto_tests: bool = CRITERIO_ESTRUCTO_TEST,
    log_msg: Optional[Callable[[str], None]] = None
    
) -> Tuple[dict, pd.Series, list]:
    """
    Selecciona de forma jerárquica un tramo representativo de una serie temporal
    para su posterior modelado con ARMA.

    El flujo aplicado considera:
        - Pruebas estadísticas (ADF y KPSS) para detectar estacionariedad.
        - Calidad estructural mediante score PACF.
        - Interpolación controlada de NaNs pequeños si fallan las pruebas anteriores.
        - Heurístico final por similitud de media/varianza si todo lo anterior falla.

    Parámetros:
    -----------
    serie (pd.Series): Serie temporal original (1D).
    prop_ventana (float): Proporción inicial del tamaño de la ventana respecto al total.
    prop_ventana_min (float): Proporción mínima de la ventana para finalizar búsqueda.
    paso (int): Tamaño del desplazamiento de la ventana en cada iteración.
    step_ventana (int): Reducción de la ventana en cada iteración si no hay éxito.
    max_huecos_interpolables (int): Máx. nulos consecutivos permitidos para interpolación.
    metodo_interpolacion (str): Método de interpolación (ej. 'linear', 'spline').
    criterio_estricto_tests (bool): Si True, ambos tests ADF y KPSS deben ser positivos.
    log_msg (Callable, opcional): Función para registrar mensajes log.

    Retorna:
    --------
    Tuple[
        dict: Información estadística del tramo seleccionado.
        pd.Series: Tramo extraído de la serie original.
        list: Lista extendida con motivos de descarte de otros tramos.
    ]
    """

    # Validaciones iniciales
    if not isinstance(serie, pd.Series):
        log_msg("❌ 'serie' debe ser una pd.Series")
        raise TypeError("❌ 'serie' debe ser una pd.Series")
    if serie.empty:
        log_msg("❌ La serie está vacía.")
        raise ValueError("❌ La serie está vacía.")

    # Inicialización de variables globales (limpieza de descartes previos)
    global motivos_descartes, motivos_descartes_extendidos
    motivos_descartes.clear()
    motivos_descartes_extendidos.clear()

    # Cálculos previos para comparativa al final
    L = len(serie)
    media_total = serie.mean()
    var_total = serie.var()
    interpolacion_aplicada = False

    # Primer intento: Tests + Score
    resultado = _buscar_por_tests(
        serie, prop_ventana, prop_ventana_min, paso, step_ventana,
        log_msg=log_msg, criterio_estricto_tests=criterio_estricto_tests
    )

    # Si no encuentra nada válido, probar usando sólo score PACF
    if resultado[2] is None:
        resultado = _buscar_por_score(
            serie, prop_ventana, prop_ventana_min, paso, step_ventana,
            log_msg=log_msg
        )

    # Guardar motivos de descarte antes de intentar interpolar
    descartes_pre_interp = list(motivos_descartes)

    # Interpolación si falló el primer intento
    if resultado[2] is None:
        log_msg("♻️ Interpolando serie (nulos pequeños)...")

        serie_interp, tramos_interpolados_log = interpolar_tramos_cortos(
            serie,
            max_huecos=max_huecos_interpolables,
            metodo=metodo_interpolacion,
            log_msg=log_msg
        )
        interpolacion_aplicada = True

        # Intentar búsqueda de nuevo sobre la serie interpolada
        motivos_descartes.clear()
        resultado = _buscar_por_tests(
            serie_interp, prop_ventana, prop_ventana_min, paso, step_ventana,
            log_msg=log_msg, criterio_estricto_tests=criterio_estricto_tests
        )
        if resultado[2] is None:
            resultado = _buscar_por_score(
                serie_interp, prop_ventana, prop_ventana_min, paso, step_ventana,
                log_msg=log_msg
            )

    descartes_post_interp = list(motivos_descartes)

    # Último recurso
    if resultado[2] is None:
        motivos_descartes.clear()
        resultado = _buscar_fallback(
            serie, prop_ventana, prop_ventana_min, paso, step_ventana,
            log_msg=log_msg
        )

    index_inicio, index_fin, serie_mejor, mejor_score, adf_p, kpss_p, metodo_seleccion = resultado

    if serie_mejor is None:
        log_msg("❌ No se pudo encontrar ningún tramo válido en la serie.")
        raise RuntimeError("❌ No se pudo encontrar ningún tramo válido en la serie.")

    # Diagnóstico estadístico extra
    try:
        pval_arch = het_arch(serie_mejor.dropna())[1]
        pval_normal = normaltest(serie_mejor.dropna())[1]
    except Exception:
        pval_arch = None
        pval_normal = None

    metadata = {
        "inicio_tramo": int(index_inicio),
        "longitud_tramo": int(index_fin - index_inicio),
        "tamaño_total_serie": int(L),
        "porcentaje_nans_original": float(serie.isna().mean()),
        "interpolacion_aplicada": interpolacion_aplicada,
        "tramos_interpolados": tramos_interpolados_log if interpolacion_aplicada else None,
        "metodo_seleccion": metodo_seleccion,
        "score": float(mejor_score) if mejor_score != -np.inf else None,
        "media": float(serie_mejor.mean()),
        "varianza": float(serie_mejor.var()),
        "std": float(serie_mejor.std()),
        "adf_p": float(adf_p) if adf_p is not None else None,
        "kpss_p": float(kpss_p) if kpss_p is not None else None,
        "diagnostico_extra": {
            "normalidad_tramo_pval": float(pval_normal) if pval_normal is not None else None,
            "heterocedasticidad_arch_pval": float(pval_arch) if pval_arch is not None else None
        },
        "comparativa_tramo_vs_total": {
            "media_total": float(media_total),
            "media_tramo": float(serie_mejor.mean()),
            "delta_media": float(abs(serie_mejor.mean() - media_total)),
            "var_total": float(var_total),
            "var_tramo": float(serie_mejor.var()),
            "delta_var": float(abs(serie_mejor.var() - var_total))
        },
        "motivo_descartes": {
            "original": dict(Counter(descartes_pre_interp)),
            "post_interpolacion": dict(Counter(descartes_post_interp)) if interpolacion_aplicada else {}
        }
    }

    todos_los_descartes = motivos_descartes_extendidos.copy()
    return metadata, serie_mejor, todos_los_descartes

def mostrar_resultados_modelado(
    metadata: dict,
    tramo: pd.Series,
    descartes_ext: Optional[List[str]] = None,
    incluir_log_extendido: bool = INCLUIR_LOG_EXTENDIDO_TRAMO_MODELOS,
    input_func: Callable[[str], str] = input,
    log_msg: Optional[Callable[[str], None]] = None,
    titulo_grafico: Optional[str] = TITULO_GRAFICO_TRAMO,
    nombre_archivo_grafico: Optional[str] = NOMBRE_ARCHIVO_TRAMO,
    res_dir: str = None,
    modo_auto: bool = MODO_AUTO,
) -> Tuple[str, Optional[str]]:
    """
    Muestra los resultados del modelado tras seleccionar un tramo representativo de la serie.

    Incluye:
        - Información del tramo (inicio, score, método).
        - Log detallado en formato JSON con estadísticas y diagnóstico.
        - Registro extendido de descartes (opcional).
        - Opción interactiva para visualizar o guardar un gráfico del tramo.

    Parámetros:
        metadata (dict): Diccionario con estadísticas y detalles del tramo seleccionado.
        tramo (pd.Series): Tramo de la serie temporal usado para modelado.
        descartes_ext (List[str], opcional): Lista de motivos extendidos de descarte por tramo.
        incluir_log_extendido (bool): Si True, muestra el log extendido de descartes.
        input_func (Callable): Función de entrada (input), mockeable para test.
        log_msg (Callable, opcional): Función para loguear los mensajes.
        titulo_grafico (str, opcional): Plantilla para el título del gráfico.
        nombre_archivo_grafico (str, opcional): Plantilla para nombre del archivo del gráfico.
        res_dir (str): Ruta del directorio donde guardar el gráfico.
        modo_auto (bool): Si True, se saltan las preguntas interactivas (modo automático).

    Retorna:
        Tuple[
            str,              # Log resumen en formato JSON (interno)
            Optional[str]     # Log extendido si aplica, o None
        ]
    """

    # Log básico del tramo seleccionado
    log_msg("✅ Tramo seleccionado:")
    log_msg(f"- Inicio: {metadata['inicio_tramo']}")
    log_msg(f"- Longitud: {metadata['longitud_tramo']}")
    log_msg(f"- Score: {metadata['score']:.4f}")
    log_msg(f"- Motivo de selección: {metadata['metodo_seleccion']}")

    # Log completo en formato JSON
    log_msg("📦 Metadata del tramo seleccionado:")
    for linea in json.dumps(metadata, indent=4, ensure_ascii=False).splitlines():
        log_msg(linea)
        
    # Log extendido de descartes (si se solicita)
    if incluir_log_extendido and descartes_ext:
        log_msg("\n📋 Log extendido de descartes:")
        if len(descartes_ext) == 0:
            log_msg("- (Sin descartes registrados)")
        else:
            for d in descartes_ext:
                log_msg(f"- {d}")

    # Preguntas para visualizar y guardar gráfico
    if preguntar_si_no("¿Deseas visualizar el tramo seleccionado?", input_func, modo_auto=modo_auto, log_msg=log_msg):
        mostrar = preguntar_si_no(
            "¿Mostrar en pantalla?", input_func,
            modo_auto=modo_auto, respuesta_modo_auto=False, log_msg=log_msg
        )
        guardar = preguntar_si_no(
            "¿Guardar gráfico del tramo en la carpeta de resultados?",
            input_func, modo_auto=modo_auto, log_msg=log_msg
        )

        # Datos del gráfico
        inicio = metadata["inicio_tramo"]
        fin = inicio + metadata["longitud_tramo"]
        nombre_base = metadata.get("nombre_base", NOMBRE_LOG_TRAMO_DEFAULT)

        titulo_final = titulo_grafico.format(inicio=inicio, fin=fin) if titulo_grafico else None
        nombre_final = nombre_archivo_grafico.format(
            nombre_base=nombre_base,
            inicio=inicio,
            fin=fin
        ) if nombre_archivo_grafico else None

        # Mostrar/guardar gráfico
        plot_serie_inicial(
            serie=tramo,
            titulo=titulo_final,
            guardar=guardar,
            nombre_archivo=nombre_final,
            mostrar=mostrar,
            res_dir=res_dir,
            log_msg=log_msg,
        )

    return

def ejecutar_modelado_arma(
    df: pd.DataFrame,
    input_func: Callable[[str], str] = input,
    prop_ventana: float = MAX_VENTANA,
    prop_ventana_min: float = MIN_VENTANA,
    paso: int = PASO,
    step_ventana: int = STEP_VENTANA,
    incluir_log_extendido_modelo: bool = INCLUIR_LOG_EXTENDIDO_TRAMO_MODELOS,
    titulo_grafico: Optional[str] = TITULO_GRAFICO_TRAMO,
    nombre_archivo_grafico: Optional[str] = NOMBRE_ARCHIVO_TRAMO,
    criterio_estricto_test: bool = CRITERIO_ESTRUCTO_TEST,
    directorio_graficas: str = None,
    log_msg: Optional[Callable[[str], None]] = None,
    modo_auto: bool = MODO_AUTO
) -> Tuple[pd.Series]:
    """
    Ejecuta el proceso completo de selección inteligente de un tramo dentro de 
    una serie temporal para su modelado con ARMA, aplicando estrategias jerárquicas.

    Este procedimiento:
    - Normaliza la serie para asegurar comparabilidad estructural entre tramos.
    - Evalúa distintos tramos mediante tests de estacionariedad (ADF y KPSS).
    - Aplica un sistema de scoring estructural (basado en PACF).
    - Interpola tramos con pocos nulos si es posible.
    - Utiliza una heurística de fallback si los métodos principales no tienen éxito.
    - Restaura el tramo seleccionado a su escala original.
    - Puede mostrar o guardar visualizaciones del tramo seleccionado.
    - Documenta todos los descartes y decisiones de forma opcional.

    Parámetros:
    ----------
    df : con una única columna representando la serie temporal a analizar.
    nombre_archivo_entrada : Nombre base utilizado para los archivos de salida. 
    input_func :Función de entrada del usuario (por defecto, `input`).
    prop_ventana : Proporción inicial de la ventana deslizante respecto al largo total.
    prop_ventana_min : Proporción mínima permitida para la ventana.
    paso : Cantidad de posiciones que se avanza al deslizar la ventana.
    step_ventana : Paso de reducción del tamaño de la ventana al no encontrar tramos válidos.
    incluir_log_extendido_modelo : Si es True, se registran y retornan los descartes detallados.
    titulo_grafico : Título del gráfico del tramo seleccionado.
    nombre_archivo_grafico : Nombre base del archivo del gráfico generado.
    criterio_estricto_test : Si es True, se aplican umbrales más rigurosos para los tests ADF y KPSS.
    directorio_graficas : Carpeta donde guardar los gráficos generados.
    log_msg : Función para registrar mensajes del proceso.
    modo_auto : Si es True, omite interacción con el usuario y aplica decisiones automáticas.

    Retorna:
    -------
    Tuple[pd.Series, Optional[list]]
        - Serie seleccionada como tramo representativo (puede ser vacía si falla).

    """
    # Crear una serie vacía con el mismo tipo de datos que la columna original
    tramo_seleccionado = pd.Series(dtype=df.iloc[:, 0].dtype)

    # Inicializar variables auxiliares
    descartes_ext = None  
    metadata = {}         

    try:
        # Extraer la única columna del DataFrame 
        serie_original = df.iloc[:, 0]

        # Normalizamos la serie para selección de tramo
        media_serie = serie_original.mean()
        std_serie = serie_original.std()
        serie = (serie_original - media_serie) / std_serie


        # Informar al usuario que comienza el proceso de modelado
        log_msg("🧪 Ejecutando selección inteligente de tramo...")

        # Ejecutar la lógica principal de búsqueda del mejor tramo para modelado ARMA
        metadata, tramo_seleccionado, descartes_ext = seleccionar_tramo_inteligente_para_arma(
            serie=serie,
            prop_ventana=prop_ventana,
            prop_ventana_min=prop_ventana_min,
            paso=paso,
            step_ventana=step_ventana,
            criterio_estricto_tests=criterio_estricto_test,
            log_msg=log_msg
        )
        
        if metadata and "inicio" in metadata and "fin" in metadata:
            tramo_seleccionado = serie.iloc[metadata["inicio"]:metadata["fin"]]

        # Visualizar resultados 
        mostrar_resultados_modelado(
            metadata=metadata,
            tramo=tramo_seleccionado,
            descartes_ext=descartes_ext,
            incluir_log_extendido=incluir_log_extendido_modelo,
            input_func=input_func,
            titulo_grafico=titulo_grafico,
            nombre_archivo_grafico=nombre_archivo_grafico,
            res_dir=directorio_graficas,
            log_msg=log_msg,
            modo_auto=modo_auto
        )

    except Exception as e:
        # Captura cualquier error inesperado en el proceso y lo registra
        log_msg(f"❌ Error durante el modelado: {e}")
        tramo_seleccionado = None  

    # Devolver el tramo final seleccionado 
    return tramo_seleccionado
