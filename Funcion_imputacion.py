"""
MÓDULO: funcion_imputacion.py

Este módulo implementa una estrategia iterativa y personalizada para la imputación
de valores faltantes (NaNs) en series temporales.

La imputación se realiza identificando bloques de nulos en la serie y aplicando
modelos ARMA previamente obtenidos, combinando predicciones hacia adelante (forward)
y hacia atrás (backward).

Además, incorpora paralelización para optimizar el proceso y un control adaptativo
de convergencia, garantizando precisión y eficiencia en la restauración de datos.

────────────────────────────────────────────────────────────────────────────
📌 FUNCIONALIDADES PRINCIPALES:

1. Análisis y segmentación de bloques nulos:
   - `analizar_bloques_nulos()`: Identifica bloques consecutivos de valores NaN en la serie,
     recopilando información contextual sobre su entorno inmediato para facilitar la imputación.
   - `actualizar_contexto_bloque()`: Actualiza el contexto (valores libres a izquierda y derecha)
     de un bloque específico en una serie modificada.

2. Imputación puntual en índice:
   - `imputar_forward_en_idx()`: Entrena modelos ARIMA sobre datos previos para predecir el valor faltante
     hacia adelante, probando diferentes tamaños de contexto hasta convergencia.
   - `imputar_backward_en_idx()`: Similar al forward, pero entrena modelos invertidos en datos posteriores,
     simulando predicciones hacia atrás en el tiempo.

3. Imputación de bloques completos:
   - `imputar_bloque_arima_simple()`: Imputa bloques enteros combinando predicciones forward y backward,
     ajustando modelos ARIMA con parámetros configurables y combinando resultados según esquemas ponderados.

4. Combinación de predicciones:
   - `combinar_predicciones()`: Fusiona predicciones forward y backward aplicando diferentes esquemas de pesos
     (uniforme, lineal, escalado, sigmoidal, centrado), y determina la situación final del bloque imputado.

5. Imputación paralela y modular:
   - `_imputar_wrapper()`: Función envoltorio para ejecutar la imputación de un bloque con logging aislado,
     facilitando la ejecución en paralelo.
   - `imputar_en_paralelo()`: Ejecuta imputaciones en paralelo sobre múltiples bloques para optimizar tiempo.

6. Ciclo iterativo de imputación:
   - `imputar_bloques_arima_con_ciclo()`: Orquestador general que:
       - Identifica y actualiza bloques nulos.
       - Ejecuta imputaciones iterativas con control de reintentos y descartes automáticos.
       - Actualiza la serie con imputaciones progresivas.
       - Detiene el proceso cuando no quedan bloques imputables o no se realizan cambios.

────────────────────────────────────────────────────────────────────────────
RETORNO FINAL:
   - Serie con los nulos imputados.
"""
# ==============================================================
# 🧱 1. LIBRERÍAS ESTÁNDAR
# ==============================================================

from typing import Any, Dict, List, Optional, Tuple, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy

# ==============================================================
# 📦 2. LIBRERÍAS DE TERCEROS
# ==============================================================

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import time

# ==============================================================
# 🔧 3. MÓDULOS INTERNOS DEL PROYECTO
# ==============================================================

from Funcion_log import crear_logger

# ==============================================================
# ⚙️ 4. CONFIGURACIÓN GLOBAL DESDE CONFIG.PY
# ==============================================================

from config import (
    # Verbosidad y parámetros de imputación
    USAR_PARAMETROS_INICIALES_IMPUTACION,
    MODO_PESOS_IMPUTACION,

    # Reglas de estabilidad del modelo ARIMA
    FORZAR_ESTACIONARIDAD_IMPUTACION,
    FORZAR_INVERTIBILIDAD_IMPUTACION,

    # Parámetros de contexto y tamaño
    STEP_CONTEXTO_IMPUTACION,
    MARGEN_SEGURIDAD_IMPUTACION,
    MAX_CONTEXTO_IMPUTACION,
    FACTOR_LARGO_IMPUTACION,
    MIN_BLOQUE_IMPUTACION,
    TAMAÑO_LOTE_BLOQUES_A_ANALIZAR
)


def analizar_bloques_nulos(serie: pd.Series) -> List[Dict[str, Any]]:
    """
    Identifica todos los bloques consecutivos de valores nulos (NaN) en una serie y
    recopila información contextual de su entorno inmediato.

    Para cada bloque se captura:
    - Posición (inicio, fin del bloque).
    - Tamaño (número de NaNs consecutivos).
    - Número de valores no nulos contiguos antes y después del bloque.
    - Si está ubicado en un extremo (inicio/final de la serie).
    - Lista vacía 'imputaciones' para registrar valores imputados posteriormente.
    - Contador 'reintentos' para controlar intentos fallidos de imputación.

    Parámetros:
        serie (pd.Series): Serie unidimensional que puede contener valores NaN.

    Retorna:
        List[Dict[str, Any]]: Lista de bloques con información contextual.
    """

    bloques = []      
    i = 0             
    n = len(serie)    

    while i < n:
        if pd.isna(serie.iloc[i]):
            # Detectado inicio de bloque de NaNs
            idx_ini = i

            # Avanzar hasta el final del bloque de NaNs
            while i < n and pd.isna(serie.iloc[i]):
                i += 1
            idx_fin = i - 1

            # Buscar cuántos valores no nulos hay justo antes (a la izquierda)
            j = idx_ini - 1
            libres_izq = 0
            while j >= 0 and not pd.isna(serie.iloc[j]):
                libres_izq += 1
                j -= 1

            # Buscar cuántos valores no nulos hay justo después (a la derecha)
            k = idx_fin + 1
            libres_der = 0
            while k < n and not pd.isna(serie.iloc[k]):
                libres_der += 1
                k += 1

            # Determinar si el bloque está en un extremo de la serie
            if idx_ini == 0:
                extremo = "inicio"
            elif idx_fin == n - 1:
                extremo = "final"
            else:
                extremo = "ninguno"

            # Construir el diccionario del bloque
            bloque = {
                "inicio": idx_ini,                     # Índice inicial del bloque
                "fin": idx_fin,                        # Índice final del bloque
                "tamano": idx_fin - idx_ini + 1,       # Cantidad de NaNs en el bloque
                "libres_izq": libres_izq,              # No nulos contiguos a la izquierda
                "libres_der": libres_der,              # No nulos contiguos a la derecha
                "situacion": "Sin Imputar",            # Estado inicial (sin procesar)
                "extremo": extremo,                    # Ubicación relativa en la serie
                "imputaciones": [],                    # Lista vacía para registrar imputaciones
                "reintentos": 0                        # Número de intentos fallidos hasta ahora
            }

            # Añadir el bloque a la lista
            bloques.append(bloque)

        else:
            # Valor no nulo, avanzar al siguiente
            i += 1

    return bloques

def actualizar_contexto_bloque(
    bloque: Dict[str, Any],
    serie: pd.Series
) -> Dict[str, Any]:
    """
    Recalcula únicamente el contexto inmediato (valores contiguos no nulos a izquierda y derecha)
    de un bloque de NaNs en una serie actualizada tras una iteración de imputación. 

    No modifica inicio, fin, situación, extremo ni otras propiedades del bloque.

    Parámetros:
        bloque (Dict[str, Any]): Información del bloque.
        serie (pd.Series): Serie posiblemente modificada (con imputaciones).

    Retorna:
        Dict[str, Any]: Mismo bloque, con 'libres_izq' y 'libres_der' actualizados.
    """
    idx_ini = bloque["inicio"]
    idx_fin = bloque["fin"]

    # Valores no nulos contiguos a la izquierda
    j = idx_ini - 1
    libres_izq = 0
    while j >= 0 and not pd.isna(serie.iloc[j]):
        libres_izq += 1
        j -= 1

    # Valores no nulos contiguos a la derecha
    k = idx_fin + 1
    libres_der = 0
    while k < len(serie) and not pd.isna(serie.iloc[k]):
        libres_der += 1
        k += 1

    # Actualización del contexto
    bloque["libres_izq"] = libres_izq
    bloque["libres_der"] = libres_der

    return bloque

def imputar_forward_en_idx(
    serie: pd.Series,
    idx: int,
    p: int,
    q: int,
    min_contexto: int,
    max_contexto: int,
    step_contexto: int = STEP_CONTEXTO_IMPUTACION,
    d: int = 0,
    enforce_stationarity: bool = FORZAR_ESTACIONARIDAD_IMPUTACION,
    enforce_invertibility: bool = FORZAR_INVERTIBILIDAD_IMPUTACION,
    parametros: Optional[List[float]] = None,
    usar_parametros_iniciales: bool = USAR_PARAMETROS_INICIALES_IMPUTACION,
    log_msg: Optional[Callable[[str], None]] = None,
) -> Optional[float]:
    """
    Imputa el valor en la posición `idx` de una serie temporal utilizando un modelo ARIMA
    entrenado hacia adelante, es decir, solo con observaciones anteriores al punto a imputar.

    La función prueba distintas longitudes de contexto, desde `min_contexto` hasta `max_contexto`,
    en incrementos de `step_contexto`, buscando el primer modelo que converja y produzca una
    predicción válida.

    Parámetros:
        serie (pd.Series): Serie temporal con posibles valores NaN.
        idx (int): Índice posicional (no label) del valor a imputar. Debe ser un NaN.
        p (int): Orden autorregresivo del modelo ARIMA.
        q (int): Orden de media móvil del modelo ARIMA.
        min_contexto (int): Mínimo tamaño del contexto (nº de observaciones previas al índice).
        max_contexto (int): Máximo tamaño del contexto.
        step_contexto (int): Incremento entre tamaños de contexto a probar.
        d (int): Orden de diferenciación del modelo ARIMA.
        enforce_stationarity (bool): Forzar estacionariedad en el ajuste.
        enforce_invertibility (bool): Forzar invertibilidad en el ajuste.
        parametros (Optional[List[float]]): Parámetros iniciales para el optimizador.
        usar_parametros_iniciales (bool): Si True, intenta ajustar primero con `parametros`.
        log_msg (Optional[Callable]): Función para registrar logs.

    Retorna:
        Optional[float]: Valor imputado si algún modelo converge; None si todos fallan.
    """
    
    for context_len in range(min_contexto, max_contexto + 1, step_contexto):
        try:
            # Obtener contexto hacia atrás desde idx
            contexto = serie.iloc[:idx].tail(context_len).reset_index(drop=True)

            if len(contexto) < context_len:
                continue  # Contexto insuficiente

            log_msg(f"\n ➡️ [FWD] idx={idx}, tamaño contexto ({context_len})")

            modelo = ARIMA(
                contexto,
                order=(p, d, q),
                enforce_stationarity=enforce_stationarity,
                enforce_invertibility=enforce_invertibility
            )
            # Primer intento con parámetros iniciales, si se desea
            if parametros and usar_parametros_iniciales:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        fitted = modelo.fit(start_params=parametros)
                    if fitted.mle_retvals.get("converged", False):
                        pred = fitted.forecast(steps=1).iloc[0]
                        return pred, context_len
                    log_msg(" ⚠️ No convergió con start_params, probando sin ellos...")
                except Exception as e:
                    log_msg(f" ⚠️ Error con start_params: {e}")

            # Segundo intento sin parámetros
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                fitted = modelo.fit()

            if fitted.mle_retvals.get("converged", False):
                pred = fitted.forecast(steps=1).iloc[0]
                log_msg(f" 🔮 Predicción forward en idx={idx}: {pred:.5f}")
                return pred, context_len

            log_msg(f" ⚠️ Modelo forward no convergió con tamaño contexto {context_len}")

        except Exception as e:
            log_msg(f" ❌ Error forward en idx={idx}, tamaño contexto={context_len}: {e}")

    return None, None

def imputar_backward_en_idx(
    serie: pd.Series,
    idx: int,
    p: int,
    q: int,
    min_contexto: int,
    max_contexto: int,
    step_contexto: int = STEP_CONTEXTO_IMPUTACION,
    d: int = 0,
    enforce_stationarity: bool = FORZAR_ESTACIONARIDAD_IMPUTACION,
    enforce_invertibility: bool = FORZAR_INVERTIBILIDAD_IMPUTACION,
    parametros: Optional[List[float]] = None,
    usar_parametros_iniciales: bool = USAR_PARAMETROS_INICIALES_IMPUTACION,
    log_msg: Optional[Callable[[str], None]] = None,
) -> Optional[float]:
    """
    Imputa el valor en la posición `idx` de una serie temporal utilizando un modelo ARIMA,
    entrenado sobre los datos posteriores al índice, pero invertidos para simular una 
    predicción "hacia atrás" en el tiempo.

    Se prueban distintas longitudes de contexto, desde `min_contexto` hasta `max_contexto`,
    y se selecciona la primera predicción válida generada por un modelo que haya convergido.

    Parámetros:
        serie (pd.Series): Serie temporal con valores NaN.
        idx (int): Índice posicional del valor a imputar. Debe corresponder a un NaN.
        p (int): Orden autorregresivo (AR) del modelo ARIMA.
        q (int): Orden de media móvil (MA) del modelo ARIMA.
        min_contexto (int): Número mínimo de observaciones a usar como contexto.
        max_contexto (int): Número máximo de observaciones a usar como contexto.
        step_contexto (int): Paso entre tamaños de contexto a evaluar.
        d (int): Orden de diferenciación.
        enforce_stationarity (bool): Si True, fuerza que el modelo sea estacionario.
        enforce_invertibility (bool): Si True, fuerza que el modelo sea invertible.
        parametros (Optional[List[float]]): Lista de parámetros iniciales para el modelo ARIMA.
        usar_parametros_iniciales (bool): Si True, intenta ajustar primero con `parametros`.
        log_msg (Optional[Callable]): Función para registrar logs.

    Retorna:
        Optional[float]: Valor imputado si el modelo converge; None en caso contrario.
    """
    for context_len in range(min_contexto, max_contexto + 1, step_contexto):
        try:
            # Extraer contexto futuro y revertirlo
            contexto = (
                serie.iloc[idx + 1:]
                .head(context_len)[::-1]
                .reset_index(drop=True)
            )
            if len(contexto) < context_len:
                continue  # Contexto insuficiente

            log_msg(f"\n ➡️ [BWD] idx={idx}, tamaño contexto ({context_len})\n{contexto.values}:")

            modelo = ARIMA(
                contexto,
                order=(p, d, q),
                enforce_stationarity=enforce_stationarity,
                enforce_invertibility=enforce_invertibility
            )

            # Primer intento con parámetros iniciales
            if parametros and usar_parametros_iniciales:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        fitted = modelo.fit(start_params=parametros)
                    if fitted.mle_retvals.get("converged", False):
                        return fitted.forecast(steps=1).iloc[0], context_len
                    log_msg(" ⚠️ No convergió con start_params, probando sin ellos...")
                except Exception as e:
                    log_msg(f" ⚠️ Error con start_params: {e}")

            # Segundo intento sin parámetros
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                fitted = modelo.fit()

            if fitted.mle_retvals.get("converged", False):
                pred = fitted.forecast(steps=1).iloc[0]
                log_msg(f" 🔮 Predicción backward en idx={idx}: {pred:.5f}")
                return pred, context_len

            log_msg(f" ⚠️ Modelo backward no convergió con tamaño contexto {context_len}")

        except Exception as e:
            log_msg(f" ❌ Error backward en idx={idx}, tamaño contexto={context_len}: {e}")

    return None, None

def combinar_predicciones(
    serie_imp: pd.Series,
    bloque: Dict[str, Any],
    pred_f: Dict[int, float],
    pred_b: Dict[int, float],
    modo_pesos: str = MODO_PESOS_IMPUTACION,
    log_msg: Optional[Callable[[str], None]] = None,
) -> Tuple[pd.Series, Dict[str, Any], bool]:
    """
    Imputa un bloque de valores nulos combinando predicciones forward y backward,
    según un esquema de ponderación definido, y actualiza el bloque con la lista de
    imputaciones realizadas. Si no hay cambios en la imputación respecto a la iteración
    anterior, se considera un intento fallido.

    Modos de combinación disponibles:
        - 'uniforme':          ambos extremos pesan igual.
        - 'lineal':            mayor peso al predictor más cercano.
        - 'escalado':          variante acentuada del lineal.
        - 'sigmoidal':         transición suave entre extremos.
        - 'centrado':          mayor peso al centro del bloque.

    Parámetros:
        serie_imp (pd.Series): Serie que será modificada (in-place).
        bloque (Dict[str, Any]): Información del bloque con claves como:
                                 'inicio', 'fin', 'tamano', 'imputaciones', etc.
        pred_f (Dict[int, float]): Predicciones en dirección forward por índice.
        pred_b (Dict[int, float]): Predicciones en dirección backward por índice.
        modo_pesos (str): Modo de ponderación entre pred_f y pred_b.
        log_msg (Callable, opcional): Función de log para registrar mensajes.

    Retorna:
        Tuple:
            - pd.Series: Serie modificada con imputaciones.
            - Dict[str, Any]: Bloque actualizado con situación e imputaciones.
            - bool: True si al menos un valor fue imputado.
    """

    # Inicialización y verificación de entrada

    idx_ini, idx_fin = bloque["inicio"], bloque["fin"]
    largo = bloque["tamano"]
    indices = list(range(idx_ini, idx_fin + 1))

    tiene_f = len(pred_f) == largo
    tiene_b = len(pred_b) == largo

    if not tiene_f and not tiene_b:
        bloque["situacion"] = "fallido"

        log_msg("❌ Ninguna predicción disponible. Bloque marcado como fallido.")
        return serie_imp, bloque, False

    if tiene_f and tiene_b:
        modo_pred = "forward+backward"
        if modo_pesos not in {"uniforme", "lineal", "escalado", "sigmoidal", "centrado"}:
            raise ValueError(f"Modo de pesos no válido: '{modo_pesos}'")
    elif tiene_f:
        modo_pred = "solo_forward"
    else:
        modo_pred = "solo_backward"


    log_msg(f"✅ Modo de imputación: {modo_pred}")

    imputaciones_realizadas = []

    for j, idx in enumerate(indices):
        if modo_pred == "forward+backward":
            val_f = pred_f[idx]
            val_b = pred_b[idx]

            if modo_pesos == "uniforme":
                wf, wb = 0.5, 0.5
            elif modo_pesos == "lineal":
                wf = 1 - (j + 1) / (largo + 1)
                wb = 1 - wf
            elif modo_pesos == "escalado":
                wf = (1 - (j + 1) / (largo + 1)) ** 2
                wb = 1 - wf
            elif modo_pesos == "sigmoidal":
                x = (j + 1) / (largo + 1)
                wf = 1 / (1 + np.exp(10 * (x - 0.5)))
                wb = 1 - wf
            elif modo_pesos == "centrado":
                x = abs((j + 1) - (largo + 1) / 2) / ((largo + 1) / 2)
                wf = 1 - x
                wb = 1 - wf

            valor = wf * val_f + wb * val_b

        elif modo_pred == "solo_forward":
            valor = pred_f[idx]

        elif modo_pred == "solo_backward":
            valor = pred_b[idx]

        serie_imp.iloc[idx] = valor
        imputaciones_realizadas.append({"indice": idx, "valor": valor})

    bloque["situacion"] = modo_pred

    # Comparar con imputaciones anteriores
    if imputaciones_realizadas == bloque.get("imputaciones", []):
        bloque["reintentos"] += 1

        log_msg("⚠️ Reintento sin cambios. Incrementando contador.")
        return serie_imp, bloque, False

    bloque["imputaciones"] = imputaciones_realizadas


    log_msg(f"✅ Imputación finalizada con método: {modo_pred}")
    return serie_imp, bloque, True

def imputar_bloque_arima_simple(
    serie: pd.Series,
    bloque: Dict[str, Any],
    orden: Tuple[int, int, int],
    margen_seguridad: int = MARGEN_SEGURIDAD_IMPUTACION,
    max_contexto: int = MAX_CONTEXTO_IMPUTACION,
    factor_largo: int = FACTOR_LARGO_IMPUTACION,
    step_contexto: int = STEP_CONTEXTO_IMPUTACION,
    enforce_stationarity: bool = FORZAR_ESTACIONARIDAD_IMPUTACION,
    enforce_invertibility: bool = FORZAR_INVERTIBILIDAD_IMPUTACION,
    parametros: Optional[List[float]] = None,
    usar_parametros_iniciales: bool = USAR_PARAMETROS_INICIALES_IMPUTACION,
    modo_pesos: str = "lineal",
    log_msg: Optional[Callable[[str], None]] = None,
) -> Tuple[pd.Series, Dict[str, Any], bool]:
    """
    Imputa un bloque de valores nulos en una serie temporal usando modelos ARIMA
    hacia adelante (forward) y hacia atrás (backward), combinando predicciones si es posible.

    La función ajusta modelos ARIMA entrenados en valores disponibles antes o después del bloque.
    Si ambos métodos son viables, se combinan ponderadamente.

    Parámetros:
        serie (pd.Series): Serie temporal con valores numéricos y NaNs.
        bloque (Dict[str, Any]): Información del bloque a imputar con claves mínimas:
            - 'inicio', 'fin', 'tamano', 'libres_izq', 'libres_der'.
        orden (Tuple[int, int, int]): Orden del modelo ARIMA (p, d, q).
        margen_seguridad (int): Observaciones mínimas adicionales. Default: config.
        max_contexto (int): Límite superior del contexto.
        factor_largo (int): Multiplicador para dimensionar el contexto base.
        step_contexto (int): Incremento entre longitudes de contexto a probar.
        enforce_stationarity (bool): Forzar estacionariedad en el modelo.
        enforce_invertibility (bool): Forzar invertibilidad.
        parametros (List[float], opcional): Parámetros iniciales para el modelo ARIMA.
        usar_parametros_iniciales (bool): Activar intento con parámetros dados.
        modo_pesos (str): Esquema de combinación de predicciones. Ej: 'lineal', 'uniforme'.
        log_msg (Callable, opcional): Logger externo para trazabilidad.

    Retorna:
        Tuple[pd.Series, Dict[str, Any], bool]:
            - Serie con el bloque imputado.
            - Diccionario del bloque actualizado (incluye 'situacion').
            - Booleano que indica si se imputó al menos un valor.
    """

    # Preparación de variables clave y evaluación de contexto
    p, q = orden
    d = 0  # se asume diferenciación 0
    idx_ini, idx_fin = bloque["inicio"], bloque["fin"]
    largo = bloque["tamano"]
    valores_izq = bloque["libres_izq"]
    valores_der = bloque["libres_der"]

    min_modelo = p + d + q + margen_seguridad
    min_tamano = min(max_contexto, largo * factor_largo)
    min_contexto = max(min_modelo, min_tamano)

    log_msg(f"\n📦 Imputando bloque {idx_ini}-{idx_fin}")
    log_msg(f"🧮 min_contexto = max({min_modelo}, {min_tamano}) = {min_contexto}")
    log_msg(f"↩️ Valores libres izquierda: {valores_izq}, ↪️ derecha: {valores_der}")

    # Determinar si se recalcula forward/backward
    situacion_previa = bloque.get("situacion")
    hacer_forward = situacion_previa in (None, "fallido", "solo_backward", "Sin Imputar")
    hacer_backward = situacion_previa in (None, "fallido", "solo_forward", "Sin Imputar")

    serie_imp = serie.copy()
    serie_imp_forward = serie.copy()
    serie_imp_backward = serie.copy()
    pred_f, pred_b = {}, {}

    contexto_efectivo_forward = min_contexto
    contexto_efectivo_backward = min_contexto
    #  Reutilización de predicciones previas si aplica
    if not hacer_forward:
        log_msg("ℹ️ No se recalcula forward; se reutilizan valores previos.")
        for idx in range(idx_ini, idx_fin + 1):
            val = serie_imp_backward.iloc[idx]
            if pd.notna(val):
                pred_f[idx] = val
                serie_imp_backward.iloc[idx] = np.nan  # limpiar para permitir nuevo backward

    if not hacer_backward:
        log_msg("ℹ️ No se recalcula backward; se reutilizan valores previos.")
        for idx in range(idx_ini, idx_fin + 1):
            val = serie_imp_forward.iloc[idx]
            if pd.notna(val):
                pred_b[idx] = val
                serie_imp_forward.iloc[idx] = np.nan  # limpiar para permitir nuevo forward

    # Imputación FORWARD si se cumplen condiciones
    if hacer_forward and valores_izq >= min_contexto:
        log_msg("➡️ Intentando imputación forward...")
        fallo_forward = False
        for idx in range(idx_ini, idx_fin + 1):
            log_msg(f"🔎 [ANTES] idx={idx}, valor en serie_imp_forward[{idx}] = {serie_imp_forward.iloc[idx]}")            
            pred, contexto_actualizado_forward = imputar_forward_en_idx(
                serie=serie_imp_forward,
                idx=idx,
                p=p,
                d=d,
                q=q,
                min_contexto=contexto_efectivo_forward,
                max_contexto=valores_izq,
                step_contexto=step_contexto,
                enforce_stationarity=enforce_stationarity,
                enforce_invertibility=enforce_invertibility,
                parametros=parametros,
                usar_parametros_iniciales=usar_parametros_iniciales,
                log_msg=log_msg
            )
            if pred is None:
                log_msg(f"❌ Predicción fallida en idx={idx}, se aborta forward.")
                fallo_forward = True
                break
            pred_f[idx] = pred
            serie_imp_forward.iloc[idx] = pred
            if contexto_actualizado_forward is not None:
                contexto_efectivo_forward = contexto_actualizado_forward
            log_msg(f"✅ [DESPUÉS] idx={idx}, imputado = {pred:.5f}")

        if fallo_forward:
            for idx in pred_f:
                serie_imp_forward.iloc[idx] = np.nan
            pred_f = {}

    #  Imputación BACKWARD si se cumplen condiciones
    if hacer_backward and valores_der >= min_contexto:
        log_msg("⬅️ Intentando imputación backward...")
        fallo_backward = False
        for idx in reversed(range(idx_ini, idx_fin + 1)):
            log_msg(f"🔎 [ANTES] idx={idx}, valor en serie_imp_backward[{idx}] = {serie_imp_backward.iloc[idx]}")
            pred,contexto_actualizado_backward = imputar_backward_en_idx(
                serie=serie_imp_backward,
                idx=idx,
                p=p,
                d=d,
                q=q,
                min_contexto=contexto_efectivo_backward,
                max_contexto=valores_der,
                step_contexto=step_contexto,
                enforce_stationarity=enforce_stationarity,
                enforce_invertibility=enforce_invertibility,
                parametros=parametros,
                usar_parametros_iniciales=usar_parametros_iniciales,
                log_msg=log_msg
            )
            if pred is None:
                log_msg(f"❌ Predicción fallida en idx={idx}, se aborta backward.")
                fallo_backward = True
                break
            pred_b[idx] = pred
            serie_imp_backward.iloc[idx] = pred
            if contexto_actualizado_backward is not None:
                contexto_efectivo_backward=contexto_actualizado_backward
            log_msg(f"✅ [DESPUÉS] idx={idx}, imputado = {pred:.5f}")

        if fallo_backward:
            for idx in pred_b:
                serie_imp_backward.iloc[idx] = np.nan
            pred_b = {}

    # Combinación final de predicciones con esquema de pesos
    serie_imp, bloque, imputado = combinar_predicciones(
        serie_imp=serie_imp,
        bloque=bloque,
        pred_f=pred_f,
        pred_b=pred_b,
        modo_pesos=modo_pesos,
        log_msg=log_msg
    )

    return serie_imp, bloque, imputado

def _imputar_wrapper(kwargs, bloque_id):
    """
    Función envoltorio (wrapper) para ejecutar imputación ARIMA sobre un bloque específico
    en paralelo o de forma modular, capturando además el log generado durante el proceso.

    Este wrapper:
    - Inyecta un logger local en `kwargs`.
    - Ejecuta `imputar_bloque_arima_simple` con los parámetros recibidos.
    - Retorna los resultados del bloque junto con los logs generados.

    Parámetros:
        kwargs (dict): Diccionario con argumentos requeridos por `imputar_bloque_arima_simple`.
                       Se modificará internamente para incluir un logger.
        bloque_id (Any): Identificador único del bloque (puede ser índice o ID hash).

    Retorna:
        dict: Contiene:
            - 'bloque_id': ID del bloque imputado.
            - 'serie_parcial': Serie imputada para el bloque.
            - 'bloque_actualizado': Información estructural del bloque tras imputación.
            - 'imputado': Booleano indicando si se imputó o no.
            - 'logs': Lista de strings con los mensajes de log generados localmente.
    """
    # Crear un buffer local para capturar mensajes del log sin imprimir en consola
    log_buffer_local = []

    # Inyectar el logger al diccionario de argumentos para imputación
    kwargs["log_msg"] = crear_logger(verbose=False, log_buffer=log_buffer_local)

    # Ejecutar la función de imputación sobre el bloque
    serie_parcial, bloque_actualizado, imputado = imputar_bloque_arima_simple(**kwargs)

    # Devolver resultados y trazas de ejecución locales
    return {
        "bloque_id": bloque_id,
        "serie_parcial": serie_parcial,
        "bloque_actualizado": bloque_actualizado,
        "imputado": imputado,
        "logs": log_buffer_local
    }

def imputar_en_paralelo(
    bloques: list,
    serie: pd.Series,
    parametros_comunes: dict
) -> list:
    """
    Ejecuta la imputación ARIMA en paralelo sobre múltiples bloques de una serie temporal.

    Para cada bloque, crea una copia independiente de la serie y ejecuta la función
    de imputación usando un wrapper que captura logs localmente.

    Parámetros:
        bloques (list): Lista de bloques a imputar. Cada bloque es un diccionario con información estructural.
        serie (pd.Series): Serie temporal original con posibles valores faltantes (NaNs).
        parametros_comunes (dict): Diccionario con parámetros comunes necesarios para la función de imputación.

    Retorna:
        list: Lista de diccionarios con resultados por bloque, incluyendo:
            - 'bloque_id': Identificador del bloque.
            - 'serie_parcial': Serie imputada parcialmente para ese bloque.
            - 'bloque_actualizado': Información actualizada del bloque tras la imputación.
            - 'imputado': Booleano indicando si se realizó imputación.
            - 'logs': Lista con mensajes de log generados localmente.
    """
    resultados = []
    with ProcessPoolExecutor() as executor:
        futures = []

        for bloque in bloques:
            # Preparar kwargs independientes por bloque
            kwargs = dict(parametros_comunes)
            kwargs["bloque"] = bloque
            kwargs["serie"] = deepcopy(serie)  # Cada proceso trabaja con su copia

            bloque_id = bloque["inicio"]  # Usamos esto como identificador

            future = executor.submit(_imputar_wrapper, kwargs, bloque_id)
            futures.append(future)

        for future in as_completed(futures):
            resultado = future.result()
            resultados.append(resultado)

    return resultados

def imputar_bloques_arima_con_ciclo(
    serie: pd.Series,
    modelo: Dict[str, Any],
    usar_parametros_iniciales: bool = USAR_PARAMETROS_INICIALES_IMPUTACION,
    margen_seguridad: int = MARGEN_SEGURIDAD_IMPUTACION,
    max_contexto: int = MAX_CONTEXTO_IMPUTACION,
    factor_largo: int = FACTOR_LARGO_IMPUTACION,
    min_bloque: int = MIN_BLOQUE_IMPUTACION,
    step_contexto: int = STEP_CONTEXTO_IMPUTACION,
    enforce_stationarity: bool = FORZAR_ESTACIONARIDAD_IMPUTACION,
    enforce_invertibility: bool = FORZAR_INVERTIBILIDAD_IMPUTACION,
    modo_pesos: str = "lineal",
    log_msg: Optional[Callable[[str], None]] = None,
) -> Tuple[pd.Series, List[Dict[str, Any]]]:
    """
    Imputa valores nulos en una serie temporal por bloques, utilizando un modelo ARIMA
    en un ciclo iterativo que evalúa contexto, reintentos y condiciones estructurales.

    La función identifica bloques de NaNs consecutivos, estima predicciones hacia adelante
    (forward) y hacia atrás (backward), y combina ambas direcciones para imputar valores
    según un esquema de ponderación configurable.

    En cada iteración:
        - Se actualiza el contexto de los bloques.
        - Se imputan aquellos bloques reintentables.
        - Se descartan automáticamente los que:
            - Están en extremos con predicción unidireccional.
            - No han cambiado en las últimas 5 iteraciones.

    Parámetros:
        serie (pd.Series): Serie de entrada con valores NaN.
        modelo (Dict[str, Any]): Diccionario con configuración del modelo ARIMA.
        usar_parametros_iniciales (bool): Usar parámetros del modelo si están definidos.
        margen_seguridad (int): Número mínimo de puntos requeridos alrededor del bloque.
        max_contexto (int): Límite superior del contexto usado para entrenar el modelo.
        factor_largo (int): Factor de escala del contexto según tamaño del bloque.
        min_bloque (int): Tamaño mínimo de bloque imputable.
        step_contexto (int): Incremento del contexto si la predicción falla.
        enforce_stationarity (bool): Forzar estacionaridad en el modelo ARIMA.
        enforce_invertibility (bool): Forzar invertibilidad en el modelo ARIMA.
        modo_pesos (str): Método de combinación de predicciones ('lineal', 'sigmoidal', etc.).
        log_msg (Callable, opcional): Función para imprimir mensajes de log.

    Retorna:
        Tuple:
            - pd.Series: Serie imputada.
            - List[Dict[str, Any]]: Lista de bloques con su estado final.
    """

    # 🔧 Inicialización
    serie = serie.copy().reset_index(drop=True)
    bloques_global = []
    iteracion = 0

    orden = modelo.get("orden", (0, 0, 0))
    parametros = modelo.get("parametros", None)

    bloques = analizar_bloques_nulos(serie)
    # Inicializar bloques_global al principio
    bloques_global = [b.copy() for b in analizar_bloques_nulos(serie)]

    # Ciclo principal de imputación
    while True:
        log_msg(f"\n🔄 analizando bloques (iteración {iteracion + 1})...")
        
        # Recalcular el contexto de los bloques existentes tras posibles imputaciones
        bloques_global = [actualizar_contexto_bloque(b, serie) for b in bloques_global]

        # Filtrar los bloques que siguen siendo válidos para intentar imputar
        situaciones_reintentables = {"fallido", "solo_forward", "solo_backward", "Sin Imputar"}

        bloques = [
            b for b in bloques_global
            if b["tamano"] >= min_bloque
            and b.get("situacion") in situaciones_reintentables
            and not (
                b["situacion"] in {"solo_forward", "solo_backward"} and b["extremo"] in {"inicio", "final"}
            )
            and b["reintentos"] <= 5
        ]

        if not bloques:
            log_msg("✅ Todos los bloques fueron imputados o descartados.")
            break

        log_msg(f"📌 Bloques reintentables esta iteración: {[ (b['inicio'], b['fin'], b['situacion']) for b in bloques ]}")
        log_msg(f"\n🔁 Iteración #{iteracion + 1} - bloques a procesar: {len(bloques)}")
        imputaciones_esta_iteracion = 0

        # Parámetros comunes para todos los bloques
        parametros_comunes = {
            "orden": orden,
            "margen_seguridad": margen_seguridad,
            "max_contexto": max_contexto,
            "factor_largo": factor_largo,
            "step_contexto": step_contexto,
            "enforce_stationarity": enforce_stationarity,
            "enforce_invertibility": enforce_invertibility,
            "parametros": parametros,
            "usar_parametros_iniciales": usar_parametros_iniciales,
            "modo_pesos": modo_pesos,
        }

        # Imputación paralela por lotes para evitar sobrecarga de memoria
        tamano_lote = TAMAÑO_LOTE_BLOQUES_A_ANALIZAR  # Número máximo de bloques a imputar simultáneamente 
        resultados = []

        for i in range(0, len(bloques), tamano_lote):
            lote = bloques[i:i + tamano_lote]
            log_msg(f"\n📦 Imputando lote {i // tamano_lote + 1} con {len(lote)} bloques (de {len(bloques)} en total)...")
            
            t_ini = time.time()
            resultado_lote = imputar_en_paralelo(lote, serie, parametros_comunes)
            t_fin = time.time()

            log_msg(f"⏱️ Tiempo lote #{i // tamano_lote + 1}: {t_fin - t_ini:.2f} segundos.")
            resultados.extend(resultado_lote)
        # Procesamiento de resultados
        for resultado in resultados:
            bloque_actualizado = resultado["bloque_actualizado"]
            imputado = resultado["imputado"]
            serie_parcial = resultado["serie_parcial"]
            logs = resultado["logs"]

            bloque_slice = slice(bloque_actualizado["inicio"], bloque_actualizado["fin"] + 1)

            log_msg(f"\n🧩 Bloque {bloque_actualizado['inicio']}-{bloque_actualizado['fin']} procesado:")
            for msg in logs:
                log_msg(msg)
            log_msg(f"   ↪️ Situación final del bloque: {bloque_actualizado['situacion']}")
            log_msg(f"   🔍 Valores imputados: {serie_parcial.iloc[bloque_slice].dropna().values}")

            if imputado:
                serie.update(serie_parcial)
                imputaciones_esta_iteracion += 1
                log_msg(f"   ✅ Serie actualizada con imputación en bloque {bloque_actualizado['inicio']}-{bloque_actualizado['fin']}")
            else:
                log_msg(f"   ❌ Bloque {bloque_actualizado['inicio']}-{bloque_actualizado['fin']} no imputado.")

            # Actualizar el estado global del bloque
            for i, b in enumerate(bloques_global):
                if b["inicio"] == bloque_actualizado["inicio"]:
                    bloques_global[i] = bloque_actualizado
                    break

        # Detener si no se imputó nada
        if imputaciones_esta_iteracion == 0:
            log_msg("⚠️ Ningún bloque se pudo imputar en esta iteración. Deteniendo.")
            break
            # Resumen parcial tras la iteración
        log_msg("\n📋 Resumen de imputación en esta iteración:")
        for b in bloques_global:
            log_msg(f" - Bloque {b['inicio']}-{b['fin']}: situación='{b['situacion']}', reintentos={b['reintentos']}, imputados={len(b['imputaciones'])}")        
        iteracion += 1

    # Resumen final de imputación
    log_msg("\n📋 Resumen final de imputación por bloques:")
    for b in bloques_global:
        log_msg(f" - Bloque {b['inicio']}-{b['fin']}: situación='{b['situacion']}', reintentos={b['reintentos']}, imputados={len(b['imputaciones'])}")

    return serie, bloques_global
