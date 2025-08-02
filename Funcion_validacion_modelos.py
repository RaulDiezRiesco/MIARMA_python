"""
MÓDULO: Funcion_validacion_modelos.py

Este módulo implementa el flujo completo para evaluar, validar y seleccionar modelos ARMA 
previamente entrenados.

Su objetivo es identificar el modelo más robusto y estadísticamente sólido entre múltiples 
combinaciones (p, q), para la imputación de valores nulos en posteriores módulos.

────────────────────────────────────────────────────────────────────────────
📌 FUNCIONALIDADES PRINCIPALES:

1. Filtrado de modelos convergidos:
   - `seleccionar_modelos_convergidos_desde_df()`: Extrae modelos ARMA válidos desde un DataFrame
     de resultados, descartando los que no tienen parámetros o residuos.

2. Validaciones estadísticas aplicadas sobre residuos:
   - `realizar_prueba_normalidad()`: Evalúa la normalidad con Shapiro-Wilk y D’Agostino-Pearson.
   - `realizar_prueba_autocorrelacion()`: Detecta autocorrelación con la prueba de Ljung-Box.
   - `realizar_prueba_heterocedasticidad()`: Verifica homocedasticidad con la prueba ARCH.

3. Selección jerárquica de modelos:
   - `seleccionar_modelo_optimo()`: Aplica una lógica de priorización:
        1) validación completa,
        2) validación parcial (solo autocorrelación y heterocedasticidad),
        3) mejor AIC o BIC disponible.

4. Evaluación orquestada:
   - `seleccionar_mejor_modelo_desde_df()`: Evalúa un conjunto de modelos, aplica validaciones
     y selecciona el más adecuado con logs extendidos y resumen.

────────────────────────────────────────────────────────────────────────────
RETORNO FINAL:
   - Modelo escogido para la imputación. 
"""


# =============================================================
# 🧱 1. LIBRERÍAS ESTÁNDAR
# =============================================================

from typing import Any, Dict, List, Optional, Tuple, Callable

# =============================================================
# 📦 2. LIBRERÍAS DE TERCEROS
# =============================================================

import numpy as np
import pandas as pd
from scipy.stats import normaltest, shapiro
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

# =============================================================
# ⚙️ 3. CONFIGURACIÓN GLOBAL (config.py)
# =============================================================

from config import (
    # Umbrales de significancia
    ALPHA_TESTS_NORMALIDAD,
    ALPHA_TESTS_AUTOCORR,
    ALPHA_TESTS_HETEROCED,

    # Parámetros para pruebas estadísticas
    LIMITE_SHAPIRO,
    FALLOS_MAXIMOS_NORMALIDAD,
    MAX_LAGS_LB,
    MIN_LAGS_LB,

    # Criterio de selección
    CRITERIO_SELECCION_MODELO,

    # Activación de tests
    USAR_TESTS_NORMALIDAD,
    USAR_TESTS_AUTOCORRELACION,
    USAR_TESTS_HETEROCEDASTICIDAD,
)

def seleccionar_modelos_convergidos_desde_df(
    df_modelos: pd.DataFrame,
    log_msg: Optional[Callable[[str], None]] = None,
) -> List[Tuple[str, Tuple[int, int], Dict[str, Any]]]:
    """
    Extrae modelos ARMA convergidos y válidos desde un DataFrame de resultados.

    Esta función filtra modelos cuya convergencia fue exitosa (`converged == True`), su fase es 'refined'
    y que contienen tanto parámetros como residuos no nulos. El objetivo es preparar
    una lista de modelos aptos para evaluación posterior (e.g. validación o selección final).

    Parámetros:
        df_modelos (pd.DataFrame): DataFrame con resultados del grid search ARMA.
            Debe contener al menos las columnas: 'p', 'q', 'params', 'residuals', 'phase', 'converged'.
            Columnas opcionales: 'nombre_serie', 'aic', 'bic'.

        log_msg (Callable, opcional): Función para registrar mensajes (log). Si no se proporciona,
            los errores se omitirán silenciosamente.

    Retorna:
        List[Tuple[str, Tuple[int, int], Dict[str, Any]]]:
            Lista de modelos válidos en el formato:
            - nombre_serie (str): Identificador del modelo o serie.
            - (p, q): Orden del modelo ARMA.
            - dict: Información adicional del modelo, incluyendo:
                - params (List[float]): Coeficientes del modelo.
                - residuals (List[float]): Residuos del ajuste.
                - phase (str): Fase de entrenamiento (ej. 'initial', 'refined').
                - aic (float): AIC del modelo (si está disponible).
                - bic (float): BIC del modelo (si está disponible).
                - orden (Tuple[int, int]): Redundante, por compatibilidad.
    """
    modelos = []

    # Filtrar solo modelos convergidos y de fase 'refined'
    df_ok = df_modelos[
        (df_modelos["converged"] == True) &
        (df_modelos["phase"] == "refined")
    ]

    for _, row in df_ok.iterrows():
        try:
            # ─ Extracción básica de información ─
            p = int(row["p"])
            q = int(row["q"])
            nombre = row.get("nombre_serie", f"modelo_{p}_{q}")
            residuals = row.get("residuals")
            params = row.get("params")
            phase = row.get("phase", "unknown")
            aic = row.get("aic") if pd.notnull(row.get("aic")) else None
            bic = row.get("bic") if pd.notnull(row.get("bic")) else None

            # ─ Validación: deben existir residuos y parámetros ─
            if residuals is None or params is None:
                log_msg(f"⚠️ Modelo ({p},{q}) omitido: faltan residuos o parámetros.")
                continue

            # ─ Modelo válido: se añade al listado ─
            modelos.append((
                nombre,
                (p, q),
                {
                    "p": p,
                    "q": q,
                    "params": params,
                    "residuals": residuals,
                    "phase": phase,
                    "aic": aic,
                    "bic": bic,
                    "orden": (p, q)
                }
            ))

        except Exception as e:
            log_msg(f"⚠️ Error al procesar modelo ({row.get('p')},{row.get('q')}): {e}")

    return modelos

def realizar_prueba_normalidad(
    resid: np.ndarray,
    alpha: float = ALPHA_TESTS_NORMALIDAD,
    fallos_maximos: int = FALLOS_MAXIMOS_NORMALIDAD,
    log_msg: Optional[Callable[[str], None]] = None
) -> Tuple[Dict[str, Any], bool]:
    """
    Evalúa la normalidad de los residuos de un modelo ARMA mediante dos pruebas estadísticas:

    1. **Shapiro-Wilk**: adecuada para muestras pequeñas o moderadas (< 5000 elementos).
    2. **D’Agostino-Pearson (normaltest)**: recomendada para tamaños grandes.

    Ambas pruebas retornan un p-valor. Si dicho p-valor es menor que `alpha`,
    se rechaza la hipótesis de normalidad.

    El vector de residuos se considera "normal" si la cantidad de pruebas con p-valor
    < `alpha` no excede el límite `fallos_maximos`.

    Parámetros:
        resid (np.ndarray): Residuos del modelo ARMA.
        alpha (float): Nivel de significancia para rechazar normalidad.
        fallos_maximos (int): Máximo de pruebas que pueden fallar para aceptar normalidad.
        log_msg (Callable, opcional): Función de log para registrar advertencias o errores.

    Retorna:
        Tuple[Dict[str, Any], bool]:
            - Dict: Resultados de cada prueba (p-valores y errores si ocurren).
                Claves posibles: 'shapiro', 'dagostino', 'shapiro_error', 'dagostino_error'.
            - bool: True si los residuos pueden considerarse normales, False en caso contrario.
    """
    resultado = {}

    try:
        # Validación de entrada
        if len(resid) == 0:
            log_msg("El vector de residuos está vacío.")
            raise ValueError("El vector de residuos está vacío.")

        # Prueba de Shapiro-Wilk (mejor en muestras pequeñas-medias)
        try:
            if len(resid) >= LIMITE_SHAPIRO:
                p_shapiro = shapiro(resid[:LIMITE_SHAPIRO])[1]  # limitar tamaño
            else:
                p_shapiro = shapiro(resid)[1]
            resultado["shapiro"] = float(p_shapiro)
        except Exception as e:
            resultado["shapiro"] = None
            resultado["shapiro_error"] = str(e)
            p_shapiro = 1.0  # se considera como no fallida

        # Prueba de D’Agostino-Pearson (más robusta en grandes muestras)
        try:
            p_dagostino = normaltest(resid)[1]
            resultado["dagostino"] = float(p_dagostino)
        except Exception as e:
            resultado["dagostino"] = None
            resultado["dagostino_error"] = str(e)
            p_dagostino = 1.0  # no penaliza

        # 🔍 Evaluación de número de fallos según p-valores
        fallos = sum(
            p < alpha for p in [p for p in [p_shapiro, p_dagostino] if p is not None]
        )
        es_normal = fallos <= fallos_maximos

        return resultado, es_normal

    except Exception as e:
        return {"error": str(e)}, False

def realizar_prueba_autocorrelacion(
    resid: np.ndarray,
    alpha: float = ALPHA_TESTS_AUTOCORR,
    log_msg: Optional[Callable[[str], None]] = None
) -> Tuple[Dict[str, Any], bool]:
    """
    Evalúa la autocorrelación en los residuos de un modelo ARMA mediante la prueba de Ljung-Box.

    Esta prueba verifica si los residuos son independientes (no autocorrelados) hasta un cierto número de lags.
    El número de lags se ajusta dinámicamente en función del tamaño de la muestra, respetando los límites
    definidos por configuración (`MIN_LAGS_LB`, `MAX_LAGS_LB`).

    Si el p-valor del último lag evaluado es mayor o igual a `alpha`, se considera que no hay evidencia significativa
    de autocorrelación → se aceptan los residuos como independientes.

    Parámetros:
        resid (np.ndarray): Vector de residuos del modelo ARMA.
        alpha (float): Nivel de significancia para la prueba. Si p < alpha, se rechaza la independencia.
        log_msg (Callable, opcional): Función para registrar mensajes en log o consola.

    Retorna:
        Tuple[Dict[str, Any], bool]:
            - dict: {'ljungbox_p': valor_p_final} o {"error": mensaje}.
            - bool: True si no hay autocorrelación (p >= alpha); False si se rechaza la independencia.
    """
    try:
        n = len(resid)

        # 🔒 Validación: longitud mínima para aplicar la prueba
        if n < 10:
            log_msg("La serie de residuos es demasiado corta para Ljung-Box.")
            raise ValueError("La serie de residuos es demasiado corta para Ljung-Box.")

        # 🧮 Determinar número de lags (entre MIN y MAX definidos por configuración)
        max_lags = min(MAX_LAGS_LB, max(MIN_LAGS_LB, n // 10))

        # 🧪 Ejecutar Ljung-Box para detectar autocorrelación
        resultado = acorr_ljungbox(
            resid,
            lags=range(1, max_lags + 1),
            return_df=True
        )

        # 🔍 Evaluar p-valor del último lag (más estricto)
        p_ljung = resultado["lb_pvalue"].iloc[-1]
        pasa_test = p_ljung >= alpha  # True → no hay autocorrelación significativa

        return {"ljungbox_p": float(p_ljung)}, pasa_test

    except Exception as e:
        return {"error": str(e)}, False

def realizar_prueba_heterocedasticidad(
    resid: np.ndarray,
    alpha: float = ALPHA_TESTS_HETEROCED,
    log_msg: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, Any], bool]:
    """
    Evalúa la heterocedasticidad en los residuos de un modelo mediante la prueba ARCH de Engle.

    La heterocedasticidad implica que la varianza de los errores cambia en el tiempo, lo cual
    viola los supuestos de muchos modelos estadísticos. Esta función verifica si los residuos
    presentan comportamiento ARCH (varianza condicional no constante).

    Si el p-valor obtenido es mayor o igual a `alpha`, se acepta homocedasticidad
    (varianza constante), lo que es deseable.

    Parámetros:
        resid (np.ndarray): Vector de residuos del modelo.
        alpha (float): Nivel de significancia para la prueba. Si p < alpha, se rechaza homocedasticidad.
        log_msg (Callable, opcional): Función para mostrar mensajes o registrar logs.

    Retorna:
        Tuple[Dict[str, Any], bool]:
            - dict: {'arch_pval': valor_p_arch} o {"error": str} si falla.
            - bool: True si los residuos se consideran homocedásticos; False si hay evidencia de heterocedasticidad.
    """

    try:
        # Validación de longitud mínima
        if len(resid) < 10:
            raise ValueError("La serie de residuos es demasiado corta para evaluar heterocedasticidad.")

        # Ejecutar prueba ARCH de Engle (detecta varianza no constante)
        _, pval_arch, _, _ = het_arch(resid)

        resultado = {"arch_pval": float(pval_arch)}

        # Decisión: si p >= alpha, no hay evidencia de heterocedasticidad
        pasa_test = pval_arch >= alpha

        return resultado, pasa_test

    except Exception as e:
        # En caso de error, registrar y retornar resultado negativo
        return {"error": str(e)}, False

def seleccionar_modelo_optimo(
    resultados: List[Tuple[Dict[str, Any], Optional[float], Optional[float], Tuple[int, int], Dict[str, Any], bool, str]],
    criterio: str = CRITERIO_SELECCION_MODELO,
    usar_normalidad: bool = USAR_TESTS_NORMALIDAD,
    usar_autocorrelacion: bool = USAR_TESTS_AUTOCORRELACION,
    usar_heterocedasticidad: bool = USAR_TESTS_HETEROCEDASTICIDAD,
    log_msg: Optional[Callable[[str], None]] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Selecciona el modelo ARMA óptimo a partir de una lista de resultados evaluados estadísticamente.

    La selección sigue una jerarquía de preferencia:
    1. Modelos que validan todas las pruebas activadas (normalidad, autocorrelación, heterocedasticidad).
    2. Modelos que validan al menos autocorrelación y heterocedasticidad (si están activadas).
    3. Modelo con el mejor valor del criterio (AIC o BIC), sin importar validaciones.

    A partir de la versión actual, los modelos se comparan según un **score total ponderado**
    que combina la métrica seleccionada (AIC o BIC) normalizada y penalizaciones proporcionales
    a los p-valores de los tests activados:

        score_total = 
            w_criterio × (AIC o BIC normalizado)
          + w_norm     × penalización de normalidad (si aplica)
          + w_auto     × penalización de autocorrelación
          + w_heter    × penalización de heterocedasticidad

    La normalización se realiza min-max entre los modelos válidos.
    Los pesos (`w_*`) pueden ajustarse dentro del bloque correspondiente.

    Parámetros:
        resultados (List[Tuple]): Lista de tuplas con resultados por modelo:
            - modelo_dict (Dict): Información del modelo.
            - aic (float | None)
            - bic (float | None)
            - orden (Tuple[int, int])
            - tests (Dict): Resultados de validaciones estadísticas.
            - validado (bool): Indicador si pasa validaciones.
            - origen (str): Nombre/identificador del modelo.

        criterio (str): Métrica principal de selección ('aic' o 'bic').
        usar_normalidad (bool): Si evaluar test de normalidad.
        usar_autocorrelacion (bool): Si evaluar test de autocorrelación (Ljung-Box).
        usar_heterocedasticidad (bool): Si evaluar test de heterocedasticidad (ARCH).
        log_msg (Callable): Función opcional para registrar mensajes (logger).

    Retorna:
        Tuple:
            - modelo_dict (Dict): Modelo seleccionado (o None si no hay candidato).
            - log_texto (str): Resumen textual explicativo del modelo y su validación.
    """

    def resumen(modelo, validacion: str) -> str:
        """
        Genera un resumen textual explicativo con los resultados de validación estadística
        del modelo seleccionado (normalidad, autocorrelación, heterocedasticidad).

        Evalúa si el modelo cumple con los supuestos deseables en los residuos y documenta
        los resultados test por test, indicando si se pasan, se omiten o están desactivados.

        Parámetros:
            modelo (tuple): Tupla con información estructurada del modelo:
                - modelo_dict: dict con info clave ('params', 'residuals', etc.).
                - aic (float): Métrica AIC del modelo.
                - bic (float): Métrica BIC del modelo.
                - orden (tuple): Par (p, q).
                - tests (dict): Resultados de los tests estadísticos.
                - validado (bool): Estado de validación.
                - origen (str): Método de selección o fuente del modelo.

            validacion (str): Estado de validación ('Sí', 'Parcialmente', 'No').

        Retorna:
            str: Cadena de texto con explicación detallada y resultados de cada prueba.
        """
        # Introducción a los criterios evaluados
        txt = "\n=== CRITERIOS DE VALIDACIÓN ===\n"
        txt += "- Normalidad: residuos deben seguir una distribución normal (Shapiro, D'Agostino)\n"
        txt += "- Autocorrelación: residuos no deben tener autocorrelación (Ljung-Box)\n"
        txt += "- Heterocedasticidad: varianza constante en el tiempo (ARCH)\n"

        # Detalles del modelo y métricas
        txt += "\n--- SELECCIÓN DEL MODELO ---\n"
        txt += f"Origen: {modelo[6]}\n"
        txt += f"Orden (p, q): {modelo[3]}\n"
        txt += f"AIC: {modelo[1]:.4f}\n"
        txt += f"BIC: {modelo[2]:.4f}\n"
        txt += f"Validado: {validacion}\n"
        txt += "Resultados de tests:\n"

        # Establecer qué tests se deben tener en cuenta según el estado de validación
        if validacion == "Sí":
            tests_usados = {
                "normalidad": USAR_TESTS_NORMALIDAD,
                "autocorrelacion": USAR_TESTS_AUTOCORRELACION,
                "heterocedasticidad": USAR_TESTS_HETEROCEDASTICIDAD
            }
        elif validacion == "Parcialmente":
            # En fase relajada se ignora la normalidad
            tests_usados = {
                "normalidad": False,
                "autocorrelacion": USAR_TESTS_AUTOCORRELACION,
                "heterocedasticidad": USAR_TESTS_HETEROCEDASTICIDAD
            }
        else:
            # En modelos no validados, se reportan todos según configuración global
            tests_usados = {
                "normalidad": USAR_TESTS_NORMALIDAD,
                "autocorrelacion": USAR_TESTS_AUTOCORRELACION,
                "heterocedasticidad": USAR_TESTS_HETEROCEDASTICIDAD
            }

        # Evaluar y agregar resultados de cada test
        for test, resultados_test in modelo[4].items():
            if test == "error":
                continue  # Ignorar campo de error global

            resumen_test = dict(resultados_test) if isinstance(resultados_test, dict) else {}

            # ─ Normalidad ─
            if test == "normalidad":
                if USAR_TESTS_NORMALIDAD:
                    if tests_usados[test]:
                        # Solo se aprueba si todos los p-valores son mayores al umbral
                        ok = all(
                            isinstance(p, (float, int)) and p > ALPHA_TESTS_NORMALIDAD
                            for p in resumen_test.values()
                            if not isinstance(p, str)
                        )
                        resumen_test["✔️"] = ok
                    else:
                        resumen_test["info"] = "omitido (fase relajada)"
                else:
                    resumen_test["info"] = "test desactivado"

            # ─ Autocorrelación ─
            elif test == "autocorrelacion":
                if USAR_TESTS_AUTOCORRELACION:
                    if tests_usados[test]:
                        resumen_test["✔️"] = resumen_test.get("ljungbox_p", 0) > ALPHA_TESTS_AUTOCORR
                    else:
                        resumen_test["info"] = "omitido (fase relajada)"
                else:
                    resumen_test["info"] = "test desactivado"

            # ─ Heterocedasticidad ─
            elif test == "heterocedasticidad":
                if USAR_TESTS_HETEROCEDASTICIDAD:
                    if tests_usados[test]:
                        resumen_test["✔️"] = resumen_test.get("arch_pval", 0) > ALPHA_TESTS_HETEROCED
                    else:
                        resumen_test["info"] = "omitido (fase relajada)"
                else:
                    resumen_test["info"] = "test desactivado"

            # Añadir al resumen textual
            txt += f"  {test}: {resumen_test}\n"

        return txt

    # Determinar el índice de la métrica en la tupla
    key_map = {"aic": 1, "bic": 2}
    criterio_idx = key_map.get(criterio.lower(), 1)

    # Modelos que validan todas las pruebas activadas
    modelos_validos = []
    for r in resultados:
        tests = r[4]
        validado = True

        if usar_normalidad:
            normalidad = tests.get("normalidad", {})
            validado &= all(
                isinstance(p, (float, int)) and p > ALPHA_TESTS_NORMALIDAD
                for p in normalidad.values()
            )

        if usar_autocorrelacion:
            validado &= tests.get("autocorrelacion", {}).get("ljungbox_p", 0) > ALPHA_TESTS_AUTOCORR

        if usar_heterocedasticidad:
            validado &= tests.get("heterocedasticidad", {}).get("arch_pval", 0) > ALPHA_TESTS_HETEROCED

        if validado:
            modelos_validos.append(r)


    # Filtrar modelos con criterio definido
    modelos_validos = [r for r in modelos_validos if r[criterio_idx] is not None]

    if modelos_validos:
        # Definición de pesos para ponderar AIC/BIC y los tests estadísticos
        w_criterio   = 0.4
        w_norm  = 0.2
        w_auto  = 0.2
        w_heter = 0.2
        # Normalización min-max del AIC/BIC
        valores_criterio = [r[criterio_idx] for r in modelos_validos]
        min_val, max_val = min(valores_criterio), max(valores_criterio)
        rango_val = max_val - min_val if max_val != min_val else 1.0

        def score_total(x):
            """
        Calcula un score total ponderado para un modelo dado.

        Combina la métrica principal (AIC/BIC) normalizada y las penalizaciones que resultan de los resultados de las pruebas estadísticas
        y de acuerdo a la ponderación. 

        Cuanto menor es el score, mejor es el modelo.

        Parámetros:
            x (tuple): Tupla con la información del modelo evaluado.

        Retorna:
            float: Score total del modelo (valor a minimizar).
            """
            norm_crit = (x[criterio_idx] - min_val) / rango_val

            penal_norm = sum(
                1.0 - p if isinstance(p, (int, float)) else 1.0
                for p in x[4].get("normalidad", {}).values()
                if not isinstance(p, str)
            )

            penal_auto = (
                1.0 - x[4].get("autocorrelacion", {}).get("ljungbox_p", 0)
                if isinstance(x[4].get("autocorrelacion", {}).get("ljungbox_p", 0), (int, float))
                else 1.0
            )

            penal_heter = (
                1.0 - x[4].get("heterocedasticidad", {}).get("arch_pval", 0)
                if isinstance(x[4].get("heterocedasticidad", {}).get("arch_pval", 0), (int, float))
                else 1.0
            )

            return (
                w_criterio   * norm_crit +
                w_norm  * penal_norm +
                w_auto  * penal_auto +
                w_heter * penal_heter
            )

        modelo_final = min(modelos_validos, key=score_total)
        score_val = score_total(modelo_final)

        log_msg(
            f"✅ Modelo validado completo: {modelo_final[6]} orden={modelo_final[3]} "
            f"{criterio.upper()}={modelo_final[criterio_idx]:.2f} | Score total ponderado={score_val:.4f}"
        )
        return modelo_final[0], resumen(modelo_final, "Sí")

    # Modelos parcialmente validados (sin normalidad)
    modelos_relajados = []
    for r in resultados:
        tests = r[4]
        auto_ok = True
        hetero_ok = True

        if usar_autocorrelacion:
            auto_ok = tests.get("autocorrelacion", {}).get("ljungbox_p", 0) > ALPHA_TESTS_AUTOCORR
        if usar_heterocedasticidad:
            hetero_ok = tests.get("heterocedasticidad", {}).get("arch_pval", 0) > ALPHA_TESTS_HETEROCED

        if auto_ok and hetero_ok:
            modelos_relajados.append(r)

    if modelos_relajados:
        # Ponderación de los criterios de validación para los modelos validados parcialmente
        w_criterio   = 0.4
        w_auto  = 0.3
        w_heter = 0.3

        # Normalización min-max del AIC/BIC
        valores_criterio = [r[criterio_idx] for r in modelos_relajados]
        min_val, max_val = min(valores_criterio), max(valores_criterio)
        rango_val = max_val - min_val if max_val != min_val else 1.0

        def score_relajado(x):
            """
        Calcula un score total ponderado para un modelo dado, sin tener en cuenta la parte de normalidad en los residuos.

        Cuanto menor es el score, mejor es el modelo.

        Parámetros:
            x (tuple): Tupla con la información del modelo evaluado.

        Retorna:
            float: Score total del modelo (valor a minimizar).
            """
            norm_crit = (x[criterio_idx] - min_val) / rango_val

            penal_auto = (
                1.0 - x[4].get("autocorrelacion", {}).get("ljungbox_p", 0)
                if isinstance(x[4].get("autocorrelacion", {}).get("ljungbox_p", 0), (int, float))
                else 1.0
            )

            penal_heter = (
                1.0 - x[4].get("heterocedasticidad", {}).get("arch_pval", 0)
                if isinstance(x[4].get("heterocedasticidad", {}).get("arch_pval", 0), (int, float))
                else 1.0
            )

            return (
                w_criterio   * norm_crit +
                w_auto  * penal_auto +
                w_heter * penal_heter
            )

        modelo_final = min(modelos_relajados, key=score_relajado)
        score_total= score_relajado(modelo_final)
        log_msg(
            f"⚠️ Modelo parcialmente validado: {modelo_final[6]} orden={modelo_final[3]} "
            f"{criterio.upper()}={modelo_final[criterio_idx]:.2f} | Score total ponderado={score_total:.4f}"
        )
        return modelo_final[0], resumen(modelo_final, "Parcialmente")

    # Selección por AIC/BIC, sin validación
    modelos_con_criterio = [r for r in resultados if r[criterio_idx] is not None]
    if modelos_con_criterio:
        modelo_final = min(modelos_con_criterio, key=lambda x: x[criterio_idx])
        log_msg(f"⚠️ Modelo por puro criterio: {modelo_final[6]} orden={modelo_final[3]} {criterio.upper()}={modelo_final[criterio_idx]:.2f}")
        return modelo_final[0], resumen(modelo_final, "No")

    # Nada válido
    log_msg("❌ No se encontró ningún modelo con datos válidos.")
    return None, "❌ No se encontró ningún modelo válido para seleccionar."

def seleccionar_mejor_modelo_desde_df(
    carpeta_modelos: pd.DataFrame,
    criterio: str = CRITERIO_SELECCION_MODELO,
    usar_normalidad: bool = USAR_TESTS_NORMALIDAD,
    usar_autocorrelacion: bool = USAR_TESTS_AUTOCORRELACION,
    usar_heterocedasticidad: bool = USAR_TESTS_HETEROCEDASTICIDAD,
    fallos_maximos: int = FALLOS_MAXIMOS_NORMALIDAD,
    alpha: float = ALPHA_TESTS_NORMALIDAD,
    alpha_auto: float = ALPHA_TESTS_AUTOCORR,
    alpha_heter: float = ALPHA_TESTS_HETEROCED,
    log_msg: Optional[Callable[[str], None]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Evalúa un conjunto de modelos ARMA desde un DataFrame y selecciona el mejor modelo disponible.

    Se aplican validaciones estadísticas sobre los residuos (normalidad, autocorrelación, heterocedasticidad)
    según configuración, y se selecciona el modelo que cumpla las condiciones más estrictas posibles.

    Además, para modelos que pasan las validaciones, se calcula un score total ponderado que combina 
    el criterio de información (AIC/BIC) con penalizaciones por resultados en los tests estadísticos.

    Prioridades:
        1. Modelos que pasan todas las validaciones activadas → se selecciona el de menor score ponderado.
        2. Modelos que pasan autocorrelación y heterocedasticidad (fase relajada).
        3. Modelo con el mejor AIC/BIC, aunque no pase validaciones.

    Parámetros:
        carpeta_modelos (pd.DataFrame): Modelos entrenados con columnas como 'p', 'q', 'residuals', etc.
        criterio (str): Criterio de selección: 'aic' o 'bic'.
        usar_normalidad (bool): Activar prueba de normalidad.
        usar_autocorrelacion (bool): Activar prueba de autocorrelación.
        usar_heterocedasticidad (bool): Activar prueba de heterocedasticidad.
        fallos_maximos (int): Fallos permitidos para considerar normalidad válida.
        alpha (float): Nivel de significancia para normalidad.
        alpha_auto (float): Nivel de significancia para autocorrelación.
        alpha_heter (float): Nivel de significancia para heterocedasticidad.
        log_msg (Callable, opcional): Función logger para imprimir mensajes.

    Retorna:
            - Modelo seleccionado (dict) o None.
    """
    resultados = []
    log_msg("🔍 Iniciando selección del mejor modelo desde DataFrame...")

    modelos = seleccionar_modelos_convergidos_desde_df(carpeta_modelos,log_msg=log_msg)
    log_msg(f"📦 Modelos convergidos encontrados: {len(modelos)}")

    for nombre_serie, (p, q), model_data in modelos:
        pruebas = {
            "normalidad": {},
            "autocorrelacion": {},
            "heterocedasticidad": {},
            "error": None
        }
        valido = True

        try:
            resid = np.array(model_data["residuals"])
            aic = model_data["aic"]
            bic = model_data["bic"]

            log_msg(f"\n➡️ Evaluando modelo: {nombre_serie} | Orden: ({p},{q})")
            log_msg(f"   AIC: {aic} | BIC: {bic}")
            log_msg(f"   Residuos: {len(resid)} observaciones")

            # ───── Normalidad
            if usar_normalidad:
                resultado, es_valido = realizar_prueba_normalidad(resid, alpha=alpha, fallos_maximos=fallos_maximos,log_msg=log_msg)
                pruebas["normalidad"] = resultado
                if not es_valido:
                    valido = False
                log_msg(f"   🔬 Normalidad: {resultado} | {'✅' if es_valido else '❌'}")

            # ───── Autocorrelación
            if usar_autocorrelacion:
                resultado, es_valido = realizar_prueba_autocorrelacion(resid, alpha=alpha_auto,log_msg=log_msg)
                pruebas["autocorrelacion"] = resultado
                if not es_valido:
                    valido = False
                log_msg(f"   🔄 Autocorrelación: {resultado} | {'✅' if es_valido else '❌'}")

            # ───── Heterocedasticidad
            if usar_heterocedasticidad:
                resultado, es_valido = realizar_prueba_heterocedasticidad(resid, alpha=alpha_heter,log_msg=log_msg)
                pruebas["heterocedasticidad"] = resultado
                if not es_valido:
                    valido = False
                log_msg(f"   📉 Heterocedasticidad: {resultado} | {'✅' if es_valido else '❌'}")

            resultados.append((model_data, aic, bic, (p, q), pruebas, valido, nombre_serie))

        except Exception as e:
            log_msg(f"⚠️ Error procesando modelo {nombre_serie} ({p},{q}): {e}")
            pruebas["error"] = str(e)
            resultados.append((model_data, None, None, (p, q), pruebas, False, nombre_serie))

    log_msg(f"\n📊 Total modelos evaluados: {len(resultados)}")

    # ───── Selección del modelo óptimo
    modelo_sel, log_resumen = seleccionar_modelo_optimo(
        resultados=resultados,
        criterio=criterio,
        usar_normalidad=usar_normalidad,
        usar_autocorrelacion=usar_autocorrelacion,
        usar_heterocedasticidad=usar_heterocedasticidad,
        log_msg=log_msg
    )

    log_msg(log_resumen)

    return modelo_sel

