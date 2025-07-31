
"""
MÓDULO: Función_arma.py

Este módulo implementa una estrategia eficiente para entrenar múltiples modelos ARMA(p, q)
utilizando dos ejecuciones, una warm-start con parámetros iniciales de modelos previos ya entrenados y 
otra refinada sin parámetros.

────────────────────────────────────────────────────────────────────────────
📌 FUNCIONALIDADES PRINCIPALES:

1. Búsqueda automatizada de modelos ARMA:
   - `run_arma_grid_search()`: Orquesta la búsqueda completa en dos fases (inicial + refinada),
     entrenando en paralelo y refinando modelos no convergidos y más prometedores.
     Una vez terminada la ejecución, borra todos los modelos .pkl.

2. Inicialización eficiente:
   - `get_start_params()`: Adapta los parámetros de vecinos entrenados como start_params.
   - `buscar_vecinos_eficientes()`: Identifica modelos vecinos relevantes ordenados por AIC/BIC para utilizar los parámetros.

3. Persistencia estructurada:
   - `save_params_to_disk()`: Guarda parámetros, métricas y residuos de forma reutilizable en disco, que luego se borrará.

4. Ajuste robusto de modelos:
   - `fit_model()`: Ajusta un modelo ARIMA(p,q). 

5. Modularidad y trazabilidad:
   - Soporte para logs de tiempos, métricas de entrenamiento, y almacenamiento de resultados en CSV.

────────────────────────────────────────────────────────────────────────────
RETORNO FINAL:
   - dataframe con la información de todos los modelos entrenados. 

"""
# =============================================================
# 🧱 1. LIBRERÍAS ESTÁNDAR
# =============================================================
import os
import time
import pickle
import warnings
from functools import partial
from typing import Any, Optional, Union, List, Tuple, Callable, Dict
import shutil

# =============================================================
# 📦 2. LIBRERÍAS DE TERCEROS
# =============================================================
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from statsmodels.tsa.arima.model import ARIMA

# =============================================================
# ⚙️ 3. CONFIGURACIÓN GLOBAL (config.py)
# =============================================================

from config import (
    # ▸ Rango de órdenes y complejidad
    ORDEN_MIN,
    ORDEN_MAX,
    UMBRAL_STARTPARAMS,
    MAX_DIST_VECINOS,
    N_GRUPOS,

    # ▸ Entrenamiento de modelos
    MAX_ITER,
    MAX_ITER_FINAL,
    TREND,
    FORZAR_ESTACIONARIA,
    FORZAR_INVERTIBILIDAD,
    COV_TYPE,
    USO_PARAMETROS_INICIALES_ARMA,

    # ▸ Selección y evaluación
    METRIC,
    CRITERIO_AIC_BIC_VECINOS,
    TOP_N,

    # ▸ Persistencia y logging
    PARAMSDIR,
    LOG_CSV,
    TIMING_LOG,
    DATASET_NAME,

    # ▸ Paralelismo y control
    NJOBS,
)

def save_params_to_disk(
    p: int,
    q: int,
    params: Any,
    params_dir: str = PARAMSDIR,
    aic: Optional[float] = None,
    bic: Optional[float] = None,
    converged: Optional[bool] = None,
    opt_message: Optional[str] = None,
    residuals: Optional[Union[list, Any]] = None,
    phase: Optional[str] = None,
    log_msg: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Guarda en disco los parámetros y métricas asociadas a un modelo ARMA en formato .pkl, los cuales se borrarán posteriormente.

    El archivo se nombra como "{p}_{q}_{phase}.pkl" y se guarda en el directorio especificado.
    Incluye información del modelo como parámetros, criterios AIC/BIC, convergencia y residuos.

    Parámetros:
        p (int): Orden autorregresivo (AR) del modelo.
        q (int): Orden de media móvil (MA) del modelo.
        params (Any): Parámetros estimados del modelo.
        params_dir (str): Carpeta donde se guardará el archivo.
        aic (float, opcional): Valor del criterio de información AIC.
        bic (float, opcional): Valor del criterio de información BIC.
        converged (bool, opcional): Si el modelo convergió.
        opt_message (str, opcional): Mensaje del optimizador.
        residuals (list o Any, opcional): Residuos del modelo.
        phase (str, opcional): Fase o contexto del modelo (e.g., "train", "val").
        log_msg (Callable, opcional): Función para registrar mensajes (logging).

    Retorna:
        None
    """
    # Asegura que el directorio destino exista
    os.makedirs(params_dir, exist_ok=True)

    # Nombre del archivo de salida (usa 'unknown' si no se define una fase)
    phase_str = phase if phase is not None else "unknown"
    path = os.path.join(params_dir, f"{p}_{q}_{phase_str}.pkl")

    # Diccionario con todos los elementos a guardar
    data = {
        "p": p,
        "q": q,
        "params": params,
        "aic": aic,
        "bic": bic,
        "converged": converged,
        "opt_message": opt_message,
        "residuals": residuals,
        "phase": phase
    }

    # Guardar el diccionario como archivo pickle
    with open(path, "wb") as f:
        pickle.dump(data, f)

    # Mensaje de confirmación
    log_msg(f"✅ Guardado: {path}")

def buscar_vecinos_eficientes(
    p: int,
    q: int,
    params_dir: str = PARAMSDIR,
    max_dist: int = MAX_DIST_VECINOS,
    criterio: str = CRITERIO_AIC_BIC_VECINOS,
    log_msg: Optional[Callable[[str], None]] = None,
) -> List[Tuple[int, int]]:
    """
    Busca modelos vecinos a (p, q) ya entrenados y almacenados, que tengan igual o menor complejidad
    y que obtengan un mejor score (AIC o BIC).

    Se considera como vecinos a aquellos modelos (p_i, q_i) tales que:
        - Están dentro de un rango de ±max_dist desde (p, q).
        - Su complejidad total (p_i + q_i) es menor o igual que la del modelo actual.
        - Si la complejidad es igual, se prioriza p_i < p.

    Solo se devuelven aquellos vecinos para los que existe un archivo de parámetros entrenados (.pkl)
    y que tengan el criterio deseado disponible.

    Parámetros:
        p (int): Orden autorregresivo del modelo actual.
        q (int): Orden de medias móviles del modelo actual.
        params_dir (str): Carpeta donde están guardados los modelos entrenados.
        max_dist (int): Distancia máxima a considerar para vecinos (en ambas direcciones).
        criterio (str): Criterio para ordenar los vecinos: "aic" o "bic".
        log_msg (Callable, opcional): Función para registrar mensajes.

    Retorna:
        List[Tuple[int, int]]: Lista ordenada de vecinos eficientes [(p_i, q_i), ...].
    """
    if criterio not in ("aic", "bic"):
        raise ValueError(f"Criterio no válido: {criterio}. Usa 'aic' o 'bic'.")

    vecinos = []

    # Iterar en una vecindad de (p, q)
    for dp in range(-max_dist, max_dist + 1):
        for dq in range(-max_dist, max_dist + 1):
            pi, qi = p + dp, q + dq

            # Evitar índices negativos o el propio modelo actual
            if pi < 0 or qi < 0 or (pi == p and qi == q):
                continue

            # Verificar que la complejidad sea igual o menor
            if (pi + qi < p + q) or (pi + qi == p + q and pi < p):
                filename = f"{pi}_{qi}_initial.pkl"
                path = os.path.join(params_dir, filename)

                # Si el modelo vecino existe, cargar su score
                if os.path.exists(path):
                    try:
                        with open(path, "rb") as f:
                            data = pickle.load(f)
                            valor = data.get(criterio)
                            if valor is not None:
                                vecinos.append(((pi, qi), valor))
                    except Exception:
                        # Si falla la lectura, se ignora ese modelo
                        continue

    # Ordenar vecinos por el valor del criterio (menor es mejor)
    vecinos.sort(key=lambda x: x[1])

    # Devolver solo los (p_i, q_i)
    return [x[0] for x in vecinos]

def get_start_params(
    p: int,
    q: int,
    buscar_vecinos_fn: Callable[[int, int], List[Tuple[int, int]]],
    params_dir: str = PARAMSDIR,
    umbral_start_params: int = UMBRAL_STARTPARAMS,
    log_msg: Optional[Callable[[str], None]] = None,
) -> Optional[List[float]]:
    """
    Genera un vector de parámetros iniciales para un modelo ARMA(p, q), 
    reutilizando información de modelos vecinos previamente entrenados 
    que sean de menor complejidad.

    La estructura devuelta sigue el orden estándar:
        [AR₁, ..., AR_p, MA₁, ..., MA_q, sigma²]

    Si el modelo vecino tiene menos parámetros que los requeridos, se completa con ceros.
    Si tiene más, se recorta para que encaje en la estructura del modelo actual.

    Parámetros:
        p (int): Orden AR (autoregresivo) del modelo actual.
        q (int): Orden MA (media móvil) del modelo actual.
        buscar_vecinos_fn (Callable): Función que devuelve vecinos [(p_i, q_i), ...] ordenados por eficiencia.
        params_dir (str): Carpeta donde están los .pkl con los parámetros entrenados.
        umbral_start_params (int): Límite mínimo de complejidad (p+q) para usar vecinos como base.
        log_msg (Callable, opcional): Función para registrar mensajes (logs).

    Retorna:
        Optional[List[float]]: Lista de parámetros iniciales [ARs + MAs + sigma²],
                               o None si no hay vecinos aptos o no se supera el umbral.
    """

    # Si el modelo es muy simple, no se buscan vecinos
    if (p + q) <= umbral_start_params:
        return None

    # Buscar modelos vecinos entrenados
    vecinos = buscar_vecinos_fn(p, q)

    for pi, qi in vecinos:
        neighbor_path = os.path.join(params_dir, f"{pi}_{qi}_initial.pkl")

        # Saltar si no existe el archivo del vecino
        if not os.path.exists(neighbor_path):
            continue

        try:
            with open(neighbor_path, "rb") as f:
                data = pickle.load(f)
        except Exception:
            continue  # Saltar si el archivo no puede abrirse

        params = data.get("params")

        # Validar que los parámetros sean una lista o tupla suficientemente larga
        if not isinstance(params, (list, tuple)) or len(params) < (pi + qi + 1):
            continue

        # ─── Separar y ajustar parámetros ───
        ar_params = list(params[:pi])          # AR coefficients
        ma_params = list(params[pi:pi + qi])   # MA coefficients
        sigma2 = params[-1]                    # sigma² al final

        # Ajustar longitud de AR
        if pi < p:
            ar_params += [0.0] * (p - pi)
        else:
            ar_params = ar_params[:p]

        # Ajustar longitud de MA
        if qi < q:
            ma_params += [0.0] * (q - qi)
        else:
            ma_params = ma_params[:q]

        # Devolver el vector completo: [ARs, MAs, sigma²]
        return ar_params + ma_params + [sigma2]

    # Si ningún vecino fue apto, devolver None
    return None

def run_arma_grid_search(
    y,
    order_min: int = ORDEN_MIN,
    order_max: int = ORDEN_MAX,
    max_iter: int = MAX_ITER,
    n_grupos: int = N_GRUPOS,
    njobs: int = NJOBS,
    metric: str = METRIC,
    top_n: int = TOP_N,
    max_iter_final: int = MAX_ITER_FINAL,
    log_csv: str = LOG_CSV,
    timing_log_csv: str = TIMING_LOG,
    dataset_name: str = DATASET_NAME,
    params_dir: str = PARAMSDIR,
    trend: str = TREND,
    enforce_stationarity: bool = FORZAR_ESTACIONARIA,
    enforce_invertibility: bool = FORZAR_INVERTIBILIDAD,
    cov_type: str = COV_TYPE,
    log_msg: Optional[Callable[[str], None]] = None,
    uso_parametros_iniciales: bool = USO_PARAMETROS_INICIALES_ARMA
) -> pd.DataFrame:
    """
    Realiza una búsqueda exhaustiva (grid search) de modelos ARMA(p, q), 
    ajustando primero todas las combinaciones dentro de un rango de órdenes y 
    luego refinando los mejores modelos no convergidos.

    Fases del proceso:
        1. Genera combinaciones (p, q) con p + q en [order_min, order_max].
        2. Ajusta todos los modelos en paralelo.
        3. Guarda resultados intermedios y selecciona los mejores no convergidos.
        4. Reajusta los modelos seleccionados con más iteraciones.
        5. Registra tiempos de ejecución y guarda todos los resultados.

    Parámetros:
        y: Serie temporal a modelar.
        order_min/order_max (int): Rango de órdenes p + q.
        max_iter (int): Iteraciones iniciales de ajuste.
        n_grupos (int): Número de grupos en los que se divide el rango del ratio estructural p / (p + q), para asegurar una selección equitativa de modelos AR, MA y mixtos en la fase de refinamiento.
        njobs (int): Núcleo de paralelismo.
        metric (str): Métrica a optimizar ('aic' o 'bic').
        top_n (int): Cuántos modelos no convergidos seleccionar en total.
        max_iter_final (int): Iteraciones para reentrenamiento final.
        log_csv (str): Ruta donde guardar el log de resultados.
        timing_log_csv (str): Ruta donde guardar los tiempos por fase.
        dataset_name (str): Nombre identificador del dataset.
        params_dir (str): Carpeta donde guardar los modelos .pkl.
        trend, enforce_stationarity, enforce_invertibility, cov_type: Configuración del modelo ARIMA.
        log_msg (Callable, opcional): Logger para mensajes.
        uso_parametros_iniciales (bool): Si se desea realizar la técnica de parámetros iniciales, o por el contrario que la ejecución sea sin parámetros iniciales. 

    Retorna:
        pd.DataFrame: Todos los resultados (fase inicial + refinada).
    """


    def generate_orders(order_min: int, order_max: int) -> List[Tuple[int, int]]:
        """
        Genera todas las combinaciones posibles de órdenes (p, q) para modelos ARMA
        donde la suma p + q está entre los valores indicados.

        Se recorre el triángulo inferior del plano (p, q) con p, q ≥ 0,
        produciendo tuplas (p, q) tales que:
            order_min ≤ p + q ≤ order_max

        Parámetros:
            order_min (int): Suma mínima deseada de órdenes p + q (inclusive).
            order_max (int): Suma máxima deseada de órdenes p + q (inclusive).

        Retorna:
            List[Tuple[int, int]]: Lista de combinaciones válidas (p, q) ordenadas por p.
        """
        # ─── Validaciones de entrada ───
        if order_min < 0 or order_max < 0:
            raise ValueError("Los valores de orden deben ser no negativos.")

        if order_min > order_max:
            raise ValueError("order_min debe ser menor o igual a order_max.")

        # ─── Generar combinaciones (p, q) con p + q = total ───
        return [
            (p, total - p)  # q = total - p
            for total in range(order_min, order_max + 1)
            for p in range(total + 1)
        ]
    
    def fit_model(
        p: int,
        q: int,
        y,
        maxiter: int,
        trend: str,
        enforce_stationarity: bool,
        enforce_invertibility: bool,
        cov_type: str,
        params_dir: str,
        dataset_name: str,
        max_dist: int = 2,
        phase: str = "initial",
        log_msg: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Ajusta un modelo ARMA(p, q) utilizando statsmodels.ARIMA,
        con posibilidad de usar warm-start desde modelos vecinos previamente entrenados.

        Parámetros:
            p (int): Orden autorregresivo (AR).
            q (int): Orden de medias móviles (MA).
            y (array-like): Serie temporal a modelar.
            maxiter (int): Número máximo de iteraciones del optimizador.
            trend (str): Tipo de tendencia (por ejemplo, 'n' sin tendencia, 'c' constante).
            enforce_stationarity (bool): Si se fuerza estacionariedad en el modelo.
            enforce_invertibility (bool): Si se fuerza invertibilidad.
            cov_type (str): Tipo de estimador de varianza-covarianza.
            params_dir (str): Carpeta donde guardar los parámetros entrenados.
            dataset_name (str): Nombre de la serie para identificación.
            max_dist (int): Distancia máxima (en términos de p y q) para buscar vecinos.
            phase (str): Fase del proceso (ej. "initial", "refined", etc.).
            log_msg (Callable, opcional): Función para registrar mensajes/logs.

        Retorna:
            Dict[str, Any]: Diccionario con los resultados del modelo ajustado.
        """

        start_time = time.time()  # Medición de tiempo de ejecución

        try:

            # Buscar modelos vecinos más simples ya entrenados si lo permite el usuario
            if uso_parametros_iniciales:
                buscar_vecinos_fn = partial(
                    buscar_vecinos_eficientes,
                    params_dir=params_dir,
                    max_dist=max_dist,
                    log_msg=log_msg
                )

                # Intentar obtener parámetros iniciales desde vecinos si lo permite el usuario
                start_params = get_start_params(
                    p, q,
                    buscar_vecinos_fn,
                    params_dir=params_dir,
                    log_msg=log_msg
                )

            # Ajustar el modelo ARIMA con o sin start_params
            with warnings.catch_warnings(record=True) as wlist:
                warnings.simplefilter("always")
                model = ARIMA(
                    y,
                    order=(p, 0, q),
                    trend=trend,
                    enforce_stationarity=enforce_stationarity,
                    enforce_invertibility=enforce_invertibility
                )

                fit_kwargs = {
                    "method_kwargs": {"maxiter": maxiter},
                    "cov_type": cov_type
                }

                # Si hay parámetros de arranque en la fase inicial, usarlos
                
                if uso_parametros_iniciales and start_params is not None and phase == "initial":
                    fit_kwargs["method_kwargs"]["start_params"] = start_params

                result = model.fit(**fit_kwargs)

                # 4. Extraer métricas y detalles del ajuste
                aic = result.aic
                bic = result.bic
                converged = result.mle_retvals.get("converged", None)
                opt_message = result.mle_retvals.get("message", "")
                params = result.params.tolist()
                residuals = result.resid.tolist()
                warning_msgs = [str(w.message) for w in wlist]

            # Guardar los parámetros en disco para uso futuro
            
            
            if uso_parametros_iniciales:
                save_params_to_disk(
                    p=p,
                    q=q,
                    params=params,
                    params_dir=params_dir,
                    aic=aic,
                    bic=bic,
                    converged=converged,
                    opt_message=opt_message,
                    residuals=residuals,
                    phase=phase,
                    log_msg=log_msg
                )

            # Retornar resumen de resultados
            return {
                "dataset_name": dataset_name,
                "p": p,
                "q": q,
                "aic": aic,
                "bic": bic,
                "params": params,
                "residuals": residuals,
                "converged": converged,
                "warnings": "; ".join(warning_msgs) if warning_msgs else "",
                "opt_message": opt_message,
                "time_sec": time.time() - start_time,
                "phase": phase
            }

        except Exception as e:
            # En caso de error, retornar estructura con valores NaN o nulos
            return {
                "dataset_name": dataset_name,
                "p": p,
                "q": q,
                "aic": np.nan,
                "bic": np.nan,
                "params": None,
                "residuals": None,
                "converged": False,
                "warnings": str(e),
                "opt_message": str(e),
                "time_sec": time.time() - start_time,
                "phase": phase
            }

    log_msg(f"🔍 Fase inicial: Generando modelos con órdenes entre {order_min} y {order_max}...")

    #  Generar todas las combinaciones (p, q) 
    orders = generate_orders(order_min, order_max)

    #  Fase inicial: ajuste paralelo de todos los modelos 
    start_initial = time.time()

    results = Parallel(n_jobs=njobs)(
        delayed(fit_model)(
            p, q,
            y,
            max_iter,
            trend,
            enforce_stationarity,
            enforce_invertibility,
            cov_type,
            params_dir,
            dataset_name,
            max_dist=2,
            phase="initial",
            log_msg=log_msg
        )
        for p, q in tqdm(orders, desc="Fase inicial")
    )

    end_initial = time.time()

    #  Guardar resultados iniciales
    df_results = pd.DataFrame(results)
    df_results.to_csv(log_csv, index=False)

    # 1. Filtrar modelos válidos y no convergidos
    df_valid = df_results.dropna(subset=[metric]).copy()
    df_valid = df_valid[df_valid["converged"] != True].copy()
    df_valid["orden_total"] = df_valid["p"] + df_valid["q"]
    df_valid = df_valid[df_valid["orden_total"] > 0]  # Evitar división por 0
    df_valid["p_ratio"] = df_valid["p"] / df_valid["orden_total"]

    # 2. Dividir p_ratio en n_grupos iguales (de 0 a 1)
    df_valid["p_bin"] = pd.cut(
        df_valid["p_ratio"],
        bins=np.linspace(0, 1, n_grupos + 1),
        labels=False,
        include_lowest=True
    )

    # 3. Seleccionar top_k_per_bin modelos por cada bin de p_ratio
    top_k_per_bin = max(1, top_n // n_grupos)
    selected_bins = []

    for bin_id in range(n_grupos):
        grupo = df_valid[df_valid["p_bin"] == bin_id]
        top_bin = grupo.nsmallest(top_k_per_bin, metric)
        selected_bins.append(top_bin)

    top_models = pd.concat(selected_bins, ignore_index=True).drop_duplicates(subset=["p", "q"])

    # 4. Si faltan modelos (por bins vacíos), completar desde el resto
    if len(top_models) < top_n:
        ya_incluidos = top_models[["p", "q"]].apply(tuple, axis=1)
        restantes = df_valid[~df_valid[["p", "q"]].apply(tuple, axis=1).isin(ya_incluidos)]
        top_extra = restantes.nsmallest(top_n - len(top_models), metric)
        top_models = pd.concat([top_models, top_extra], ignore_index=True).drop_duplicates(subset=["p", "q"])

    conteo_por_bin = top_models["p_bin"].value_counts().sort_index()
    print("Modelos seleccionados por bin de p_ratio:")
    print(conteo_por_bin)

    k = len(top_models)
    log_msg(f"🔁 Reajuste de top {k} modelos no convergidos (max_iter={max_iter_final})...")

    #  Fase refinada: reentrenamiento 
    if k > 0:
        start_refined = time.time()
        refined_results = Parallel(n_jobs=njobs)(
            delayed(fit_model)(
                row.p, row.q,
                y,
                max_iter_final,
                trend,
                enforce_stationarity,
                enforce_invertibility,
                cov_type,
                params_dir,
                dataset_name,
                max_dist=2,
                phase="refined",
                log_msg=log_msg
            )
            for _, row in tqdm(top_models.iterrows(), total=k, desc="Fase refinada")
        )
        end_refined = time.time()
    else:
        refined_results = []
        start_refined = end_refined = time.time()
        log_msg("⚠️ No hay modelos para refinar.")

    # 8. Consolidar todos los resultados ─────
    df_refined = pd.DataFrame(refined_results)
    df_all = pd.concat([df_results, df_refined], ignore_index=True)
    df_all.to_csv(log_csv, index=False)

    #  Registrar tiempos de ejecución ─────
    timing_data = [
        {
            "dataset_name": dataset_name,
            "phase": "initial",
            "n_models": len(orders),
            "time_sec": round(end_initial - start_initial, 4)
        },
        {
            "dataset_name": dataset_name,
            "phase": "refined",
            "n_models": k,
            "time_sec": round(end_refined - start_refined, 4)
        },
        {
            "dataset_name": dataset_name,
            "phase": "total",
            "n_models": len(orders) + k,
            "time_sec": round((end_initial - start_initial) + (end_refined - start_refined), 4)
        }
    ]

    df_timing = pd.DataFrame(timing_data)
    df_timing.to_csv(
        timing_log_csv,
        mode='a',
        index=False,
        header=not os.path.exists(timing_log_csv)
    )

    if os.path.exists(params_dir):
        shutil.rmtree(params_dir)
        log_msg(f"✅ Carpeta con los pickle de los modelos eliminada: {params_dir}")

    return df_all