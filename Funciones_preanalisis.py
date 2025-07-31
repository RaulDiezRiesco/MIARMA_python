"""
MÓDULO: Funciones_preanálisis.py

Este módulo proporciona una herramienta interactiva o automatizada para realizar 
un análisis exploratorio inicial sobre series temporales o numéricas univariadas.

Incluye validaciones básicas, análisis de distribución, detección de zonas con 
alta variabilidad usando IQR segmentado, y visualización configurable de la serie.

Este módulo proporciona al usuario una serie de visualizaciones y datos para poder obtener 
información estadística y gráfica de la serie que desea imputar

────────────────────────────────────────────────────────────────────────────
📌 FUNCIONALIDADES:

1. Análisis completo en un solo paso:
   - `analisis_exploratorio_interactivo()`: Orquesta el análisis completo con interacción opcional.

2. Validación de series:
   - `validar_serie_para_analisis()`: Evalúa si la serie es numérica, no vacía y válida para análisis. Descarta posibles series con 
datos no válidos. 

3. Visualización de distribución:
   - `resumen_distribucion()`: Genera gráficos y estadísticas (asimetría, curtosis, outliers, etc.).

4. Variabilidad por tramos:
   - `indice_intercuartilico()`: Divide la serie en segmentos y analiza variaciones significativas de IQR.

5. Visualización básica:
   - `plot_serie_inicial()`: Muestra o guarda una línea de tiempo simple de la serie.

6. Interacción controlada:
   - `preguntar_si_no()`: Función lógica para interactuar con el usuario. 

────────────────────────────────────────────────────────────────────────────
RETORNO FINAL:
   - información del análisis realizado en log y visualizaciones.
"""

# =============================================================
# 🧱 1. LIBRERÍAS ESTÁNDAR
# =============================================================

import os

# =============================================================
# 📦 2. LIBRERÍAS DE TERCEROS
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from typing import Callable,Tuple,Union,List,Optiona

# =============================================================
# ⚙️ 3. CONFIGURACIÓN GLOBAL (config.py)
# =============================================================

from config import (
    
    # RUTAS Y CONFIGURACIÓN GENERAL

    DEFAULT_RES_DIR,

    # MENSAJES DE VALIDACIÓN Y RESPUESTAS

    MSG_RESPUESTA_INVALIDA_SN,
    RESPUESTAS_POSITIVAS,
    RESPUESTAS_NEGATIVAS,

    # CONFIGURACIÓN ANÁLISIS DISTRIBUCIÓN

    APLICAR_LOG_TRANSFORM,
    DISTRIBUCION_PLOT_TYPES,
    MAX_MUESTRA_DISTRIBUCION,
    RANDOM_STATE_MUESTRA,
    COLOR_HISTOGRAMA,
    COLOR_BOXPLOT,
    COLOR_VIOLINPLOT,
    HIST_BINS,
    ORIENTACION_GRAFICOS,
    DEFAULT_NOMBRE_DISTRIBUCION,
    UMBRAL_ASIMETRIA,
    UMBRAL_CURTOSIS,

    # CONFIGURACIÓN ÍNDICE INTERCUARTÍLICO

    IQR_SEGMENTOS_AUTO,
    MIN_DATOS_POR_SEGMENTO,
    UMBRAL_VARIACION_IQR,
    COLOR_IQR_ALERTA,
    COLOR_IQR_NORMAL,
    FIGSIZE_IQR,
    MAX_LABELS_IQR,
    IQR_ETIQUETAS_ROTACION,
    DEFAULT_NOMBRE_IQR,

    # CONFIGURACIÓN VISUALIZACIÓN DE SERIE

    PLOT_FIGSIZE,
    PLOT_LINEWIDTH,
    PLOT_ALPHA,
    PLOT_GRID_STYLE,
    PLOT_GRID_ALPHA,
    PLOT_MAX_LABELS,
    COLOR_LINEA_SERIE_INICIAL,
    TITULO_POR_DEFECTO_SERIE,
    DEFAULT_NOMBRE_SERIE_INICIAL,

    # PREGUNTAS INTERACTIVAS (EXPLORATORIO)

    PREGUNTA_ANALISIS_DISTRIBUCION,
    PREGUNTA_MOSTRAR_GRAFICOS,
    PREGUNTA_GUARDAR_GRAFICOS,
    PREGUNTA_MOSTRAR_SERIE_COMPLETA,
    PREGUNTA_MOSTRAR_EN_PANTALLA,
    PREGUNTA_GUARDAR_IMAGEN,

    # MODO AUTOMÁTICO
    MODO_AUTO
)


def validar_serie_para_analisis(
    resumen: dict,
    log_msg: Optional[Callable[[str], None]] = None,
) -> bool:
    """
    Valida si una serie es apta para el análisis automático, utilizando su resumen que se ha generado previamente.

    Condiciones de exclusión:
        - La serie contiene solo strings.
        - La serie no es numérica.
        - La serie está completamente vacía (todos los valores son NaN).

    Parámetros:
        resumen (dict): Diccionario resumen generado por `resumen_inicial_serie()`.
        log_msg (Callable, opcional): Función para registrar mensajes en el log. 
                                      Si no se proporciona, no se registra nada.

    Retorna:
        bool: 
            - True si la serie es válida para análisis.
            - False si no lo es. Los motivos se loguean si se proporciona `log_msg`.
    """
    errores = []

    if resumen.get("contiene_strings", False):
        errores.append("La serie contiene valores de tipo string.")

    if resumen.get("resumen_estadistico") == "No disponible (dato no numérico)":
        errores.append("La serie no es numérica; no se puede generar resumen estadístico.")

    if resumen.get("valores_nulos", 0) == resumen.get("longitud", 1):
        errores.append("La serie está completamente vacía (todos los valores son NaN).")

    if errores:
        for msg in errores:
            log_msg(f"❌ {msg}")
        return False

    return True

def preguntar_si_no(
    pregunta: str,
    input_func: Callable[[str], str] = input,
    log_msg: Optional[Callable[[str], None]] = None,
    modo_auto: bool = MODO_AUTO,
    respuesta_modo_auto: Optional[bool] = True
) -> bool:
    """
    Pregunta al usuario una confirmación de sí o no, y devuelve la respuesta como booleano (True o False).

    Si se ejecuta en modo automático, devuelve la respuesta predeterminada sin interacción.

    Parámetros:
        pregunta (str): Texto de la pregunta que se mostrará al usuario.
        input_func (Callable): Función utilizada para capturar la entrada del usuario (por defecto, input).
        log_msg (Callable, opcional): Función para registrar mensajes en un log. Si es None, no se loguea nada.
        modo_auto (bool): Si True, se omite la interacción y se retorna directamente respuesta_modo_auto.
        respuesta_modo_auto (bool, opcional): Valor devuelto automáticamente si modo_auto está activado.

    Retorna:
        bool: True si la respuesta es afirmativa, False si es negativa.
    """

    if modo_auto:
        valor = respuesta_modo_auto if respuesta_modo_auto is not None else True
        if log_msg:
            log_msg(f"[modo_auto] '{pregunta}' → respuesta automática: {'sí' if valor else 'no'}")
        return valor

    while True:
        respuesta = input_func(f"{pregunta} [s/n]: ").strip().lower()

        if respuesta in RESPUESTAS_POSITIVAS:
            return True
        elif respuesta in RESPUESTAS_NEGATIVAS:
            return False
        else:
            log_msg(MSG_RESPUESTA_INVALIDA_SN)

def resumen_distribucion(
    serie: pd.Series,
    plot: bool = False,
    guardar: bool = False,
    nombre_archivo: str = DEFAULT_NOMBRE_DISTRIBUCION,
    res_dir: str = DEFAULT_RES_DIR,
    aplicar_logaritmo: bool = APLICAR_LOG_TRANSFORM,
    tipos_plot: List[str] = DISTRIBUCION_PLOT_TYPES,
    max_muestra: int = MAX_MUESTRA_DISTRIBUCION,
    random_state: int = RANDOM_STATE_MUESTRA,
    bins: int = HIST_BINS,
    color_hist: str = COLOR_HISTOGRAMA,
    color_box: str = COLOR_BOXPLOT,
    color_violin: str = COLOR_VIOLINPLOT,
    orientacion: str = ORIENTACION_GRAFICOS,
    umbral_asimetria: float = UMBRAL_ASIMETRIA,
    umbral_curtosis: float = UMBRAL_CURTOSIS,
    log_msg: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Genera un resumen gráfico y estadístico de la distribución de una serie numérica.

    La función puede aplicar una transformación logarítmica si el usuario lo desea, 
    graficar la distribución (histograma, boxplot, violinplot), guardar la imagen,
    y calcular estadísticas como asimetría, curtosis y outliers.

    Los argumentos de la función son configurables desde el módulo "config". 

    Parámetros:
        serie (pd.Series): Serie numérica a analizar.
        plot (bool): Si True, muestra los gráficos en pantalla.
        guardar (bool): Si True, guarda los gráficos como imagen PNG.
        nombre_archivo (str): Nombre base del archivo a guardar (sin extensión).
        res_dir (str): Carpeta donde guardar los gráficos si guardar=True.
        aplicar_logaritmo (bool): Si True y todos los valores son positivos, aplica log-transformación.
        tipos_plot (List[str]): Tipos de gráfico a incluir: "hist", "box", "violin".
        max_muestra (int): Tamaño máximo de muestra para graficar (reduce carga en series grandes).
        random_state (int): Semilla para muestreo reproducible.
        bins (int): Número de bins del histograma.
        color_hist (str): Color del histograma.
        color_box (str): Color del boxplot.
        color_violin (str): Color del violinplot.
        orientacion (str): Orientación de los gráficos ("horizontal" o "vertical").
        umbral_asimetria (float): Límite a partir del cual se considera que hay asimetría.
        umbral_curtosis (float): Límite a partir del cual se considera que hay colas pesadas.
        log_msg (Callable, opcional): Función para registrar mensajes en un log externo.

    Returns:
        dict: Diccionario con estadísticas descriptivas de la distribución, 
              incluyendo asimetría, curtosis, cuantiles, outliers, y observaciones.
    """


    # Validar que la entrada sea una serie numérica
    if not isinstance(serie, pd.Series):
        log_msg("❌ Se esperaba una pd.Series como entrada.")
        raise TypeError("Se esperaba una pd.Series como entrada.")
    if not pd.api.types.is_numeric_dtype(serie):
        log_msg("❌ La serie debe contener datos numéricos.")
        raise ValueError("La serie debe contener datos numéricos.")

    # Aplicar logaritmo si corresponde
    if aplicar_logaritmo and (serie > 0).all():
        log_msg("🔁 Aplicando transformación logarítmica.")
        serie = np.log(serie)

    # Eliminar valores NaN
    serie = serie.dropna()
    total = len(serie)
    

    # Si la serie queda vacía, se devuelve un resumen nulo
    if total == 0:
        log_msg("⚠️ Serie vacía tras eliminar NaN.")
        return {
            "descripcion": "La serie está vacía. No se pueden generar estadísticas.",
            "q1": None, "q3": None, "iqr": None,
            "skewness": None, "kurtosis": None,
            "n_outliers": None, "proporcion_outliers": None,
            "es_asimetrica": None, "tiene_colas_pesadas": None
        }

    # Si la serie es muy grande, se toma una muestra para graficar
    if total > max_muestra:
        log_msg(f"📉 Muestreando la serie (máx {max_muestra})...")
        muestra = serie.sample(max_muestra, random_state=random_state)
    else:
        muestra = serie.copy()

    # Graficar si se pide
    if plot or guardar:
        log_msg("📊 Generando visualización...")
        _, axs = plt.subplots(1, 3, figsize=(16, 4))
        i = 0

        if "hist" in tipos_plot:
            sns.histplot(muestra, kde=True, ax=axs[i], bins=bins, color=color_hist)
            axs[i].set_title("Histograma + KDE")
            i += 1

        if "box" in tipos_plot:
            sns.boxplot(x=muestra, ax=axs[i], color=color_box, orient=orientacion)
            axs[i].set_title("Boxplot")
            i += 1

        if "violin" in tipos_plot:
            sns.violinplot(x=muestra, ax=axs[i], color=color_violin, orient=orientacion)
            axs[i].set_title("Violin Plot")

        plt.tight_layout()

        # Guardar gráfico si se solicita
        if guardar:
            os.makedirs(res_dir, exist_ok=True)
            ruta = os.path.join(res_dir, f"{nombre_archivo}_distribucion.png")
            plt.savefig(ruta, bbox_inches="tight")
            if log_msg: log_msg(f"💾 Gráfico guardado en: {ruta}")

        # Mostrar o cerrar
        if plot:
            plt.show()
        else:
            plt.close()

    # Calcular estadísticas básicas
    skew_val = skew(serie)
    kurt_val = kurtosis(serie)
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1

    # Detectar outliers usando IQR
    outliers = ((serie < (q1 - 1.5 * iqr)) | (serie > (q3 + 1.5 * iqr))).sum()
    prop_outliers = outliers / total

    log_msg("✅ Cálculo de estadísticas completado.")

    # Devolver resumen como diccionario
    return {
        "descripcion": "Resumen gráfico y estadístico de la distribución.",
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "iqr": round(iqr, 4),
        "skewness": {
            "valor": round(skew_val, 4),
            "descripcion": "Coeficiente de asimetría (0: simétrico)."
        },
        "kurtosis": {
            "valor": round(kurt_val, 4),
            "descripcion": "Curtosis (0: normal, >0: colas pesadas)."
        },
        "n_outliers": {
            "valor": int(outliers),
            "descripcion": "Cantidad de valores atípicos (IQR)."
        },
        "proporcion_outliers": {
            "valor": round(prop_outliers, 4),
            "descripcion": "Proporción de outliers respecto al total."
        },
        "es_asimetrica": {
            "valor": abs(skew_val) > umbral_asimetria,
            "descripcion": f"True si |asimetría| > {umbral_asimetria}"
        },
        "tiene_colas_pesadas": {
            "valor": kurt_val > umbral_curtosis,
            "descripcion": f"True si curtosis > {umbral_curtosis}"
        }
    }

def indice_intercuartilico(
    serie: pd.Series,
    segmentos: Union[str, None, List[tuple]] = 'auto',
    plot: bool = False,
    guardar: bool = False,
    nombre_archivo: str = DEFAULT_NOMBRE_IQR,
    res_dir: str = DEFAULT_RES_DIR,
    min_datos_segmento: int = MIN_DATOS_POR_SEGMENTO,
    segmentos_auto: int = IQR_SEGMENTOS_AUTO,
    umbral_variacion_iqr: float = UMBRAL_VARIACION_IQR,
    figsize: tuple = FIGSIZE_IQR,
    color_alerta: str = COLOR_IQR_ALERTA,
    color_normal: str = COLOR_IQR_NORMAL,
    max_labels: int = MAX_LABELS_IQR,
    rotacion_iqr: int = IQR_ETIQUETAS_ROTACION,
    log_msg: Optional[Callable[[str], None]] = None
) -> dict:
    """
    Calcula el IQR global y por segmentos en una serie numérica.

    Permite identificar zonas con alta variación estadística, usando segmentos 
    definidos por el usuario o generados automáticamente. Puede graficar los resultados
    y marcar en color los tramos con cambios significativos.

    Args:
        serie (pd.Series): Serie numérica a analizar.
        segmentos (str | None | List[tuple]): 'auto', None, o lista de (inicio, fin) por segmento.
        plot (bool): Si True, muestra el gráfico.
        guardar (bool): Si True, guarda el gráfico como imagen.
        nombre_archivo (str): Nombre base del archivo de salida.
        res_dir (str): Carpeta donde se guarda el gráfico si guardar=True.
        min_datos_segmento (int): Cantidad mínima de datos válidos por tramo.
        segmentos_auto (int): Número de segmentos si se usa segmentación automática.
        umbral_variacion_iqr (float): Umbral (relativo al IQR global) para marcar una alerta.
        figsize (tuple): Tamaño del gráfico.
        color_alerta (str): Color para tramos con alta variación.
        color_normal (str): Color para tramos normales.
        max_labels (int): Máximo número de etiquetas visibles en eje x.
        rotacion_iqr (int): Rotación de etiquetas del eje x.
        log_msg (Callable, opcional): Función para registrar mensajes.

    Returns:
        dict: Resumen con IQR global, por tramo, y tramos con variación significativa.
    """

    # Validar entrada
    if not isinstance(serie, pd.Series):
        if log_msg: log_msg("❌ Se esperaba una pandas Series como entrada.")
        raise TypeError("Se esperaba una pandas Series como entrada.")

    # Limpiar datos nulos
    serie = serie.dropna()
    total = len(serie)

    if total == 0:
        if log_msg: log_msg("⚠️ Serie vacía tras eliminar nulos.")
        return {
            "descripcion": "La serie está vacía. No se puede calcular el IQR.",
            "iqr_general": None,
            "variaciones_significativas": None,
            "iqr_por_segmento": []
        }

    # Calcular IQR global
    iqr_general = np.percentile(serie, 75) - np.percentile(serie, 25)
    if log_msg: log_msg(f"📈 IQR global de la serie: {iqr_general:.4f}")

    resumen = {
        "descripcion": "Cálculo del rango intercuartílico (IQR) global y por segmentos.",
        "iqr_general": round(iqr_general, 4),
        "variaciones_significativas": False,
        "iqr_por_segmento": []
    }

    # Segmentación automática (dividir en partes iguales)
    if segmentos == 'auto' and total >= 100:
        step = total // segmentos_auto
        segmentos = [
            (serie.index[i], serie.index[min(i + step, total - 1)])
            for i in range(0, total, step)
        ]
        if log_msg: log_msg(f"📊 Segmentación automática en {len(segmentos)} tramos.")

    # Preparar variables para iterar
    iqr_prev = None
    etiquetas, valores_iqr, alertas, variaciones = [], [], [], []

    # Calcular IQR por cada tramo definido
    if isinstance(segmentos, list):
        for inicio, fin in segmentos:
            tramo = serie.loc[inicio:fin].dropna()
            if len(tramo) < min_datos_segmento:
                if log_msg: log_msg(f"⚠️ Tramo {inicio} → {fin} omitido (pocos datos).")
                continue

            iqr_tramo = np.percentile(tramo, 75) - np.percentile(tramo, 25)
            variacion = abs(iqr_tramo - iqr_prev) if iqr_prev is not None else 0
            hay_alerta = iqr_prev is not None and variacion > umbral_variacion_iqr * iqr_general
            resumen["variaciones_significativas"] |= hay_alerta

            etiqueta = f"{inicio}\n→ {fin}"
            resumen["iqr_por_segmento"].append((
                etiqueta,
                round(iqr_tramo, 4),
                round(variacion, 4),
                hay_alerta
            ))

            etiquetas.append(etiqueta)
            valores_iqr.append(round(iqr_tramo, 4))
            variaciones.append(round(variacion, 4))
            alertas.append(hay_alerta)

            if log_msg:
                log_msg(f"[{etiqueta}] IQR={iqr_tramo:.4f} | Δ={variacion:.4f} | {'⚠️ ALERTA' if hay_alerta else 'OK'}")

            iqr_prev = iqr_tramo

    # Crear gráfico si se pide
    if plot or guardar:
        if log_msg: log_msg("📉 Generando gráfico de IQR por segmento...")
        _, ax = plt.subplots(figsize=figsize)
        colors = [color_alerta if a else color_normal for a in alertas]
        bars = ax.bar(etiquetas, valores_iqr, color=colors)

        # Línea de referencia (IQR global)
        ax.axhline(iqr_general, color="black", linestyle="--", label="IQR global")
        ax.set_title("IQR por segmento")
        ax.set_ylabel("IQR")

        # Etiquetas x: rotadas si son pocas, ocultas si son muchas
        if len(etiquetas) > max_labels:
            ax.set_xticks([])
        else:
            ax.set_xticklabels(etiquetas, rotation=rotacion_iqr, ha="right")

        ax.legend()

        # Mostrar valores sobre las barras
        for bar, iqr_val in zip(bars, valores_iqr):
            height = bar.get_height()
            ax.annotate(f"{iqr_val:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        # Guardar imagen si corresponde
        if guardar:
            os.makedirs(res_dir, exist_ok=True)
            ruta = os.path.join(res_dir, f"{nombre_archivo}_iqr.png")
            plt.savefig(ruta, bbox_inches="tight")
            if log_msg: log_msg(f"💾 Gráfico guardado en: {ruta}")

        # Mostrar o cerrar figura
        if plot:
            plt.show()
        else:
            plt.close()

    return resumen

def plot_serie_inicial(
    serie: pd.Series,
    titulo: str = TITULO_POR_DEFECTO_SERIE,
    guardar: bool = False,
    nombre_archivo: str = DEFAULT_NOMBRE_SERIE_INICIAL,
    mostrar: bool = False,
    res_dir: str = DEFAULT_RES_DIR,
    color: str = COLOR_LINEA_SERIE_INICIAL,
    figsize: Tuple[int, int] = PLOT_FIGSIZE,
    linewidth: float = PLOT_LINEWIDTH,
    alpha: float = PLOT_ALPHA,
    grid_style: str = PLOT_GRID_STYLE,
    grid_alpha: float = PLOT_GRID_ALPHA,
    max_labels: int = PLOT_MAX_LABELS,
    log_msg: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Genera una visualización de una serie temporal, con parámetros de visualización configurables desde el módulo config.

    Parámetros:
        serie (pd.Series): Serie numérica o temporal a graficar.
        titulo (str): Título del gráfico.
        guardar (bool): Si True, guarda el gráfico como imagen PNG.
        nombre_archivo (str): Nombre del archivo sin extensión.
        mostrar (bool): Si True, muestra el gráfico; si False, solo lo guarda o cierra.
        res_dir (str): Carpeta donde se guarda el gráfico si guardar=True.
        color (str): Color de la línea.
        figsize (Tuple[int, int]): Tamaño del gráfico (ancho, alto).
        linewidth (float): Grosor de la línea.
        alpha (float): Transparencia de la línea.
        grid_style (str): Estilo de la cuadrícula.
        grid_alpha (float): Transparencia de la cuadrícula.
        max_labels (int): Máximo de etiquetas visibles en eje x.
        log_msg (Callable, opcional): Función para registrar mensajes.
    
    Retorna:
        None
    """
    # Validar tipo de entrada
    if not isinstance(serie, pd.Series):
        log_msg("❌ Se esperaba una pd.Series como entrada.")
        raise TypeError("Se esperaba una pd.Series como entrada.")

    if serie.empty:
        log_msg("⚠️ Serie vacía. No se generará gráfico.")
        return

    # Crear figura
    plt.figure(figsize=figsize)
    plt.plot(serie, linewidth=linewidth, alpha=alpha, color=color)

    # Títulos y etiquetas
    plt.title(titulo)
    plt.xlabel("Índice")
    plt.ylabel("Valor")
    plt.grid(True, linestyle=grid_style, alpha=grid_alpha)

    # Ajuste de etiquetas del eje x
    ax = plt.gca()
    total_ticks = len(serie)
    if total_ticks > max_labels:
        step = max(1, total_ticks // max_labels)
        ax.set_xticks(serie.index[::step])  # Reduce saturación de etiquetas

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Guardar gráfico si se solicita
    if guardar:
        os.makedirs(res_dir, exist_ok=True)
        ruta = os.path.join(res_dir, f"{nombre_archivo}.png")
        plt.savefig(ruta, bbox_inches="tight")
        log_msg(f"💾 Gráfico guardado en: {ruta}")

    # Mostrar o cerrar gráfico
    if mostrar:
        plt.show()
        log_msg("📊 Gráfico mostrado en pantalla.")
    else:
        plt.close()
        log_msg("📉 Gráfico cerrado (modo automático o mostrar=False).")

def analisis_exploratorio_interactivo(
    df: pd.DataFrame,
    resumen: dict,
    input_func: Callable[[str], str] = input,
    log_msg: Optional[Callable[[str], None]] = None,
    modo_auto: bool = MODO_AUTO,

    # Config análisis distribución
    tipos_plot: List[str] = DISTRIBUCION_PLOT_TYPES,
    color_hist: str = COLOR_HISTOGRAMA,
    color_box: str = COLOR_BOXPLOT,
    color_violin: str = COLOR_VIOLINPLOT,
    bins: int = HIST_BINS,
    orientacion: str = ORIENTACION_GRAFICOS,
    max_muestra: int = MAX_MUESTRA_DISTRIBUCION,
    random_state: int = RANDOM_STATE_MUESTRA,
    nombre_distribucion: str = DEFAULT_NOMBRE_DISTRIBUCION,

    # Config IQR
    segmentos_auto: int = IQR_SEGMENTOS_AUTO,
    min_datos_segmento: int = MIN_DATOS_POR_SEGMENTO,
    umbral_variacion_iqr: float = UMBRAL_VARIACION_IQR,
    color_alerta: str = COLOR_IQR_ALERTA,
    color_normal: str = COLOR_IQR_NORMAL,
    figsize_iqr: tuple = FIGSIZE_IQR,
    max_labels_iqr: int = MAX_LABELS_IQR,
    rotacion_iqr: int = IQR_ETIQUETAS_ROTACION,
    nombre_iqr: str = DEFAULT_NOMBRE_IQR,

    # Config serie inicial
    figsize_plot: tuple = PLOT_FIGSIZE,
    color_linea: str = COLOR_LINEA_SERIE_INICIAL,
    linewidth: float = PLOT_LINEWIDTH,
    alpha: float = PLOT_ALPHA,
    grid_style: str = PLOT_GRID_STYLE,
    grid_alpha: float = PLOT_GRID_ALPHA,
    max_labels: int = PLOT_MAX_LABELS,
    titulo_serie: str = TITULO_POR_DEFECTO_SERIE,
    nombre_serie: str = DEFAULT_NOMBRE_SERIE_INICIAL,

    # Config carpeta resultados
    res_dir: str = DEFAULT_RES_DIR,
) -> str:
    """
    Ejecuta un análisis exploratorio interactivo o automático sobre una serie numérica.

    El análisis incluye las funciones mencionadas previamente:
        - Validación básica de la serie.
        - Visualización y resumen estadístico de la distribución.
        - Análisis del rango intercuartílico (IQR) por segmentos.
        - Gráfico general de la serie completa.

    El comportamiento puede ser completamente automático (sin interacción), útil para entornos de producción o test.

    Parámetros:
    -----------
    df (pd.DataFrame): DataFrame con una sola columna que representa la serie numérica.
    resumen (dict): Diccionario resumen generado por `resumen_inicial_serie()`.
    input_func (Callable): Función usada para recibir entradas del usuario (por defecto: `input`).
    log_msg (Callable, opcional): Función para registrar logs. Si no se proporciona, no se loguea nada.
    modo_auto (bool): Si es True, omite interacción y asume respuestas por defecto.

    Configuración del análisis de distribución:
    -------------------------------------------
    tipos_plot (List[str]): Gráficos a generar. Opciones: ["hist", "box", "violin"].
    color_hist (str): Color del histograma.
    color_box (str): Color del boxplot.
    color_violin (str): Color del violinplot.
    bins (int): Número de bins para el histograma.
    orientacion (str): "horizontal" o "vertical" para box y violin plot.
    max_muestra (int): Tamaño máximo de muestra para gráficos (en series grandes).
    random_state (int): Semilla aleatoria para muestreo.
    nombre_distribucion (str): Nombre base del archivo para guardar la distribución.

    Configuración del análisis IQR:
    --------------------------------
    segmentos_auto (int): Número de segmentos automáticos.
    min_datos_segmento (int): Mínimo de datos requeridos por segmento.
    umbral_variacion_iqr (float): Umbral relativo para detectar cambios significativos.
    color_alerta (str): Color para segmentos con variaciones destacadas.
    color_normal (str): Color para segmentos normales.
    figsize_iqr (tuple): Tamaño del gráfico de IQR.
    max_labels_iqr (int): Máximo de etiquetas visibles en eje x.
    rotacion_iqr (int): Rotación (en grados) de etiquetas del eje x.
    nombre_iqr (str): Nombre base del archivo para guardar el gráfico de IQR.

    Configuración de visualización de la serie:
    -------------------------------------------
    figsize_plot (tuple): Tamaño del gráfico de la serie.
    color_linea (str): Color de la línea principal.
    linewidth (float): Grosor de la línea.
    alpha (float): Transparencia de la línea.
    grid_style (str): Estilo de la cuadrícula.
    grid_alpha (float): Transparencia del grid.
    max_labels (int): Máximo de etiquetas en el eje x.
    titulo_serie (str): Título del gráfico.
    nombre_serie (str): Nombre base del archivo del gráfico de la serie.

    Configuración general:
    -----------------------
    res_dir (str): Carpeta de salida donde se guardarán los gráficos si `guardar=True`.

    Retorna:
    --------
    str: Mensaje resumen del proceso, o None si se gestiona externamente por `log_msg`.
    """

    if not validar_serie_para_analisis(resumen, log_msg=log_msg):
        log_msg("⛔ Análisis interrumpido.")
        return 

    serie = df.iloc[:, 0]  # Extraer la serie de la única columna del DataFrame

    try:
        # Preguntar si se desea hacer análisis de distribución
        if preguntar_si_no(PREGUNTA_ANALISIS_DISTRIBUCION, input_func, modo_auto=modo_auto, log_msg=log_msg):

            # Preguntar si mostrar o guardar gráficos
            mostrar = preguntar_si_no(PREGUNTA_MOSTRAR_GRAFICOS, input_func, modo_auto=modo_auto, respuesta_modo_auto=False, log_msg=log_msg)
            guardar = preguntar_si_no(PREGUNTA_GUARDAR_GRAFICOS, input_func, modo_auto=modo_auto, log_msg=log_msg)

            # Ejecutar análisis de distribución
            distribucion = resumen_distribucion(
                serie=serie,
                plot=mostrar,
                guardar=guardar,
                nombre_archivo=nombre_distribucion,
                res_dir=res_dir,
                tipos_plot=tipos_plot,
                max_muestra=max_muestra,
                random_state=random_state,
                bins=bins,
                color_hist=color_hist,
                color_box=color_box,
                color_violin=color_violin,
                orientacion=orientacion,
                log_msg=log_msg
            )

            # Mostrar resumen de distribución en el log
            distribucion_str = "\n📈 Resumen de distribución:\n" + "\n".join(f"- {k}: {v}" for k, v in distribucion.items())
            if log_msg: log_msg(distribucion_str)

            # Ejecutar análisis de IQR por segmentos
            iqr_resumen = indice_intercuartilico(
                serie=serie,
                segmentos='auto',
                plot=mostrar,
                guardar=guardar,
                nombre_archivo=nombre_iqr,
                res_dir=res_dir,
                segmentos_auto=segmentos_auto,
                min_datos_segmento=min_datos_segmento,
                umbral_variacion_iqr=umbral_variacion_iqr,
                color_alerta=color_alerta,
                color_normal=color_normal,
                figsize=figsize_iqr,
                max_labels=max_labels_iqr,
                rotacion_iqr=rotacion_iqr,
                log_msg=log_msg
            )

            # Log de resultados del IQR
            iqr_str = "\n📏 Análisis del rango intercuartílico:\n"
            iqr_str += f"- IQR general: {iqr_resumen['iqr_general']}\n"
            iqr_str += f"- Variaciones significativas entre segmentos: {'Sí' if iqr_resumen['variaciones_significativas'] else 'No'}"
            if log_msg: log_msg(iqr_str)

    except Exception as e:
        if log_msg: log_msg(f"\n⚠️ Error durante el análisis exploratorio: {e}\n")

    try:
        # Preguntar si se desea visualizar la serie completa
        if preguntar_si_no(PREGUNTA_MOSTRAR_SERIE_COMPLETA, input_func, modo_auto=modo_auto, log_msg=log_msg):

            mostrar = preguntar_si_no(PREGUNTA_MOSTRAR_EN_PANTALLA, input_func, modo_auto=modo_auto, log_msg=log_msg, respuesta_modo_auto=False)
            guardar = preguntar_si_no(PREGUNTA_GUARDAR_IMAGEN, input_func, modo_auto=modo_auto, log_msg=log_msg)

            # Graficar la serie original
            plot_serie_inicial(
                serie=serie,
                titulo=titulo_serie,
                guardar=guardar,
                nombre_archivo=nombre_serie,
                mostrar=mostrar,
                res_dir=res_dir,
                figsize=figsize_plot,
                color=color_linea,
                linewidth=linewidth,
                alpha=alpha,
                grid_style=grid_style,
                grid_alpha=grid_alpha,
                max_labels=max_labels,
                log_msg=log_msg
            )

    except Exception as e:
        if log_msg: log_msg(f"⚠️ Error al visualizar la serie: {e}")

    if log_msg:
        log_msg("✅ Análisis exploratorio finalizado.")

    return None  
