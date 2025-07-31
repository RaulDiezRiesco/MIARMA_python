"""
MÓDULO: modulo_funcion_graficas_guardado.py

Tras la imputación de valores faltantes en series temporales, este módulo ofrece
herramientas para analizar y visualizar la serie completa, permitiendo una inspección
profunda de los resultados. Facilita la identificación visual de las zonas imputadas
y proporciona una visión global del comportamiento de la serie restaurada.

Además, permite descomponer la serie en bloques para análisis segmentados y realizar
análisis espectral mediante Transformada de Fourier (FFT). Estas funcionalidades
son clave para validar cuantitativamente el proceso de imputación y documentar el
desempeño del sistema.

────────────────────────────────────────────────────────────────────────────
📌 FUNCIONALIDADES PRINCIPALES:

1. Guardado estructurado de resultados:
   - `guardar_serie_imputada_csv()`: Exporta la serie imputada a CSV, manteniendo índice y valores.

2. Visualización global de la serie imputada:
   - `graficar_serie_con_imputaciones()`: Genera un gráfico completo donde se resaltan las zonas
     imputadas y no imputadas, acompañado de un resumen estadístico para comparar antes y después
     de la imputación.

3. Visualización detallada por bloques imputados:
   - `graficar_bloques_con_contexto()`: Crea gráficos individuales por cada bloque imputado,
     mostrando un margen de contexto ajustable, con valores imputados destacados e información clave
     del bloque.

4. Análisis espectral mediante FFT:
   - `graficar_fft_comparacion_nulos_imputada()` y `calcular_metricas_fft_comparacion()`:
     Comparan el espectro de frecuencias entre la serie original con nulos y la imputada, mostrando
     gráficamente las diferencias y calculando métricas para evaluar la conservación espectral
     y la calidad de la imputación.

5. Orquestación integral del postprocesamiento:
   - `procesar_serie_imputada()`: Función principal que coordina todo el flujo post-imputación:
     guardado CSV, gráficos globales, gráficos por bloques y análisis FFT, automatizando la
     evaluación y documentación del proceso.

────────────────────────────────────────────────────────────────────────────
RETORNO FINAL:
   - Csv de la serie imputada
   - Gráficas 
        - Serie completa
        - Por bloques de nulos
        - FFT

"""
# ==============================================================
# 🧱 1. LIBRERÍAS ESTÁNDAR 
# ==============================================================

import os
from pathlib import Path
from itertools import groupby
from operator import itemgetter
from typing import List, Dict, Any, Optional, Callable, Tuple

# ==============================================================
# 📦 2. LIBRERÍAS DE TERCEROS
# ==============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import pearsonr, spearmanr

# ==============================================================
# ⚙️ 3. CONFIGURACIÓN GLOBAL DESDE CONFIG.PY
# ==============================================================

from config import (
    # ─── Estilo y estética de gráficos ──────────────────────────
    PLOT_FIGSIZE,
    PLOT_LINEWIDTH,
    PLOT_GRID_STYLE,
    PLOT_GRID_ALPHA,

    # ─── Control de visualización automática ────────────────────
    MOSTRAR_GRAFICOS_IMPUTACION,
    MOSTRAR_GRAFICOS_BLOQUES,
    MOSTRAR_GRAFICAS_FFT,
    MODO_AUTO,

    # ─── Parámetros gráficos por bloque ─────────────────────────
    CONTEXT_WINDOW_BLOQUES_IMPUTADOS,

    # ─── Parámetros gráficos espectrales ────────────────────────
    MAX_FREQ_VISUALIZACION_FFT,
    GUARDAR_GRAFICAS_FFT,
    UMBRAL_FFT_DOMINANTES,

    # ─── Nombres base para guardado ─────────────────────────────
    NOMBRE_BASE_IMPUTACIONES,
    NOMBRE_GRAFICA_BLOQUES,
    NOMBRE_FFT,

    # ─── Rutas de salida ────────────────────────────────────────
    SALIDA_DATOS_IMPUTADOS,
    SALIDA_GRAFICA_IMPT,
    SALIDA_GRAFICA_BLOQUES,
    SALIDA_GRAFICA_FFT,

)

def guardar_serie_imputada_csv(
    serie_imputada: pd.Series,
    nombre_base_imputaciones: str = NOMBRE_BASE_IMPUTACIONES,
    carpeta_salida: str = SALIDA_DATOS_IMPUTADOS,
    log_msg: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Guarda una serie imputada como CSV, con su índice y una columna 'valor'.

    Parámetros:
        serie_imputada (pd.Series): Serie ya imputada, con índice.
        nombre_base_imputaciones (str): Nombre base del archivo.
        carpeta_salida (str): Carpeta de salida.
        log_msg (Callable, opcional): Función de log.

    Retorna:
        None
    """
    import os

    os.makedirs(carpeta_salida, exist_ok=True)

    ruta = os.path.join(
        carpeta_salida,
        f"{nombre_base_imputaciones}_ARMA_imputada.csv"
    )

    serie_a_guardar = serie_imputada.copy()
    serie_a_guardar.name = "valor"

    serie_a_guardar.to_csv(
        ruta,
        index=True,
        index_label="index"
    )


    log_msg(f"💾 Serie imputada guardada en: {ruta}")

def graficar_serie_con_imputaciones(
    serie_original: pd.Series,
    serie_imputada: pd.Series,
    nombre_base_imputaciones: str = NOMBRE_BASE_IMPUTACIONES,
    carpeta_salida: str = SALIDA_GRAFICA_IMPT,
    mostrar: bool = MOSTRAR_GRAFICOS_IMPUTACION,
    modo_auto: bool = MODO_AUTO,
    log_msg: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Genera y guarda un gráfico de la serie imputada, destacando:
    - La serie imputada como línea continua.
    - Zonas donde no se pudo imputar (NaNs persistentes).
    - Un resumen textual con:
        - Conteo de NaNs originales, imputados, no imputados y modificados.
        - Estadísticas antes y después de la imputación (media y desviación estándar).
        - Diferencias estadísticas introducidas por la imputación.

    Parámetros:
        serie_original (pd.Series): Serie temporal original con posibles valores faltantes (NaNs).
        serie_imputada (pd.Series): Serie imputada (misma longitud que la original).
        nombre_base_imputaciones (str): Nombre base para el archivo de salida (PNG).
        carpeta_salida (str): Carpeta donde se guarda la imagen generada.
        mostrar (bool): Si True, se muestra el gráfico en pantalla (ignorado si modo_auto=True).
        modo_auto (bool): Si True, solo guarda la imagen sin mostrarla (modo no interactivo).
        log_msg (Callable, opcional): Función de logging para registrar mensajes del proceso.

    Retorna:
        None: No devuelve nada, solo guarda y/o muestra la figura.
    """
    
    assert len(serie_original) == len(serie_imputada), "Las series deben tener la misma longitud."

    # Detección de cambios 
    nulos_original = serie_original.isna()
    nulos_imputada = serie_imputada.isna()
    imputados = nulos_original & serie_imputada.notna()
    no_imputados = nulos_original & nulos_imputada
    diferencia_valores = (serie_original != serie_imputada) & ~nulos_original

    # Estadísticas globales
    media_original = serie_original.mean()
    media_imputada = serie_imputada.mean()
    std_original = serie_original.std()
    std_imputada = serie_imputada.std()

    resumen_texto = (
        f"Total: {len(serie_original)}\n"
        f"NaNs originales: {nulos_original.sum()} ({nulos_original.mean() * 100:.1f}%)\n"
        f"Imputados: {imputados.sum()}\n"
        f"No imputados: {no_imputados.sum()}\n"
        f"Modificados sin ser NaN: {diferencia_valores.sum()}\n"
        f"\nMedia original: {media_original:.3f}\n"
        f"Media imputada: {media_imputada:.3f}\n"
        f"Δ Media: {(media_imputada - media_original):.3f}\n"
        f"\nStd original: {std_original:.3f}\n"
        f"Std imputada: {std_imputada:.3f}\n"
        f"Δ Std: {(std_imputada - std_original):.3f}"
    )

    # Bloques no imputados 
    positions_no_imputados = np.flatnonzero(no_imputados)
    bloques_no_imputados = []
    for _, g in groupby(enumerate(positions_no_imputados), lambda ix: ix[0] - ix[1]):
        grupo = list(map(itemgetter(1), g))
        bloques_no_imputados.append((grupo[0], grupo[-1]))

    # Gráfico 
    plt.figure(figsize=PLOT_FIGSIZE)
    plt.plot(serie_imputada.index, serie_imputada, label="Serie Imputada", linewidth=PLOT_LINEWIDTH)

    for inicio, fin in bloques_no_imputados:
        plt.axvspan(
            serie_imputada.index[inicio],
            serie_imputada.index[fin],
            color='orange',
            alpha=0.2,
            label="Zona no imputada"
        )

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.title(f"{nombre_base_imputaciones} – Serie imputada", fontsize=14)
    plt.xlabel("Índice")
    plt.ylabel("Valor")
    plt.grid(True, linestyle=PLOT_GRID_STYLE, alpha=PLOT_GRID_ALPHA)

    # Si el índice es de tipo datetime, aplicar formato legible en eje X
    if isinstance(serie_imputada.index, pd.DatetimeIndex):
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gcf().autofmt_xdate()  # Rota las fechas automáticamente

    # Anotación resumen
    plt.annotate(
        resumen_texto,
        xy=(1.01, 0.5),
        xycoords='axes fraction',
        fontsize=10,
        verticalalignment='center',
        bbox=dict(boxstyle="round", alpha=0.1)
    )

    # Guardado 
    from pathlib import Path
    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    ruta_grafico = carpeta_salida / f"{nombre_base_imputaciones}_serie_completa.png"
    plt.savefig(ruta_grafico, bbox_inches="tight")

    log_msg(f"📈 Gráfico guardado en: {ruta_grafico}")

    if mostrar and not modo_auto:
        plt.show()
    else:
        plt.close()

def graficar_bloques_con_contexto(
    serie_referencia: pd.Series,
    serie_imputada: pd.Series,
    bloques: List[Dict[str, Any]],
    nombre_base_bloques: str = NOMBRE_GRAFICA_BLOQUES,
    carpeta_salida: str = SALIDA_GRAFICA_BLOQUES,
    contexto: int = CONTEXT_WINDOW_BLOQUES_IMPUTADOS,
    mostrar: bool = MOSTRAR_GRAFICOS_BLOQUES,
    modo_auto: bool = MODO_AUTO,
    log_msg: Optional[Callable[[str], None]] = None
) -> None:
    """
    Genera gráficos por bloque imputado, incluyendo ventana de contexto antes y después.

    Parámetros:
        serie_referencia (pd.Series): Serie original con NaNs.
        serie_imputada (pd.Series): Serie resultante de imputación.
        bloques (List[Dict]): Lista de bloques con info como 'inicio', 'fin', etc.
        nombre_base_bloques (str): Nombre base para los archivos guardados.
        carpeta_salida (str): Carpeta donde se guardarán los gráficos.
        contexto (int): Puntos de contexto antes y después del bloque.
        mostrar (bool): Si True, muestra el gráfico (ignorado si modo_auto=True).
        modo_auto (bool): Si True, nunca muestra los gráficos, solo guarda.
        log_msg (Callable, opcional): Función para registrar logs del proceso.

    
    Resultado:
        None: No devuelve nada, solo guarda.
    """

    os.makedirs(carpeta_salida, exist_ok=True)

    for i, bloque in enumerate(bloques):
        idx_ini = max(0, bloque["inicio"] - contexto)
        idx_fin = min(len(serie_referencia), bloque["fin"] + contexto + 1)

        rango_idx = serie_referencia.index[idx_ini:idx_fin]
        bloque_idx = serie_referencia.index[bloque["inicio"]:bloque["fin"] + 1]

        plt.figure(figsize=PLOT_FIGSIZE)
        # Línea imputada completa
        plt.plot(rango_idx, serie_imputada.loc[rango_idx], label="Serie imputada", linewidth=PLOT_LINEWIDTH)

        # Sombra amarilla en zona imputada
        plt.axvspan(
            bloque_idx[0],
            bloque_idx[-1],
            color="yellow",
            alpha=0.3,
            label="Zona imputada"
        )
        # Tamaño dinámico del punto según el tamaño del bloque
        if bloque["tamano"] <= 200:
            tam_punto = 20
        elif bloque["tamano"] <= 500:
            tam_punto = 8
        else:
            tam_punto = 4  

        plt.scatter(
            bloque_idx,
            serie_imputada.loc[bloque_idx],
            color="red",
            s=tam_punto,
            label="Imputados",
            zorder=5
        )

        # Título informativo
        titulo = (
            f"{nombre_base_bloques} – Bloque {i+1} ({bloque['inicio']}–{bloque['fin']})\n"
            f"Tam: {bloque['tamano']} | Libres izq: {bloque['libres_izq']} | Libres der: {bloque['libres_der']}"
        )
        if "situacion" in bloque:
            titulo += f" | {bloque['situacion']}"
        plt.title(titulo, fontsize=12)

        plt.xlabel("Índice")
        plt.ylabel("Valor")
        plt.grid(True, linestyle=PLOT_GRID_STYLE, alpha=PLOT_GRID_ALPHA)

        # Leyenda sin duplicados
        handles, labels = plt.gca().get_legend_handles_labels()
        unique = dict(zip(labels, handles))  # labels como clave evita duplicados
        plt.legend(unique.values(), unique.keys())

        # Guardar archivo
        ruta_archivo = os.path.join(carpeta_salida, f"{nombre_base_bloques}_bloque_{i+1}_contexto.png")
        plt.savefig(ruta_archivo, bbox_inches="tight")


        log_msg(f"🧩 Gráfico bloque {i+1} guardado en: {ruta_archivo}")

        if mostrar and not modo_auto:
            plt.show()
        else:
            plt.close()

def calcular_metricas_fft_comparacion(
    freqs_nulos: np.ndarray,
    mags_nulos: np.ndarray,
    freqs_imp: np.ndarray,
    mags_imp: np.ndarray,
    n_nulos: int,
    umbral_energia: float = UMBRAL_FFT_DOMINANTES
) -> str:
    """
    Calcula un conjunto de métricas comparativas entre el espectro de frecuencias de 
    una serie con valores faltantes (NaNs) y su versión imputada.

    Las métricas incluyen comparación de frecuencias dominantes, medidas estadísticas
    (media y desviación estándar), índice de conservación espectral (SCI), y correlaciones.

    Parámetros:
        freqs_nulos (np.ndarray): Array de frecuencias obtenidas de la FFT de la serie original (con NaNs).
        mags_nulos (np.ndarray): Magnitudes correspondientes a `freqs_nulos`.
        freqs_imp (np.ndarray): Array de frecuencias de la FFT de la serie imputada.
        mags_imp (np.ndarray): Magnitudes correspondientes a `freqs_imp`.
        n_nulos (int): Número total de valores que fueron imputados en la serie.
        umbral_energia (float): Porcentaje acumulado de energía (en [0, 1]) usado para definir
            cuántas frecuencias se consideran "dominantes". Por defecto: `UMBRAL_FFT_DOMINANTES`.

    Retorna:
        str: Texto multilinea con el resumen de las métricas calculadas.
            Incluye:
            - Conteo de frecuencias dominantes y cuántas fueron conservadas.
            - Estadísticas básicas (media y std) de magnitudes antes y después.
            - Diferencias absolutas de media y std.
            - SCI (Spectral Conservation Index): índice entre 0 y 1.
            - Correlación espectral: Pearson y Spearman.

    Detalles:
        - SCI (Spectral Conservation Index): mide la similitud general del espectro,
          considerando la superposición entre ambas FFT. Valores cercanos a 1 indican alta conservación.
        - Las frecuencias dominantes se ordenan por energía (magnitud) y se seleccionan
          aquellas necesarias para alcanzar el umbral acumulado indicado.

    """
    # Ordenar por magnitud descendente para frecuencias dominantes
    orden_nulos = np.argsort(mags_nulos)[::-1]
    mags_ordenadas = mags_nulos[orden_nulos]
    frecs_ordenadas = freqs_nulos[orden_nulos]

    # Determinar cuántas frecuencias representan el umbral de energía total
    energia_total = np.sum(mags_ordenadas)
    energia_acumulada = np.cumsum(mags_ordenadas)
    n_dom = np.searchsorted(energia_acumulada, umbral_energia * energia_total) + 1

    frecs_dom = np.round(frecs_ordenadas[:n_dom], 4)
    frecs_imp_dom = np.round(freqs_imp[np.argsort(mags_imp)[::-1][:n_dom]], 4)
    conservadas = sum(np.isin(frecs_dom, frecs_imp_dom))

    # Estadísticas básicas
    media_nulos = np.nanmean(mags_nulos)
    media_imp = np.nanmean(mags_imp)
    std_nulos = np.nanstd(mags_nulos)
    std_imp = np.nanstd(mags_imp)
    dif_media = media_imp - media_nulos
    dif_std = std_imp - std_nulos

    # Índice de Conservación Espectral (SCI)
    numerador = np.sum(np.minimum(mags_nulos, mags_imp))
    denominador = np.sum(np.maximum(mags_nulos, mags_imp))
    sci = numerador / denominador if denominador != 0 else np.nan

    # Correlaciones espectrales
    corr_pearson = pearsonr(mags_nulos, mags_imp)[0]
    corr_spearman = spearmanr(mags_nulos, mags_imp)[0]

    # Formato resumen
    resumen = "\n".join([
        f"Nº valores imputados:                      {n_nulos}",
        f"Frecuencias dominantes detectadas:         {n_dom} (según umbral {umbral_energia:.0%})",
        f"Frecuencias conservadas en imputación:     {conservadas} / {n_dom}",
        "",
        f"Media magnitud original:                   {media_nulos:.4f}",
        f"Media magnitud imputada:                   {media_imp:.4f}",
        f"Δ Media:                                   {dif_media:.4f}",
        "",
        f"STD magnitud original:                     {std_nulos:.4f}",
        f"STD magnitud imputada:                     {std_imp:.4f}",
        f"Δ STD:                                     {dif_std:.4f}",
        "",
        f"SCI (Índice conservación espectral):       {sci:.4f}",
        f"Correlación espectral (Pearson):           {corr_pearson:.4f}",
        f"Correlación espectral (Spearman):          {corr_spearman:.4f}",
    ])

    return resumen

def graficar_fft_comparacion_nulos_imputada(
    serie_nulos: pd.Series,
    serie_imputada: pd.Series,
    nombre_base_fft: str = NOMBRE_FFT,
    ruta_guardado: Path = SALIDA_GRAFICA_FFT,
    delta_t: float = 1.0,
    max_freq: float = MAX_FREQ_VISUALIZACION_FFT,
    umbral_energia: float = UMBRAL_FFT_DOMINANTES,
    guardar: bool = GUARDAR_GRAFICAS_FFT,
    mostrar: bool = MOSTRAR_GRAFICAS_FFT,
    modo_auto: bool = MODO_AUTO,
    log_msg: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Genera una figura de tres paneles comparando la FFT de una serie con nulos (rellenados con ceros)
    frente a su versión imputada. También se incluyen métricas espectrales detalladas.

    Paneles:
        1. FFT de la serie original con nulos (NaNs convertidos temporalmente en ceros).
        2. FFT de la serie imputada.
        3. Diferencia espectral entre ambas (error absoluto).

    Parámetros:
        serie_nulos (pd.Series): Serie original con valores NaN.
        serie_imputada (pd.Series): Serie imputada, sin valores NaN.
        nombre_base_fft (str): Nombre base del archivo gráfico a guardar.
        ruta_guardado (Path): Carpeta de salida para el gráfico.
        delta_t (float): Intervalo de muestreo entre observaciones (frecuencia inversa).
        max_freq (float): Frecuencia máxima a mostrar en el eje X.
        umbral_energia (float): Umbral de energía acumulada para definir frecuencias dominantes.
        guardar (bool): Si True, guarda el gráfico como PNG.
        mostrar (bool): Si True, muestra la figura en pantalla (excepto si `modo_auto=True`).
        modo_auto (bool): Si True, fuerza `mostrar = False` (modo automático).
        log_msg (Callable, opcional): Función para registrar mensajes.

    Retorna:
        None: No devuelve nada, solo guarda.
    """
    if modo_auto:
        mostrar = False

    ruta_guardado = Path(ruta_guardado)
    ruta_guardado.mkdir(parents=True, exist_ok=True)

    # FUNCIÓN AUXILIAR: FFT de una serie temporal
    def fft_info(signal: pd.Series, delta_t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula la transformada rápida de Fourier (FFT) de una serie,
        retornando frecuencias positivas y sus magnitudes.

        Args:
            signal (pd.Series): Serie temporal (sin NaNs).
            delta_t (float): Intervalo de muestreo entre observaciones.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Frecuencias positivas y magnitudes asociadas.
        """
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), d=delta_t)
        mags = np.abs(fft)[:len(signal) // 2]
        freqs = freqs[:len(signal) // 2]
        return freqs, mags

    # CÁLCULO FFTs Y DIFERENCIA ESPECTRAL
    freqs_nulos, mags_nulos = fft_info(serie_nulos.fillna(0), delta_t)
    freqs_imp, mags_imp = fft_info(serie_imputada, delta_t)
    error_espectral = mags_nulos - mags_imp
    filtro_visual = freqs_nulos < max_freq

    # CONFIGURACIÓN DE LOS SUBPANELES

    _, axs = plt.subplots(3, 1, figsize=(14, 15), gridspec_kw={'height_ratios': [1.5, 1.5, 2]})

    axs[0].plot(freqs_nulos[filtro_visual], mags_nulos[filtro_visual],
                label="Serie con nulos (NaNs = 0)", color='gray')
    axs[0].set_title("FFT – Serie original con nulos")
    axs[0].set_ylabel("Magnitud")
    axs[0].set_xlabel("Frecuencia")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(freqs_imp[filtro_visual], mags_imp[filtro_visual],
                label="Serie imputada", color='orange')
    axs[1].set_title("FFT – Serie imputada")
    axs[1].set_ylabel("Magnitud")
    axs[1].set_xlabel("Frecuencia")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(freqs_nulos[filtro_visual], error_espectral[filtro_visual],
                label="Error espectral (original - imputada)", color='purple')
    axs[2].set_title("Diferencia espectral")
    axs[2].set_xlabel("Frecuencia")
    axs[2].set_ylabel("Diferencia de magnitud")
    axs[2].legend()
    axs[2].grid(True)

    # CÁLCULO DE MÉTRICAS Y ANOTACIÓN EN GRÁFICO
    resumen = calcular_metricas_fft_comparacion(
        freqs_nulos=freqs_nulos,
        mags_nulos=mags_nulos,
        freqs_imp=freqs_imp,
        mags_imp=mags_imp,
        n_nulos=serie_nulos.isna().sum(),
        umbral_energia=umbral_energia
    )

    axs[2].text(1.02, 0.5, resumen, transform=axs[2].transAxes,
                fontsize=9, verticalalignment='center', family='monospace',
                bbox=dict(boxstyle="round", facecolor='whitesmoke', alpha=0.2))

    plt.tight_layout()

    # GUARDADO Y OPCIONAL DISPLAY
    ruta_figura = ruta_guardado / f"fft_comparacion_{nombre_base_fft}.png"

    if guardar:
        plt.savefig(ruta_figura, dpi=300, bbox_inches="tight")
        log_msg(f"✅ Imagen FFT guardada en: {ruta_figura}")

    if mostrar:
        plt.show()
    else:
        plt.close()

    return 

def procesar_serie_imputada(
    serie_original: pd.Series,
    serie_imputada: pd.Series,
    bloques_imputados: List[Dict[str, Any]],
    nombre_base_imputaciones: str = NOMBRE_BASE_IMPUTACIONES,
    nombre_base_bloques: str = NOMBRE_GRAFICA_BLOQUES,
    nombre_base_fft: str = NOMBRE_FFT,
    # === Parámetros para guardar CSV ===
    carpeta_salida_csv: str = SALIDA_DATOS_IMPUTADOS,
    # === Parámetros para gráfico serie completa ===
    carpeta_salida_serie: str = SALIDA_GRAFICA_IMPT,
    mostrar_serie: bool = MOSTRAR_GRAFICOS_IMPUTACION,
    # === Parámetros para gráfico por bloques ===
    carpeta_salida_bloques: str = SALIDA_GRAFICA_BLOQUES,
    contexto_bloques: int = CONTEXT_WINDOW_BLOQUES_IMPUTADOS,
    mostrar_bloques: bool = MOSTRAR_GRAFICOS_BLOQUES,
    # === Parámetros para gráfico FFT ===
    carpeta_salida_fft: Path = SALIDA_GRAFICA_FFT,
    delta_t: float = 1.0,
    max_freq: float = MAX_FREQ_VISUALIZACION_FFT,
    umbral_energia: float = UMBRAL_FFT_DOMINANTES,
    guardar_fft: bool = GUARDAR_GRAFICAS_FFT,
    mostrar_fft: bool = MOSTRAR_GRAFICAS_FFT,
    # === Control general ===
    modo_auto: bool = MODO_AUTO,
    log_msg: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Función central de post-procesamiento tras imputar una serie. Orquesta:
        - Guardado de CSV de la serie imputada.
        - Generación de gráfico completo con valores imputados.
        - Generación de gráficos individuales por bloque con contexto.
        - Comparación espectral vía FFT.

    Parámetros:
        serie_original (pd.Series): Serie con valores faltantes (NaNs).
        serie_imputada (pd.Series): Serie con los NaNs imputados.
        bloques_imputados (List[Dict]): Lista con info de bloques imputados.
        nombre_base_imputaciones (str): Nombre base para los archivos.
        nombre_base_bloques (str): Nombre base para los gráficos por bloque.
        nombre_base_fft (str): Nombre base para el gráfico FFT.
        carpeta_salida_csv (str): Carpeta donde guardar el CSV de la serie imputada.
        carpeta_salida_serie (str): Carpeta donde guardar el gráfico general.
        mostrar_serie (bool): Mostrar en pantalla el gráfico completo.
        carpeta_salida_bloques (str): Carpeta para guardar gráficos por bloque.
        contexto_bloques (int): Puntos antes/después del bloque a mostrar.
        mostrar_bloques (bool): Mostrar en pantalla los gráficos de bloque.
        carpeta_salida_fft (Path): Carpeta para guardar el gráfico FFT.
        delta_t (float): Intervalo de muestreo entre puntos.
        max_freq (float): Frecuencia máxima a visualizar en el gráfico FFT.
        umbral_energia (float): Umbral de energía acumulada para FFT.
        guardar_fft (bool): Si True, guarda el gráfico FFT.
        mostrar_fft (bool): Mostrar gráfico FFT.
        modo_auto (bool): Si True, fuerza `mostrar_* = False` (modo batch/silencioso).
        log_msg (Callable, optional): Función para registrar mensajes (log buffer o print).
    """
    if modo_auto:
        mostrar_serie = False
        mostrar_bloques = False
        mostrar_fft = False

    log_msg(f"🚀 Iniciando procesamiento para: {nombre_base_imputaciones}")

    # 1️⃣ Guardado de la serie imputada
    guardar_serie_imputada_csv(
        serie_imputada=serie_imputada,
        nombre_base_imputaciones=nombre_base_imputaciones,
        carpeta_salida=carpeta_salida_csv,
        log_msg=log_msg
    )

    # 2️⃣ Gráfico completo con valores imputados
    graficar_serie_con_imputaciones(
        serie_original=serie_original,
        serie_imputada=serie_imputada,
        nombre_base_imputaciones=nombre_base_imputaciones,
        carpeta_salida=carpeta_salida_serie,
        mostrar=mostrar_serie,
        modo_auto=modo_auto,
        log_msg=log_msg
    )

    # 3️⃣ Gráficos individuales por bloque
    graficar_bloques_con_contexto(
        serie_referencia=serie_original,
        serie_imputada=serie_imputada,
        bloques=bloques_imputados,
        nombre_base_bloques=nombre_base_bloques,
        carpeta_salida=carpeta_salida_bloques,
        contexto=contexto_bloques,
        mostrar=mostrar_bloques,
        modo_auto=modo_auto,
        log_msg=log_msg
    )

    # 4️⃣ Comparación espectral (FFT)
    graficar_fft_comparacion_nulos_imputada(
        serie_nulos=serie_original,
        serie_imputada=serie_imputada,
        nombre_base_fft=nombre_base_fft,
        ruta_guardado=carpeta_salida_fft,
        delta_t=delta_t,
        max_freq=max_freq,
        umbral_energia=umbral_energia,
        guardar=guardar_fft,
        mostrar=mostrar_fft,
        modo_auto=modo_auto,
        log_msg=log_msg
    )

    log_msg(f"✅ Finalizado procesamiento de: {nombre_base_imputaciones}")
