"""
MÓDULO: main.py

Este módulo representa el punto de entrada principal del sistema de análisis,
modelado e imputación de series temporales con valores faltantes (NaNs). 

Orquesta toda la ejecución, desde la carga interactiva de datos hasta la generación
de gráficas y logs, siguiendo una secuencia lógica e interactiva de pasos
totalmente trazables. Su diseño está orientado a un flujo robusto y adaptable
a ejecución tanto manual como automatizada (`modo_auto`).

────────────────────────────────────────────────────────────────────────────
📌 FUNCIONALIDADES PRINCIPALES:

1. Carga de datos y preprocesamiento:
   - `cargar_datos_interactivamente()`: Carga un archivo desde un directorio base
     y devuelve la serie, resumen estadístico y hash único del archivo.

2. Análisis exploratorio:
   - `analisis_exploratorio_interactivo()`: Genera gráficas preliminares para
     ayudar a entender la estructura y distribución de los datos.

3. Selección de tramo óptimo para modelado:
   - `ejecutar_modelado_arma()`: Permite seleccionar (manual o automáticamente)
     el tramo más informativo para calibrar modelos ARMA.

4. Búsqueda y validación de modelos ARMA:
   - `run_arma_grid_search()`: Ejecuta una búsqueda de modelos ARMA dentro de un
     rango de órdenes, almacenando resultados y estadísticas.
   - `seleccionar_mejor_modelo_desde_df()`: Filtra y selecciona el modelo óptimo
     usando criterios estadísticos de calidad (AIC, normalidad, autocorrelación, etc.).

5. Imputación basada en el mejor modelo:
   - `imputar_bloques_arima_con_ciclo()`: Imputa todos los NaNs de la serie
     usando modelos ARIMA locales por bloque, de manera iterativa.
   - `procesar_serie_imputada()`: Guarda y grafica los resultados de la imputación
     (serie completa, bloques, espectro, CSV, etc.).

6. Registro de ejecución:
   - Todo el flujo está trazado mediante `log_msg()`, que permite visualizar
     en consola y guardar en archivo `.txt` un log completo del proceso.

────────────────────────────────────────────────────────────────────────────

"""


# ==============================================================
# 🧱 1. LIBRERÍAS ESTÁNDAR DE PYTHON
# ==============================================================

import os
import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

# ==============================================================
# 📦 2. LIBRERÍAS DE TERCEROS
# ==============================================================

import pandas as pd

# ==============================================================
# 📦 3. IMPORTS INTERNOS (otros módulos del proyecto)
# ==============================================================

# Entrada y configuración
from Funciones_iniciales import cargar_datos_interactivamente
from Funcion_configuracion_rutas import ConfigRutaResultados as configurar_rutas_resultados
from Funcion_log import crear_logger

# Preanálisis visual
from Funciones_preanalisis import analisis_exploratorio_interactivo

# Tramo y modelado ARMA
from Funcion_tramo_inteligente import ejecutar_modelado_arma
from Funcion_arma import run_arma_grid_search

# Validación de modelos
from Funcion_validacion_modelos import seleccionar_mejor_modelo_desde_df

# Imputación
from Funcion_imputacion import imputar_bloques_arima_con_ciclo

# Visualización y guardado
from Funcion_graficas_guardado import procesar_serie_imputada

# ==============================================================
# ⚙️ 4. CONFIGURACIÓN GLOBAL DESDE CONFIG.PY
# ==============================================================

from config import (
    # 🗂️ Rutas base y nombres
    BASE_DIR,
    NOMBRE_BASE_IMPUTACIONES,
    NOMBRE_FFT,
    NOMBRE_GRAFICA_BLOQUES,

    # ⚙️ Parámetros de modelado ARMA
    COV_TYPE,
    CRITERIO_SELECCION_MODELO,
    FORZAR_ESTACIONARIA,
    FORZAR_INVERTIBILIDAD,
    MAX_ITER,
    MAX_ITER_FINAL,
    METRIC,
    NJOBS,
    N_GRUPOS,
    ORDEN_MAX,
    ORDEN_MIN,
    TOP_N,
    TREND,
    USO_PARAMETROS_INICIALES_ARMA,

    # ✅ Validaciones estadísticas
    ALPHA_TESTS_AUTOCORR,
    ALPHA_TESTS_HETEROCED,
    ALPHA_TESTS_NORMALIDAD,
    FALLOS_MAXIMOS_NORMALIDAD,
    USAR_TESTS_AUTOCORRELACION,
    USAR_TESTS_HETEROCEDASTICIDAD,
    USAR_TESTS_NORMALIDAD,

    # 📊 Visualización
    CONTEXT_WINDOW_BLOQUES_IMPUTADOS,
    GUARDAR_GRAFICAS_FFT,
    MAX_FREQ_VISUALIZACION_FFT,
    MOSTRAR_GRAFICAS_FFT,
    MOSTRAR_GRAFICOS_BLOQUES,
    MOSTRAR_GRAFICOS_IMPUTACION,
    UMBRAL_FFT_DOMINANTES,
    DELTA_T_FFT,

    # Modo automático
    INDICE_AUTO,
    MODO_AUTO,
)

def init(
    base_dir: str = BASE_DIR,
    input_func: Callable[[str], str] = input, 
    modo_auto=MODO_AUTO,
    indice_auto= INDICE_AUTO
):
    """
    Función principal que orquesta la ejecución completa del sistema de análisis,
    modelado e imputación de series temporales con valores faltantes.

    Realiza una ejecución secuencial y trazable que incluye:
    - Carga interactiva o automática de datos desde un directorio base.
    - Análisis exploratorio preliminar con visualizaciones.
    - Selección automática o manual del tramo óptimo para modelado ARMA.
    - Búsqueda y validación de modelos ARMA sobre el tramo seleccionado.
    - Imputación iterativa de valores faltantes en la serie completa con el modelo óptimo.
    - Generación de gráficos y exportación de resultados.
    - Registro detallado de logs con mensajes para seguimiento y debugging.

    Parámetros:
    -----------
    base_dir (str): Directorio base donde se buscan los archivos de entrada.
    input_func (Callable): Función para solicitar entradas al usuario, por defecto `input`.
    modo_auto (bool): Indica si el flujo es completamente automático (sin interacción).
    indice_auto (int): Índice del archivo a procesar en modo automático.

    Nota:
    -----
    En modo automático (`modo_auto=True`), la función procesa archivos en lote
    sin requerir intervención manual, ideal para ejecución por lotes o entornos batch.
    Para el modo automático, se deberá de introducir los csv de manera que los datos siempre se encuentren 
    en la columna 0, y el índice (si se desea conservar) en la columna 1

    """
    # Evitar errores en el modo automático
    import matplotlib
    if modo_auto:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    #  Crear lista para guardar logs
    log_lines = []
    log_msg = crear_logger(verbose=True, log_buffer=log_lines)
    tramo_seleccionado = None  

    try:
        # ─────────────────────────────────────────────
        #  Cargar datos
        df, resumen, hash_archivo, nombre_archivo = cargar_datos_interactivamente(
            base_dir=base_dir,
            input_func=input_func,
            log_msg=log_msg,
            modo_auto=modo_auto,
            indice_auto=indice_auto
        )

        if df is None:
            log_msg("⏹ Operación cancelada o archivo no seleccionado.")
            return

        rutas_modificadas= configurar_rutas_resultados(hash_archivo,nombre_archivo)
        # ─────────────────────────────────────────────
        # 2. Análisis exploratorio
        analisis_exploratorio_interactivo(df,resumen,input_func=input_func,log_msg=log_msg,res_dir=rutas_modificadas.grafica_preanalisis,modo_auto=modo_auto)
        # ─────────────────────────────────────────────
        #  Modelado ARMA con selección automática de tramo
        tramo_seleccionado= ejecutar_modelado_arma(df,input_func=input_func,directorio_graficas=rutas_modificadas.grafica_tramo,log_msg=log_msg,modo_auto=modo_auto)

        # ─────────────────────────────────────────────
        #  Búsqueda de modelos ARMA en el tramo seleccionado
        if tramo_seleccionado is not None:
            log_msg("🚀 Ejecutando búsqueda ARMA en el tramo seleccionado...")
            
            # Creación de directorios
            os.makedirs(rutas_modificadas.parametros_globales, exist_ok=True)
            os.makedirs(os.path.dirname(rutas_modificadas.log_csv), exist_ok=True)
            os.makedirs(os.path.dirname(rutas_modificadas.tiempos_csv), exist_ok=True)

            df_modelos = run_arma_grid_search(
                y=tramo_seleccionado,
                order_min=ORDEN_MIN,
                order_max=ORDEN_MAX,
                max_iter=MAX_ITER,
                n_grupos=N_GRUPOS,
                njobs=NJOBS,
                metric=METRIC,
                top_n=TOP_N,
                max_iter_final=MAX_ITER_FINAL,
                log_csv=rutas_modificadas.log_csv,
                timing_log_csv=rutas_modificadas.tiempos_csv,
                dataset_name=hash_archivo, 
                params_dir=rutas_modificadas.parametros_globales,
                trend=TREND,
                enforce_stationarity=FORZAR_ESTACIONARIA,
                enforce_invertibility=FORZAR_INVERTIBILIDAD,
                cov_type=COV_TYPE,
                log_msg=log_msg,
                uso_parametros_iniciales=USO_PARAMETROS_INICIALES_ARMA
            )

            log_msg(f"✅ Búsqueda ARMA completada. {len(df_modelos)} modelos generados.")
            # ─────────────────────────────────────────────
            # 5. Selección del mejor modelo validado
            log_msg("🔍 Seleccionando mejor modelo validado...")
            modelo_optimo= seleccionar_mejor_modelo_desde_df(
                carpeta_modelos=df_modelos,
                criterio=CRITERIO_SELECCION_MODELO,
                usar_normalidad=USAR_TESTS_NORMALIDAD,
                usar_autocorrelacion=USAR_TESTS_AUTOCORRELACION,
                usar_heterocedasticidad=USAR_TESTS_HETEROCEDASTICIDAD,
                fallos_maximos=FALLOS_MAXIMOS_NORMALIDAD,
                alpha=ALPHA_TESTS_NORMALIDAD,
                alpha_auto=ALPHA_TESTS_AUTOCORR,
                alpha_heter=ALPHA_TESTS_HETEROCED,
                log_msg=log_msg
            )
            if modelo_optimo:
                log_msg("✅ Modelo seleccionado correctamente.")
            else:
                log_msg("⚠️ No se encontró un modelo válido tras las validaciones.")
            # ─────────────────────────────────────────────
            # 6. Imputación ARIMA en la serie original completa
            if modelo_optimo:
                log_msg("🧩 Ejecutando imputación con modelo óptimo...")
                try:

                    serie_imputada, bloques_info = imputar_bloques_arima_con_ciclo(
                        serie=df.squeeze(),  
                        modelo=modelo_optimo,
                        log_msg=log_msg 
                    )
                    # Restaurár índice original
                    serie_imputada.index = df.index  
                    log_msg(f"✅ Imputación completada. {len(bloques_info)} bloques procesados.")

                    procesar_serie_imputada(
                        serie_original=df.squeeze(),
                        serie_imputada=serie_imputada,
                        bloques_imputados=bloques_info,

                        # Nombres base
                        nombre_base_fft=NOMBRE_FFT,
                        nombre_base_bloques=NOMBRE_GRAFICA_BLOQUES,
                        nombre_base_imputaciones=NOMBRE_BASE_IMPUTACIONES,

                        # CSV
                        carpeta_salida_csv=rutas_modificadas.imputados,

                        # Serie completa
                        carpeta_salida_serie=rutas_modificadas.grafica_impt,
                        mostrar_serie=MOSTRAR_GRAFICOS_IMPUTACION,

                        # Bloques individuales
                        carpeta_salida_bloques=rutas_modificadas.grafica_bloques,
                        contexto_bloques=CONTEXT_WINDOW_BLOQUES_IMPUTADOS,
                        mostrar_bloques=MOSTRAR_GRAFICOS_BLOQUES,

                        # FFT
                        carpeta_salida_fft=rutas_modificadas.grafica_fft,
                        delta_t=DELTA_T_FFT,
                        max_freq=MAX_FREQ_VISUALIZACION_FFT,
                        umbral_energia=UMBRAL_FFT_DOMINANTES,
                        guardar_fft=GUARDAR_GRAFICAS_FFT,
                        mostrar_fft=MOSTRAR_GRAFICAS_FFT,
                        log_msg=log_msg

                    )

                except Exception as e:
                    msg = f"❌ Error durante la imputación: {e}"
                    log_msg(msg)

    except RuntimeError as e:
        log_msg(str(e))

    except Exception as e:
        msg = f"❌ Error inesperado en la selección de tramo: {e}"
        log_msg(msg)

    finally:
        # Definir hash y carpeta de destino para el log.

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(rutas_modificadas.log_total, exist_ok=True)
        log_completo = "\n".join(log_lines)
        if log_completo.strip(): 
            ruta_log_completo = os.path.join(rutas_modificadas.log_total, f"log_completo_{timestamp}.txt")
            with open(ruta_log_completo, "w", encoding="utf-8") as f:
                f.write(log_completo)
            log_msg(f"🧾 Log completo guardado en: {ruta_log_completo}")


if __name__ == "__main__":

    """
    Punto de entrada principal. Permite elegir entre modo manual o automático.
    - Modo manual: interacción paso a paso para selección y análisis.
    - Modo automático: procesa en lote todos los archivos CSV en el directorio base.
    """
    print("¿Modo de ejecución?")
    print("1. Manual (selección interactiva)")
    print("2. Automático (procesar todos los archivos en lote)")

    opcion = input("Selecciona [1/2]: ").strip()

    if opcion == "2":
        base_dir = BASE_DIR 
        archivos = sorted(Path(base_dir).glob("*.csv"))
        if not archivos:
            print("⚠️ No se encontraron archivos .csv en la carpeta.")
        else:
            for i, archivo in enumerate(archivos, 1):
                print(f"\n🟦 Ejecutando procesamiento automático del archivo #{i} ({archivo.name})...")
                init(modo_auto=True, input_func=input, base_dir=base_dir, indice_auto=i-1)
    elif opcion == "1":
        init(modo_auto=False, input_func=input)
    else:
        print("⚠️ Opción no válida. Por favor ejecuta el programa de nuevo y elige '1' o '2'.")