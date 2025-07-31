"""
Modulo: config_ruta_resultados.py

Este módulo define la clase `ConfigRutaResultados`, utilizada para generar 
y centralizar las rutas de carpetas y archivos de salida relacionados con 
el proceso de imputación de datos mediante modelos ARMA.

El objetivo principal es estructurar de manera coherente las rutas de trabajo 
según un identificador único (`hash_datos`) junto al nombre del archivo, facilitando así el almacenamiento 
y recuperación de resultados, gráficas, parámetros y logs.

Librerías utilizadas:
---------------------
- os: Para crear rutas de archivos y carpetas de manera compatible con cualquier sistema operativo.
- config.RES_DIR: Variable importada desde el módulo de configuración, que define la ruta base de resultados.
"""

import os
from config import RES_DIR

class ConfigRutaResultados:
    """
    Clase para construir y organizar las rutas de salida de resultados 
    del proceso de imputación ARMA, basado en un identificador de datos (hash_datos) y el nombre del archivo.

    Atributos:
        base (str): Ruta base de trabajo para un conjunto de datos específico.
        imputados (str): Ruta a los datos imputados.
        graficas (str): Carpeta principal de gráficas.
        grafica_impt (str): Gráficas de imputaciones individuales.
        grafica_bloques (str): Gráficas de bloques de imputación.
        grafica_fft (str): Gráficas de análisis en frecuencia (FFT).
        grafica_preanalisis (str): Gráficas previas al análisis.
        grafica_tramo (str): Gráficas de tramos específicos.
        default_res_dir (str): Ruta por defecto para guardar gráficas.
        parametros_globales (str): Carpeta con parámetros generales del modelo.
        log_csv (str): Log acumulativo de modelos ARMA generados.
        tiempos_csv (str): Log de tiempos de ejecución del proceso ARMA.
        log_total (str): Log acumulativo de toda la ejecución del proceso ARMA. 

    Parámetros:
        nombre_archivo (str): Identificador único del conjunto de datos para 
                          construir rutas específicas.
        hash_datos (str): Identificador único del conjunto de datos para 
                          construir rutas específicas.

        
    """
    def __init__(self,nombre_archivo: str,hash_datos: str):
        # Base general para este hash de datos
        self.base = os.path.join(RES_DIR, "ARMA", f"{hash_datos}_{nombre_archivo}")

        # Subcarpetas de salida de datos imputados
        self.imputados = os.path.join(self.base, "003_Datos_Imputados")

        # Carpeta general de gráficas
        self.graficas = os.path.join(self.base, "004_graficas")
        self.grafica_impt = os.path.join(self.graficas, "002_imputaciones")
        self.grafica_bloques = os.path.join(self.graficas, "003_imputacion_bloques")
        self.grafica_fft = os.path.join(self.graficas, "004_fft")
        self.grafica_preanalisis = os.path.join(self.graficas, "001_graficas_preanalisis")
        self.grafica_tramo = os.path.join(self.graficas, "001_grafica_tramo")

        # Por defecto, guardar las gráficas iniciales aquí
        self.default_res_dir = self.grafica_preanalisis

        # Parámetros globales de ejecución ARMA
        self.parametros_globales = os.path.join(self.base, "000_parametros_Arma")

        # Log de resultados ARMA (CSV acumulativo)
        self.log_csv = os.path.join(self.base, "001_log_ARMA", "arma_modelos_log.csv")

        # Log de tiempos de ejecución
        self.tiempos_csv = os.path.join(self.base, "002_tiempos_ARMA", "arma_tiempos_log.csv")

        # Log general por archivo/lote
        self.log_total = os.path.join(self.base, "005_log")

