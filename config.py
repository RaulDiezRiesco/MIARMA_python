"""
MÓDULO: config.py

Este módulo centraliza todos los parámetros de configuración del proyecto,
agrupando y documentando variables clave utilizadas en las distintas etapas
de preprocesamiento, análisis, modelado, imputación y visualización de series
temporales.

Su objetivo es facilitar el mantenimiento del código, garantizar consistencia
entre módulos y permitir un control flexible de opciones mediante edición
de un único archivo.

"""
# ========================================
# 📁 LIBRERÍAS
# ========================================
import os

# ========================================
# 📁 DIRECTORIOS
# ========================================

# 📁 Base path del proyecto
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# 📂 Carpetas principales
BASE_DIR = os.path.join(BASE_PATH, "001_datos")
LOG_DIR = os.path.join(BASE_PATH, "003_logs")
RES_DIR = os.path.join(BASE_PATH, "002_resultados")

# 📁 Salida general de modelo ARMA
SALIDA_DIR = os.path.join(RES_DIR, "ARMA")

# 📁 Subcarpetas organizadas
# --- Datos imputados como CSV u otros ---
SALIDA_DATOS_IMPUTADOS = os.path.join(SALIDA_DIR, "003_Datos_Imputados")

# --- Agrupación de todas las gráficas ---
SALIDA_GRAFICAS = os.path.join(SALIDA_DIR, "004_graficas")

# --- Subcarpetas de gráficas específicas ---
SALIDA_GRAFICA_IMPT = os.path.join(SALIDA_GRAFICAS, "002_imputaciones")
SALIDA_GRAFICA_BLOQUES = os.path.join(SALIDA_GRAFICAS, "003_imputacion_bloques")
SALIDA_GRAFICA_FFT = os.path.join(SALIDA_GRAFICAS, "004_fft")

# 📈 Gráficas del preanálisis
DEFAULT_RES_DIR = os.path.join(SALIDA_GRAFICAS, "000_graficas_preanalisis")
DEFAULT_RES_DIR_TRAMO = os.path.join(SALIDA_GRAFICAS, "001_grafica_tramo")

INCLUIR_LOG_EXTENDIDO_POST_MODELADO: bool = True

# ─────────────────────────────────────────────
# 📁 MÓDULO: FUNCIONES INICIALES
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# ⚙️ CONFIGURACIÓN GENERAL
# ─────────────────────────────────────────────

DEFAULT_SEPARATOR = ","            # Separador por defecto usado en los archivos CSV
CSV_EXTENSION = ".csv"             # Extensión de archivo CSV esperada             
TAMAÑO_NOMBRE_HASH: int = 8        # Longitud del hash visible (identificador corto)
MODO_AUTO = False                  # ¿Modo automático activado?
INDICE_AUTO = 0                    # Índice por defecto del modo automático para encontrar los datos

# ─────────────────────────────────────────────
# 🧠 RESPUESTAS VÁLIDAS PARA INTERACCIÓN
# ─────────────────────────────────────────────

RESPUESTAS_POSITIVAS = {"s", "si", "sí", "y", "yes"}   # Respuestas positivas aceptadas
RESPUESTAS_NEGATIVAS = {"n", "no"}                     # Respuestas negativas aceptadas

# ─────────────────────────────────────────────
# 📈 RESUMEN DE SERIES TEMPORALES
# ─────────────────────────────────────────────
RESUMEN_SERIE_DESCRIPCION = (
    "Resumen diagnóstico de una serie temporal: incluye información sobre tipo de dato, "
    "valores nulos, tipo de índice, presencia de strings en la columna y estadísticas básicas "
    "si la serie es numérica."
)
RESUMEN_SERIE_EXPLICACIONES = {
    "columna": "Nombre de la única columna analizada en el DataFrame.",
    "tipo_dato": "Tipo de dato de la columna (por ejemplo, float64, int64, object).",
    "valores_nulos": "Cantidad total de valores nulos (NaN) en la serie.",
    "porcentaje_nulos": "Porcentaje de valores nulos con respecto al total.",
    "indice_temporal": "Indica si el índice del DataFrame es de tipo datetime.",
    "tipo_indice": "Tipo del índice del DataFrame (por ejemplo, datetime64[ns], int64).",
    "contiene_strings": "Verdadero si hay al menos un string en la columna.",
    "resumen_estadistico_sin_NaN": "Diccionario con estadísticas básicas si los datos son numéricos.",
    "resumen_estadistico": "Texto explicativo si no se pueden calcular estadísticas numéricas.",
}

# ─────────────────────────────────────────────
# 📁 MÓDULO: FUNCIONES_PREANÁLISIS
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# 💬 MENSAJES DE INTERACCIÓN SÍ/NO
# ─────────────────────────────────────────────

MSG_RESPUESTA_INVALIDA_SN = "⚠️ Responde con 's' (sí) o 'n' (no)."                  # Mensaje cuando la respuesta no es válida en preguntas sí/no

# ─────────────────────────────────────────────
# 📊 PARÁMETROS PARA RESUMEN DE DISTRIBUCIÓN
# ─────────────────────────────────────────────

APLICAR_LOG_TRANSFORM = False                           # Si es True, se aplica una transformación logarítmica a la serie para la distribución
DISTRIBUCION_PLOT_TYPES = ["hist", "box", "violin"]     # Tipos de gráficos posibles para distribución
MAX_MUESTRA_DISTRIBUCION = 10000                        # Máximo de puntos para graficar. Si hay más del valor, se muestrea.
RANDOM_STATE_MUESTRA = 42                               # Semilla para reproducibilidad del muestreo
COLOR_HISTOGRAMA = "skyblue"                            # Color por defecto para el histograma
COLOR_BOXPLOT = "lightcoral"                            # Color por defecto para el boxplot
COLOR_VIOLINPLOT = "lightgreen"                         # Color por defecto para el violinplot
HIST_BINS = 30                                          # Número de bins (barras) en el histograma
ORIENTACION_GRAFICOS = "horizontal"                     # Orientación de los gráficos ('horizontal' o 'vertical')
DEFAULT_NOMBRE_DISTRIBUCION = "grafica_distribucion"    # Nombre por defecto para guardar la gráfica de distribución
UMBRAL_ASIMETRIA = 0.5                                  # Valor a partir del cual se considera que hay asimetría significativa
UMBRAL_CURTOSIS = 0.5                                   # Valor a partir del cual se considera que hay colas pesadas

# ─────────────────────────────────────────────
# 📏 PARÁMETROS PARA ÍNDICE INTERCUARTÍLICO (IQR)
# ─────────────────────────────────────────────

IQR_SEGMENTOS_AUTO = 4                                  # Número de segmentos automáticos para comparar IQR
UMBRAL_VARIACION_IQR = 0.30                             # Umbral de variación relativa entre IQRs para marcar alerta
MIN_DATOS_POR_SEGMENTO = 10                             # Mínimo de datos requeridos por segmento para calcular IQR
COLOR_IQR_ALERTA = "tomato"                             # Color de alerta cuando hay variaciones anómalas
COLOR_IQR_NORMAL = "skyblue"                            # Color normal para segmentos sin alerta
FIGSIZE_IQR = (12, 6)                                   # Tamaño por defecto del gráfico de IQR
MAX_LABELS_IQR = 10                                     # Máximo número de etiquetas en el eje para no saturar
IQR_ETIQUETAS_ROTACION = 45                             # Rotación de etiquetas en el gráfico
DEFAULT_NOMBRE_IQR = "grafica_iqr"                          # Nombre por defecto para guardar el gráfico de IQR

# ─────────────────────────────────────────────
# 📈 PARÁMETROS PARA VISUALIZACIÓN DE SERIE INICIAL
# ─────────────────────────────────────────────

PLOT_FIGSIZE = (20, 6)                                  # Tamaño por defecto de la figura para la serie
PLOT_COLOR = "steelblue"                                # Color principal de la línea
PLOT_LINEWIDTH = 1.5                                    # Grosor de la línea
PLOT_ALPHA = 0.75                                       # Transparencia de la línea
PLOT_GRID_STYLE = "--"                                  # Estilo de la grilla
PLOT_GRID_ALPHA = 0.3                                   # Transparencia de la grilla
PLOT_MAX_LABELS = 12                                    # Número máximo de etiquetas en eje para evitar saturación
COLOR_LINEA_SERIE_INICIAL = "#007acc"                 # Color específico para la línea de la serie inicial
TITULO_POR_DEFECTO_SERIE = "Visualización de la serie"  # Título por defecto del gráfico
DEFAULT_NOMBRE_SERIE_INICIAL = "grafica_completa"       # Nombre por defecto del archivo guardado

# ─────────────────────────────────────────────
# ❓ PARÁMETROS VARIOS PARA INTERACCIÓN
# ─────────────────────────────────────────────

PREGUNTA_ANALISIS_DISTRIBUCION = "¿Deseas realizar un análisis exploratorio de la distribución?"        # Pregunta si se desea análisis de distribución
PREGUNTA_MOSTRAR_GRAFICOS = "¿Quieres visualizar los gráficos en pantalla?"                             # Pregunta si se desea mostrar gráficos
PREGUNTA_GUARDAR_GRAFICOS = "¿Quieres guardar los gráficos en la carpeta de resultados?"                # Pregunta si se desea guardar gráficos
PREGUNTA_MOSTRAR_SERIE_COMPLETA = "¿Quieres visualizar la serie completa?"                              # Pregunta si se desea ver la serie completa
PREGUNTA_MOSTRAR_EN_PANTALLA = "¿Mostrar en pantalla?"                                                  # Pregunta corta de mostrar
PREGUNTA_GUARDAR_IMAGEN = "¿Guardar imagen en la carpeta de resultados?"                                # Pregunta corta de guardar imagen

# ─────────────────────────────────────────────
# 📁 MÓDULO: FUNCION_TRAMO_INTELIGENTE - CONFIG
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# 🔧 VARIABLES COMPARTIDAS (usadas en varias funciones)
# ─────────────────────────────────────────────

MIN_VARIANZA_ADMISIBLE: float = 1e-3              # Variancia mínima aceptada para considerar que hay variabilidad útil
MAX_NULOS_INTERPOLABLES: int = 5                  # Máximo número de NaNs permitidos antes de interpolar
METODO_INTERPOLACION_RELLENO: str = "linear"      # Método de interpolación por defecto para rellenar datos faltantes

# ─────────────────────────────────────────────
# 📈 calcular_score (estructura PACF ponderada)
# ─────────────────────────────────────────────

MIN_LARGO_SUBSERIE: int = 10                      # Longitud mínima de una subserie válida para cálculo de score
MAX_DELTA_MEAN: float = 0.2                       # Diferencia máxima tolerada de media entresubserie y la serie original
MAX_DELTA_VAR: float = 0.2                        # Diferencia máxima tolerada de varianza entresubserie y la serie original
NLAGS_SCORE_MIN: int = 3                          # Número mínimo de lags para score PACF
NLAGS_SCORE_MAX: int = 20                         # Número máximo de lags para score PACF
NLAGS_PACF_MINIMO_EFECTIVO: int = 2               # Límite mínimo de lags significativos en PACF para considerar válido
PACF_METHOD: str = "ywmle"                        # Método de cálculo para PACF (ej: 'ywunbiased', 'ywmle')

# ─────────────────────────────────────────────
# 🧠 seleccionar_tramo_inteligente_para_arma
# ─────────────────────────────────────────────

MIN_VENTANA: float = 0.10                         # Porcentaje mínimo del total que debe tener un tramo (ventana)
MAX_VENTANA: float = 0.40                         # Porcentaje máximo del total que puede tener un tramo (ventana)
PASO: float = 0.005                               # Desplazamiento (step) para avanzar entre posibles ventanas en porcentaje
STEP_VENTANA: float = 0.005                       # Reducción de la ventana

# ─────────────────────────────────────────────
# 📉 _buscar_por_tests
# ─────────────────────────────────────────────

PVALOR_ADF_UMBRAL: float = 0.05                   # Umbral máximo para aceptar estacionariedad en ADF
PVALOR_KPSS_UMBRAL: float = 0.05                  # Umbral máximo para aceptar estacionariedad en KPSS
CRITERIO_ESTRUCTO_TEST: bool = False              # Si es True, se deben de pasar ambos test

# 🧪 MOTIVOS DE DESCARTE → TESTS (ADF + KPSS)
DESCARTE_TEST_NAN = "TEST -> contiene NaNs"                                            # La subserie tiene datos faltantes
DESCARTE_TEST_VAR_CERO = "TEST -> varianza muy baja"                                   # Varianza cercana a cero
DESCARTE_TEST_PVALORES = "TEST -> p-valores no válidos"                                # Los p-valores no son interpretables
DESCARTE_TEST_SCORE_CORTA = "TEST -> score inválido: tramo demasiado corto"            # Tramo muy corto para score
DESCARTE_TEST_SCORE_ESTADISTICA = "TEST -> score inválido: sin estructura estadística" # Score sin estructura
DESCARTE_TEST_SCORE_PACF_INSUF = "TEST -> score inválido: PACF insuficiente"           # PACF no confiable
DESCARTE_TEST_SCORE_PACF_ERROR = "TEST -> error al calcular PACF"                      # Error durante PACF
DESCARTE_TEST_SCORE_DIVISION_CERO = "TEST -> score inválido: división por cero"        # División inválida
DESCARTE_TEST_SCORE_DESCONOCIDO = "TEST -> motivo desconocido"                         # Descarte sin clasificación

# ─────────────────────────────────────────────
# 🔎 _buscar_por_score
# ─────────────────────────────────────────────

DESCARTE_SCORE_NAN: str = "SCORE -> contiene NaNs"                                # La subserie tiene NaNs
DESCARTE_SCORE_VAR_CERO: str = "SCORE -> varianza cercana a cero"                 # Varianza insuficiente
DESCARTE_SCORE_CORTA: str = "SCORE -> subserie demasiado corta"                   # Subserie muy corta
DESCARTE_SCORE_ESTADISTICA: str = "SCORE -> diferencias estadísticas altas"       # Score rechaza por diferencias grandes
DESCARTE_SCORE_PACF_INSUF: str = "SCORE -> pocos lags para PACF"                  # Pocos lags significativos
DESCARTE_SCORE_PACF_ERROR: str = "SCORE -> error al calcular PACF"                # Error técnico en PACF
DESCARTE_SCORE_DIVISION_CERO: str = "SCORE -> división inválida (denominador 0)"  # División no permitida
DESCARTE_SCORE_DESCONOCIDO: str = "SCORE -> motivo desconocido"                   # Descarte genérico sin clasificación

# ─────────────────────────────────────────────
# 🆘 _buscar_fallback
# ─────────────────────────────────────────────

DESCARTE_FALLBACK_NAN: str = "FALLBACK -> contiene NaNs"                # Fallback fallido por NaNs
DESCARTE_FALLBACK_VAR_CERO: str = "FALLBACK -> varianza cercana a cero" # Fallback con baja variabilidad
DESCARTE_FALLBACK_DIVISION_CERO: str = "FALLBACK -> división inválida"  # División no válida en fallback

# ─────────────────────────────────────────────
# 🧪 adf_seguro / kpss_seguro (tests robustos)
# ─────────────────────────────────────────────

AUTO_LAG_ADF_DEFAULT: str = "AIC"               # Criterio para seleccionar número óptimo de lags en ADF
FALLBACK_PVALUE_ADF: float = 1.0                # p-valor por defecto si ADF falla
NLAGS_KPSS_DEFAULT: str = "auto"                # Lags automáticos en KPSS
FALLBACK_PVALUE_KPSS: float = 0.00000000001     # p-valor por defecto si KPSS falla

# ─────────────────────────────────────────────
# 📦 mostrar_resultados_modelado (visual + log)
# ─────────────────────────────────────────────

INCLUIR_LOG_EXTENDIDO_TRAMO_MODELOS: bool = False                                 # Si es True, incluye detalles extendidos en log
NOMBRE_LOG_TRAMO_DEFAULT: str = "tramo"                                           # Nombre por defecto para el log de tramos
TITULO_GRAFICO_TRAMO: str = "📊 Tramo seleccionado (índices {inicio} - {fin})"    # Título del gráfico del tramo
NOMBRE_ARCHIVO_TRAMO: str = "{nombre_base}_tramo_{inicio}_{fin}"                  # Formato para guardar el gráfico del tramo


# ─────────────────────────────────────────────
# 📁 MÓDULO: BUSQUEDA_MODELOS_ARMA - CONFIG
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# 🔧 VARIABLES COMPARTIDAS (usadas en varias funciones)
# ─────────────────────────────────────────────

PARAMSDIR = os.path.join(RES_DIR, "ARMA", "000_parametros_Arma")     # Carpeta donde se guardan los parámetros de los modelos
USO_PARAMETROS_INICIALES_ARMA:bool = True                            # Estima si se utilizan los parámetros iniciales de modelos ARMA vecinos ya calculados en la fase inicial. 
DATASET_NAME: str = "Serie"                                          # Nombre por defecto del dataset (para identificación)
COV_TYPE: str = "opg"                                                # Tipo de estimador de la matriz de covarianza ("opg", "oim", "robust", etc.)

# ─────────────────────────────────────────────
# 📐 generate_orders (generación de órdenes)
# ─────────────────────────────────────────────

ORDEN_MIN: int = 1                               # Orden mínimo del modelo ARMA (p + q >= ORDEN_MIN) 
ORDEN_MAX: int = 30                              # Orden máximo del modelo ARMA (p + q <= ORDEN_MAX)

# ─────────────────────────────────────────────
# 🧠 run_arma_grid_search (búsqueda de modelos)
# ─────────────────────────────────────────────

NJOBS: int = -1                                                                         # Número de procesos en paralelo (-1 usa todos los núcleos disponibles)
TOP_N: int = 40                                                                         # Número de mejores modelos a conservar tras búsqueda
N_GRUPOS: int = 10                                                                      # Número de grupos en los que se divide el rango del ratio estructural p / (p + q), para asegurar una selección equitativa de modelos AR, MA y mixtos en la fase de refinamiento.
METRIC: str = "aic"                                                                     # Métrica de evaluación (puede ser "bic", "aic")
LOG_CSV = os.path.join(RES_DIR, "ARMA", "001_log_ARMA", "arma_modelos_log.csv")         # Ruta del archivo log de resultados
TIMING_LOG = os.path.join(RES_DIR, "ARMA", "002_tiempos_ARMA", "arma_tiempos_log.csv")  # Log de tiempos de entrenamiento

# ─────────────────────────────────────────────
# ⚙️ fit_model (entrenamiento del modelo)
# ─────────────────────────────────────────────

MAX_ITER: int = 10                                   # Número de iteraciones para el primer ajuste rápido
MAX_ITER_FINAL: int = 200                            # Número de iteraciones para el ajuste final (más preciso)
TREND: str = "n"                                     # Tipo de tendencia ('n': ninguna, 'c': constante, 't': lineal, 'ct': ambas)
FORZAR_ESTACIONARIA: bool = False                    # Forzar que el modelo sea estacionario durante el ajuste
FORZAR_INVERTIBILIDAD: bool = False                  # Forzar que el modelo sea invertible durante el ajuste

# ─────────────────────────────────────────────
# ♻️ get_start_params (warm start con vecinos)
# ─────────────────────────────────────────────

UMBRAL_STARTPARAMS: int = 20                         # Límite mínimo de complejidad (p+q) para usar vecinos como base.

# ─────────────────────────────────────────────
# 🔍 buscar_vecinos_eficientes (vecinos previos)
# ─────────────────────────────────────────────

CRITERIO_AIC_BIC_VECINOS: str = "aic"               # Criterio usado para buscar vecinos más eficientes ("aic" o "bic")
MAX_DIST_VECINOS: int = 5                           # Distancia máxima entre órdenes (p, q) para considerar que dos modelos son vecinos


# ─────────────────────────────────────────────
# 📁 MÓDULO: FUNCION_VALIDACION_MODELOS - CONFIG
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# 🔧 VARIABLES COMPARTIDAS (usadas en varias funciones)
# ─────────────────────────────────────────────

CRITERIO_SELECCION_MODELO: str = "aic"               # Criterio para elegir el mejor modelo ("aic", "bic", etc.)

USAR_TESTS_NORMALIDAD: bool = True                   # Ejecutar tests de normalidad sobre los residuos
USAR_TESTS_AUTOCORRELACION: bool = True              # Ejecutar tests de autocorrelación sobre los residuos
USAR_TESTS_HETEROCEDASTICIDAD: bool = True           # Ejecutar tests de heterocedasticidad sobre los residuos

# ─────────────────────────────────────────────
# 🧪 TESTS DE NORMALIDAD (residuos)
# ─────────────────────────────────────────────

ALPHA_TESTS_NORMALIDAD: float = 0.05                 # Nivel de significancia para los tests de normalidad
FALLOS_MAXIMOS_NORMALIDAD: int = 1                   # Número máximo de fallos aceptados en los tests de normalidad
LIMITE_SHAPIRO: int = 500                            # Máximo número de muestras para aplicar test de Shapiro (más lento en grandes muestras)

# ─────────────────────────────────────────────
# 🔁 TEST DE AUTOCORRELACIÓN (residuos)
# ─────────────────────────────────────────────

ALPHA_TESTS_AUTOCORR: float = 0.05                   # Nivel de significancia para test de autocorrelación (Ljung-Box)
MIN_LAGS_LB: int = 5                                 # Mínimo número de lags a usar en test Ljung-Box
MAX_LAGS_LB: int = 30                                # Máximo número de lags a usar en test Ljung-Box

# ─────────────────────────────────────────────
# 📉 TEST DE HETEROCEDASTICIDAD (residuos)
# ─────────────────────────────────────────────

ALPHA_TESTS_HETEROCED: float = 0.05                  # Nivel de significancia para test de heterocedasticidad (ej. ARCH)

# ─────────────────────────────────────────────
# 📁 MÓDULO: IMPUTACION_ARIMA - CONFIG
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# 🔧 VARIABLES COMPARTIDAS (usadas en varias funciones)
# ─────────────────────────────────────────────

TAMAÑO_LOTE_BLOQUES_A_ANALIZAR: int = 20              # Número de bloques de nulos a analizar en paralelo, de acuerdo con tu número de núcleos y tus capacidades de cómputo.
STEP_CONTEXTO_IMPUTACION: int = 25                    # Paso (step) por defecto para expandir el contexto alrededor del bloque a imputar
USAR_PARAMETROS_INICIALES_IMPUTACION: bool = False    # Si es True, reutiliza parámetros iniciales conocidos para imputar
FORZAR_ESTACIONARIDAD_IMPUTACION: bool = False        # Si es True, impone que el modelo usado en la imputación sea estacionario
FORZAR_INVERTIBILIDAD_IMPUTACION: bool = False        # Si es True, fuerza que el modelo sea invertible durante la imputación

# ─────────────────────────────────────────────
# 📦 IMPUTACIÓN DE BLOQUES (bloque_arima_simple)
# ─────────────────────────────────────────────

MARGEN_SEGURIDAD_IMPUTACION: int = 25                 # Número adicional de observaciones antes y después del bloque para seguridad
MAX_CONTEXTO_IMPUTACION: int = 300                    # Tamaño máximo del contexto usado para imputar un bloque
FACTOR_LARGO_IMPUTACION: int = 2                      # Multiplicador para estimar el contexto mínimo respecto al largo del bloque a imputar
MODO_PESOS_IMPUTACION: str = "lineal"                 # Modo de ponderación entre valores contextuales ('lineal', 'uniforme', etc.)

# ─────────────────────────────────────────────
# 🔁 IMPUTACIÓN ITERATIVA (con ciclo)
# ─────────────────────────────────────────────

MIN_BLOQUE_IMPUTACION: int = 1                        # Tamaño mínimo de un bloque de NaNs para considerar imputación iterativa

# ─────────────────────────────────────────────
# 📁 MÓDULO: GRAFICAS
# ─────────────────────────────────────────────

USAR_INDICE_ORIGINAL: bool = True

# ─────────────────────────────────────────────
# 📈 CONFIGURACIÓN PARA FUNCIONES DE IMPUTACIÓN Y GRAFICADO
# ─────────────────────────────────────────────

MOSTRAR_GRAFICOS_IMPUTACION: bool = False         # Mostrar gráfico completo de la serie imputada
GUARDAR_CSV_SERIE_IMPUTADA: bool = True           # Guardar serie imputada en archivo CSV

CONTEXT_WINDOW_BLOQUES_IMPUTADOS: int = 10        # Número de datos de contexto a mostrar antes/después de cada bloque imputado
MOSTRAR_GRAFICOS_BLOQUES: bool = False            # Mostrar gráficos individuales por cada bloque imputado

NOMBRE_GRAFICA_BLOQUES = "bloque"                 # Nombre base para los archivos de gráficos por bloque
NOMBRE_FFT = "fft"                                 # Nombre base para el gráfico de análisis FFT
NOMBRE_BASE_IMPUTACIONES = "imputacion"           # Prefijo común para todos los resultados de imputación

# ─────────────────────────────────────────────
# ⚡️ CONFIGURACIÓN PARA ANÁLISIS FFT
# ─────────────────────────────────────────────

GUARDAR_GRAFICAS_FFT: bool = True                 # Guardar imagen del análisis FFT comparativo
UMBRAL_FFT_DOMINANTES: float = 0.75               # Umbral acumulado de energía para identificar frecuencias dominantes
MOSTRAR_GRAFICAS_FFT: bool = False                # Mostrar gráfico de análisis FFT en pantalla
MAX_FREQ_VISUALIZACION_FFT: float = 5.0           # Frecuencia máxima a mostrar en el eje X del gráfico de FFT
DELTA_T_FFT: float = 1.0                          # Intervalo de muestreo temporal entre observaciones (frecuencia = 1 / Δt)