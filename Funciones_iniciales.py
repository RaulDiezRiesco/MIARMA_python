"""
MÓDULO: Funciones_iniciales.py

Este módulo proporciona una interfaz estructurada y flexible para cargar archivos CSV de forma interactiva o automática. 

FUNCIONALIDADES CLAVE:

1. Exploración de directorios con archivos CSV:
   - Detección y listado de archivos con extensión `.csv`. Modificable para poder detectar otras extensiones si se desea
   - Formato de presentación legible y numerado para la selección manual.

2. **Carga y validación de datos desde CSV**:
   - Detección automática (o manual) de encabezados.
   - Selección de separador de columnas.
   - Cálculo de un hash único por archivo para control de versiones o integridad.

3. **Selección de columna principal de análisis**:
   - Por nombre o índice.
   - Compatibilidad con archivos con o sin encabezados.

4. **Gestión de índice temporal**:
   - Opción de establecer una columna como índice temporal (datetime) o numérico. 
   - Reset automático del índice si no se establece.

5. **Resumen diagnóstico inicial**:
   - Información general sobre la columna principal (tipo, nulos, strings), con estadísticas básicas y validaciónes.

MODOS DE USO:

- **Interactivo**: el usuario selecciona manualmente los parámetros.
- **Automático**: el usuario no interactua, se elige AUTOMÁTICAMENTE la colúmna 0 como datos, y la 1 (si existe), como índice temporal. Si solamente existe 
una columna, se establecerá un índice numérico. 

RETORNO FINAL:
   - DataFrame procesado para procesarse en el siguiente módulo.
   - Diccionario con resumen estructural.
   - Hash único del archivo leído.

"""

# =============================================================
# 🧱 1. LIBRERÍAS ESTÁNDAR
# =============================================================
import os
import hashlib

# =============================================================
# 📦 2. LIBRERÍAS DE TERCEROS
# =============================================================

import pandas as pd
from typing import Callable, Optional, Tuple, Union, List
from scipy.stats import skew, kurtosis


# =============================================================
# ⚙️ 3. CONFIGURACIÓN GLOBAL (config.py)
# =============================================================

from config import (
    # Configuración general
    BASE_DIR,
    DEFAULT_SEPARATOR,
    TAMAÑO_NOMBRE_HASH,

    # Modo automático / índice por defecto
    MODO_AUTO,
    INDICE_AUTO,

    # Diagnóstico de resumen
    RESUMEN_SERIE_DESCRIPCION,
    RESUMEN_SERIE_EXPLICACIONES,

    RESPUESTAS_POSITIVAS,
    RESPUESTAS_NEGATIVAS)


def hay_archivos_csv(
    base_dir: str = BASE_DIR,
    log_msg: Optional[Callable[[str], None]] = None,
) -> bool:
    """
    Verifica si existen archivos con la extensión CSV en el directorio especificado.

    Parámetros:
        base_dir (str): Ruta del directorio a inspeccionar.
        log_msg (Callable, opcional): Función para registrar mensajes.

    Retorna:
        bool:
            - True → Si hay al menos un archivo con la extensión CSV.
            - False → Si no se encuentran archivos o si ocurre un error de acceso.
    """

    try:
        # Obtener la lista de archivos en el directorio y verificar si hay al menos un .csv
        archivos = os.listdir(base_dir)
        hay_csv = any(f.lower().endswith(".csv") for f in archivos)
        log_msg(f"📁 Archivos encontrados en '{base_dir}': {len(archivos)}. ¿Hay CSV?: {hay_csv}")

        return hay_csv

    except (FileNotFoundError, PermissionError) as e:
        # Captura errores de acceso al directorio y los reporta
        log_msg(f"❌ No se pudo acceder al directorio '{base_dir}': {e}")
        return False

def pedir_nombre_archivo(
    input_func: Callable[[str], str] = input,
    modo_auto: bool = MODO_AUTO,
    log_msg: Optional[Callable[[str], None]] = None,
) -> Optional[str]:
    """
    Solicita al usuario el nombre de un archivo CSV (sin incluir la extensión).

    Comportamiento:
        - Si `modo_auto` es True, no se solicita nada y se retorna `None`.
        - Si el usuario ingresa un nombre vacío, también se retorna `None`.
        - El nombre se limpia de espacios al inicio y final.

    Parámetros:
        input_func (Callable): Función utilizada para capturar entrada del usuario.
        modo_auto (bool): Si es True, se omite la interacción y se retorna None.
        log_msg (Callable, opcional): Función para registrar logs. 

    Retorna:
        Optional[str]: Nombre del archivo sin la extensión, o None si está vacío
                       o si se ejecuta en modo automático.
    """

    # Si está activado el modo automático, se omite la solicitud manual
    if modo_auto:
        log_msg(f"[modo_auto] Se omite la solicitud del nombre del archivo. → None")
        return None

    # Solicita el nombre del archivo al usuario y limpia espacios en blanco
    nombre = input_func("Nombre del archivo (sin '.csv', Enter para listar): ").strip()
    resultado = nombre or None

    # Registra el nombre ingresado (o si fue vacío)
    log_msg(f"Nombre de archivo recibido: {resultado if resultado else '[vacío] → None'}")
    return resultado

def pedir_separador(
    input_func: Callable[[str], str] = input,
    modo_auto: bool = MODO_AUTO,
    log_msg: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Solicita al usuario el separador de columnas para leer el archivo CSV.

    Comportamiento:
        - En modo automático, se devuelve directamente el separador por defecto (`DEFAULT_SEPARATOR`).
        - En modo interactivo, se solicita al usuario un separador. Si no se ingresa nada, se usa el valor por defecto ( , ).

    Parámetros:
        input_func (Callable): Función para capturar la entrada del usuario.
        modo_auto (bool): Si es True, se omite la solicitud manual.
        log_msg (Callable, opcional): Función para registrar mensajes.

    Retorna:
        str: Separador ingresado por el usuario, o el separador por defecto si está vacío o en modo automático.
    """
    # Si está activado el modo automático, se omite la solicitud y se usa el separador por defecto
    if modo_auto:
        log_msg(f"[modo_auto] Usando separador por defecto: '{DEFAULT_SEPARATOR}'")
        return DEFAULT_SEPARATOR

    # Solicita el separador al usuario y limpia espacios en blanco
    sep = input_func(f"Separador (Enter para usar ','): " ).strip()

    # Si el usuario no ingresó nada, se usa el separador por defecto
    if sep:
        resultado = sep
    else:
        resultado = DEFAULT_SEPARATOR

    # 📝 Log del separador ingresado o asumido
    log_msg(f"Separador recibido: '{resultado}'")
    return resultado

def listar_archivos_csv(
    base_dir: str = BASE_DIR,
    log_msg: Optional[Callable[[str], None]] = None) -> List[str]:
    """
    Devuelve una lista ordenada alfabéticamente con los archivos CSV del directorio especificado.

    Parámetros:
        base_dir (str): Ruta del directorio a inspeccionar.
        log_msg (Callable, opcional): Función para registrar mensajes de log.

    Retorna:
        List[str]:
            - Lista de nombres de archivos que terminan con la extensión CSV.
            - Lista vacía si el directorio no existe, no es accesible o no contiene archivos válidos.


    """
    try:
        # Listar y filtrar archivos que terminan con la extensión CSV
        archivos = [
            f for f in os.listdir(base_dir)
            if f.lower().endswith(".csv") and os.path.isfile(os.path.join(base_dir, f))
        ]
        # Ordenar la lista alfabéticamente
        archivos_ordenados = sorted(archivos)
        # Log del total de archivos encontrados
        log_msg(f"{len(archivos_ordenados)} archivos CSV encontrados en '{base_dir}'.")

        return archivos_ordenados

    except (FileNotFoundError, PermissionError) as e:
        # Error al acceder al directorio → log y lista vacía
        log_msg(f"❌ No se pudo acceder al directorio '{base_dir}': {e}")
        return []

def formatear_lista_archivos(
    archivos: List[str],
    base_dir: str = BASE_DIR) -> str:
    """
    Genera un mensaje enumerado para mostrar la lista de archivos CSV encontrados en un directorio.

    Parámetros:
        archivos (List[str]): Lista de nombres de archivos CSV.
        base_dir (str): Ruta del directorio donde están los archivos.

    Retorna:
        str:
            - Texto formateado con los archivos numerados (ej. "[0] archivo.csv").
            - Si no hay archivos, devuelve un mensaje indicando que el directorio está vacío.
    """

    # Si no hay archivos, se devuelve un mensaje indicando que el directorio está vacío
    if not archivos:
        return f"No se encontraron archivos en '{base_dir}'"

    # Construcción del mensaje enumerado con los archivos disponibles
    mensaje = f"\n Archivos disponibles en '{base_dir}'\n"
    mensaje += "\n".join(f"  [{i}] {nombre}" for i, nombre in enumerate(archivos))
    return mensaje

def seleccionar_archivo_desde_lista(
    archivos: List[str],
    base_dir: str = BASE_DIR,
    input_func: Callable[[str], str] = input,
    log_msg: Optional[Callable[[str], None]] = None,
    modo_auto: bool = MODO_AUTO,
    indice_auto: int = INDICE_AUTO
) -> Optional[str]:
    """
    Permite seleccionar un archivo desde una lista, ya sea por índice o por nombre (sin extensión).

    Comportamiento:
        - En modo automático, selecciona directamente por índice (`indice_auto`) sin interacción.
        - En modo interactivo, el usuario puede elegir por índice o nombre.

    Parámetros:
        archivos (List[str]): Lista de nombres de archivos disponibles.
        base_dir (str): Ruta del directorio donde están los archivos.
        input_func (Callable): Función para capturar la entrada del usuario.
        log_msg (Callable, opcional): Función para registrar mensajes.
        modo_auto (bool): Si es True, se selecciona automáticamente sin preguntar.
        indice_auto (int): Índice usado en modo automático.

    Retorna:
        Optional[str]: Ruta absoluta del archivo seleccionado, o None si la selección es inválida.
    """
    # MODO AUTOMÁTICO: Selección directa por índice sin intervención del usuario
    if modo_auto:
        if not archivos:
            log_msg("⚠️ [modo_auto] No hay archivos disponibles para seleccionar.")
            return None
        if not (0 <= indice_auto < len(archivos)):
            log_msg(f"⚠️ [modo_auto] Índice fuera de rango: {indice_auto}.")
            return None

        archivo = archivos[indice_auto]
        ruta = os.path.join(base_dir, archivo)
        log_msg(f"[modo_auto] Archivo seleccionado automáticamente: [{indice_auto}] {archivo}")
        return ruta

    # MODO INTERACTIVO: El usuario debe elegir por índice o nombre
    seleccion = input_func(f"Selecciona el número o escribe el nombre del archivo (sin '.csv'): ").strip()

    # Opción 1: Selección por índice numérico
    if seleccion.isdigit():
        idx = int(seleccion)
        if 0 <= idx < len(archivos):
            archivo = archivos[idx]
            ruta = os.path.join(base_dir, archivo)
            log_msg(f"Selección por índice válida: [{idx}] {archivo}")
            return ruta
        else:
            log_msg(f"❌ Índice fuera de rango." )
            return None

    # Opción 2: Selección por nombre (sin extensión)
    archivo_completo = seleccion + ".csv"
    archivos_lower = [f.lower() for f in archivos]

    if archivo_completo.lower() in archivos_lower:
        index = archivos_lower.index(archivo_completo.lower())
        archivo = archivos[index]
        ruta = os.path.join(base_dir, archivo)
        log_msg(f"Selección por nombre válida: {archivo_completo}")
        return ruta

    # Si ninguna opción fue válida, mostrar mensaje de error
    log_msg("❌ Archivo no encontrado.")
    return None

def seleccionar_archivo_csv(
    archivo: Optional[str],
    base_dir: str = BASE_DIR,
    input_func: Callable[[str], str] = input,
    log_msg: Optional[Callable[[str], None]] = None,
    modo_auto: bool = MODO_AUTO,
    indice_auto: int = INDICE_AUTO,
) -> Optional[str]:
    """
    Permite seleccionar un archivo CSV dentro del directorio especificado.

    Comportamiento:
        - Si se proporciona un nombre de archivo (sin extensión), se verifica directamente su existencia.
        - Si no se proporciona, se listan los archivos disponibles y se selecciona uno.
        - En modo automático, se selecciona por índice (`indice_auto`) sin interacción.

    Parámetros:
        archivo (Optional[str]): Nombre del archivo (sin extensión). Si es None, se activa selección manual o automática.
        base_dir (str): Directorio donde buscar los archivos.
        input_func (Callable): Función utilizada para capturar entrada del usuario.
        log_msg (Callable, opcional): Función para registrar mensajes.
        modo_auto (bool): Si es True, se evita interacción y se selecciona automáticamente.
        indice_auto (int): Índice del archivo a usar en modo automático.

    Retorna:
        Optional[str]: Ruta absoluta del archivo seleccionado, o None si no se encuentra o falla la selección.
    """

    # Caso 1: Se proporciona un nombre de archivo directamente (sin extensión)
    if archivo:
        archivo_completo = archivo + ".csv" 
        ruta = os.path.join(base_dir, archivo_completo)

        # Verifica que el archivo exista físicamente
        if not os.path.isfile(ruta):
            log_msg(f"❌ El archivo '{archivo}' no existe en '{base_dir}'.")
            return None

        # Archivo válido y existente
        log_msg(f"✅ Archivo detectado directamente: {archivo_completo}")
        return ruta

    # 📌 Caso 2: No se especificó archivo → Se listan los archivos disponibles
    archivos = listar_archivos_csv(base_dir, log_msg=log_msg)
    log_msg(formatear_lista_archivos(archivos, base_dir))

    # 📥 Delegar la selección a función externa (interactivo o automático)
    return seleccionar_archivo_desde_lista(
        archivos=archivos,
        base_dir=base_dir,
        input_func=input_func,
        log_msg=log_msg,
        modo_auto=modo_auto,
        indice_auto=indice_auto
    )

def calcular_hash_archivo(
    ruta: str,
    bytes_leer: int = -1,
    tamaño_nombre: int = TAMAÑO_NOMBRE_HASH
) -> str:
    """
    Calcula un hash SHA-256 a partir del contenido binario de un archivo.

    El hash puede truncarse para usar como identificador corto en logs, nombres de archivos, etc.

    Parámetros:
        ruta (str): Ruta absoluta del archivo.
        bytes_leer (int): 
            - -1 (por defecto): lee el archivo completo.
            - >0: lee solo los primeros `bytes_leer` bytes.
        tamaño_nombre (int): Cantidad de caracteres del hash a devolver (por defecto, 12).

    Retorna:
        str: Substring del hash SHA-256 del archivo, con la longitud indicada.
    """
    # Crear el objeto hash SHA-256
    sha = hashlib.sha256()

    try:
        # Abrir el archivo en modo binario y leer contenido
        with open(ruta, "rb") as f:
            contenido = f.read() if bytes_leer == -1 else f.read(bytes_leer)
            sha.update(contenido)

    except Exception as e:
        # Si falla la lectura, se lanza un error informativo
        raise IOError(f"❌ Error al leer el archivo para calcular hash: {e}")

    # Devolver el hash truncado a la longitud deseada
    return sha.hexdigest()[:tamaño_nombre]

def leer_csv(
    ruta: str,
    separador: str = DEFAULT_SEPARATOR,
    tiene_encabezado: Optional[bool] = None,
    log_msg: Optional[Callable[[str], None]] = None,
    modo_auto: bool = MODO_AUTO
) -> Tuple[pd.DataFrame, bool, str, str]:
    """
    Lee un archivo CSV desde una ruta dada, permitiendo especificar o detectar automáticamente si el archivo tiene encabezado
    revisando la primera columna, la cual no debe de ser numérica.
    También registra el proceso y calcula un hash único del archivo.

    Parámetros:
        ruta (str): Ruta absoluta del archivo CSV.
        separador (str): Separador de columnas usado en el archivo.
        tiene_encabezado (Optional[bool]): 
            - True: fuerza el uso de encabezado.
            - False: fuerza que no hay encabezado.
            - None: intenta detectarlo automáticamente, salvo en modo automático.
        log_msg (Callable, opcional): Función para registrar mensajes de log.
        modo_auto (bool): Si es True, se asume automáticamente que el archivo tiene encabezado.

    Retorna:
        Tuple[
            pd.DataFrame,  # DataFrame leído desde el archivo
            bool,          # Indicador de si se usó encabezado
            str,           # Hash SHA-256 del archivo
            str            # Nombre del archivo
        ]
    """

    # Verificar que el archivo exista físicamente
    if not os.path.isfile(ruta):
        raise FileNotFoundError(f"❌ Archivo no encontrado: {ruta}")

    # Calcular hash del archivo para control de versiones o integridad
    hash_archivo = calcular_hash_archivo(ruta)
    log_msg(f"🔑 Hash del archivo (SHA-256 recortado): {hash_archivo}")
    log_msg(f"📄 Leyendo archivo: {ruta}")
    log_msg(f"Separador: '{separador}'")
    nombre_archivo = os.path.basename(ruta)
    log_msg(f"📁 Nombre del archivo: {nombre_archivo}")
    # Detección de encabezado
    try:
        if tiene_encabezado is None:
            if modo_auto:
                # En modo automático se asume que hay encabezado
                tiene_encabezado = True
                log_msg("⚙️ [modo_auto] Se asume que el archivo tiene encabezado.")
            else:
                # En modo manual, se intenta detectar mirando la primera fila
                try:
                    df_test = pd.read_csv(ruta, sep=separador, nrows=1)
                    tiene_encabezado = not df_test.columns[0].isdigit()
                    log_msg("🔍 Encabezado detectado automáticamente.")
                except Exception:
                    tiene_encabezado = False
                    log_msg("⚠️ No se pudo detectar el encabezado automáticamente. Asumido como ausente.")
        # Leer el archivo completo con o sin encabezado según detección o forzado
        header = 0 if tiene_encabezado else None
        log_msg(f"📌 Leyendo con header = {header}")
        df = pd.read_csv(ruta, sep=separador, header=header)

    except Exception as e:
        # Cualquier error de lectura se encapsula y se lanza como ValueError
        raise ValueError(f"❌ Error al leer el archivo: {e}")

    return df, tiene_encabezado, hash_archivo, nombre_archivo

def _seleccionar_columna_datos(
    df: pd.DataFrame,
    tiene_encabezado: bool,
    input_func: Callable[[str], str],
    log_msg: Optional[Callable[[str], None]] = None,
    modo_auto: bool = MODO_AUTO
) -> Union[str, int]:
    """
    Permite seleccionar una columna principal desde el DataFrame, ya sea por nombre o por índice.

    Comportamiento:
        - Si `tiene_encabezado` es True, se puede seleccionar por nombre o por índice numérico.
        - Si es False, las columnas deben seleccionarse por índice (0, 1, 2...).
        - En modo automático, se selecciona la primera columna sin interacción.

    Parámetros:
        df (pd.DataFrame): DataFrame cargado.
        tiene_encabezado (bool): Indica si las columnas tienen nombres (True) o son numéricas (False).
        input_func (Callable): Función para capturar entrada del usuario.
        log_msg (Callable, opcional): Función para registrar mensajes.
        modo_auto (bool): Si es True, se selecciona automáticamente la primera columna.

    Retorna:
        Union[str, int]: Nombre o índice entero de la columna seleccionada.

    """
    # MODO AUTOMÁTICO: se elige la primera columna
    if modo_auto:
        columna = df.columns[0] if tiene_encabezado else 0
        log_msg(f"🤖 [modo_auto] Columna seleccionada automáticamente → {columna}")
        return columna

    # MODO INTERACTIVO CON ENCABEZADO: puede elegir por nombre o índice
    if tiene_encabezado:
        log_msg("📋 Columnas disponibles:")
        for i, c in enumerate(df.columns):
            log_msg(f"  [{i}] {c}")

        seleccion = input_func("Introduce el nombre o índice de la columna a usar como datos: ").strip()

        try:
            columna = df.columns[int(seleccion)] if seleccion.isdigit() else seleccion
        except Exception:
            log_msg("❌ Entrada inválida para selección de columna.")
            raise ValueError("❌ Entrada inválida para selección de columna.")

        if columna not in df.columns:
            log_msg(f"❌ La columna '{columna}' no existe.")
            raise ValueError(f"❌ La columna '{columna}' no existe.")

        return columna

    else:
        log_msg("\n📋 Columnas disponibles (índice + vista previa):")
        log_msg("-" * 50)
        for i in range(df.shape[1]):
            muestra = df.iloc[:5, i].tolist()
            log_msg(f"  [{i}] → {muestra}")
        log_msg("-" * 50)

        seleccion = input_func("Introduce el índice de la columna a usar como datos: ").strip()
        try:
            columna = int(seleccion)
            if not 0 <= columna < df.shape[1]:
                raise ValueError
            return columna
        except ValueError:
            log_msg("❌ Índice inválido para selección de columna.")
            raise ValueError("❌ Índice inválido para selección de columna.")

def _gestionar_indice_temporal(
    df: pd.DataFrame,
    input_func: Callable[[str], str],
    log_msg: Optional[Callable[[str], None]] = None,
    modo_auto: bool = MODO_AUTO,
    respuestas_positivas: set = RESPUESTAS_POSITIVAS,
    respuestas_negativas: set = RESPUESTAS_NEGATIVAS
) -> Optional[Union[str, int]]:
    """
    Permite establecer una columna del DataFrame como índice temporal (eje de tiempo).

    Comportamiento:
        - Si `modo_auto` está activo:
            - Si hay más de una columna, se usará la segunda (índice 1) como índice temporal.
              Se intentará convertirla a datetime automáticamente.
            - Si hay solo una columna, se reseteará el índice (índice numérico por defecto).
        - En modo interactivo:
            - Se pregunta si se desea establecer un índice temporal.
            - Si sí, se solicita el nombre o índice de columna.
            - Si no, se mantiene o resetea el índice.

    Parámetros:
        df (pd.DataFrame): DataFrame cargado del cual seleccionar el índice.
        input_func (Callable): Función para capturar entrada del usuario.
        log_msg (Callable, opcional): Función para registrar mensajes de log.
        modo_auto (bool): Si es True, se usa lógica automática sin interacción.

    Retorna:
        Optional[Union[str, int]]:
            - Nombre o índice de la columna usada como índice temporal.
            - None si no se estableció ningún índice temporal.
    """

    # === MODO AUTOMÁTICO ===
    if modo_auto:
        if df.shape[1] > 1:
            # Si hay más de una columna, usar la segunda como índice
            columna_indice = df.columns[1]
            try:
                df[columna_indice] = pd.to_datetime(df[columna_indice], errors="raise")
                log_msg(f"📅 [modo_auto] Columna '{columna_indice}' convertida a datetime.")
            except Exception:
                log_msg(f"⚠️ [modo_auto] No se pudo convertir '{columna_indice}' a datetime.")

            df.set_index(columna_indice, inplace=True)
            log_msg(f"✅ [modo_auto] '{columna_indice}' se usó como índice temporal.")
            return columna_indice
        else:
            # Si solo hay una columna, resetear el índice
            df.reset_index(drop=True, inplace=True)
            log_msg("⏩ [modo_auto] Solo hay una columna. Se usará índice numérico.")
            return None

    # === MODO INTERACTIVO ===
    respuesta = input_func("¿Deseas usar una columna como índice temporal? (s/N): ").strip().lower()

    if respuesta in respuestas_positivas:
        seleccion_indice = input_func("Introduce el nombre o índice de la columna a usar como índice temporal: ").strip()

        if seleccion_indice.isdigit():
            try:
                columna_indice = df.columns[int(seleccion_indice)]
            except (IndexError, ValueError):
                log_msg("❌ Entrada inválida para índice temporal.")
                raise ValueError("❌ Entrada inválida para índice temporal.")
        else:
            if seleccion_indice not in df.columns:
                log_msg("❌ Entrada inválida para índice temporal.")
                raise ValueError("❌ Entrada inválida para índice temporal.")
            columna_indice = seleccion_indice

        try:
            df[columna_indice] = pd.to_datetime(df[columna_indice], errors="raise")
            log_msg(f"📅 Columna '{columna_indice}' convertida a datetime correctamente.")
        except Exception:
            log_msg(f"⚠️ No se pudo convertir '{columna_indice}' a datetime. Se mantendrá sin conversión.")

        df.set_index(columna_indice, inplace=True)
        log_msg(f"✅ Columna '{columna_indice}' establecida como índice temporal.")
        return columna_indice

    elif respuesta in respuestas_negativas:
        df.reset_index(drop=True, inplace=True)
        log_msg("⏩ No se estableció un índice temporal. Se usará índice automático.")
        return None

    else:
        df.reset_index(drop=True, inplace=True)
        log_msg("⚠️ Respuesta inválida. Se usará índice automático.")
        return None

def seleccionar_columna_interactivamente(
    df: pd.DataFrame,
    tiene_encabezado: bool,
    input_func: Callable[[str], str] = input,
    log_msg: Optional[Callable[[str], None]] = None,
    modo_auto: bool = MODO_AUTO
) -> Tuple[Union[str, int], str]:
    """
    Permite seleccionar una columna del DataFrame como principal para el análisis,
    y opcionalmente establecer una como índice temporal.

    Comportamiento:
        - Si solo hay una columna, se usa automáticamente.
        - Si hay varias, se permite seleccionar manualmente o automáticamente.
        - En ambos casos, se da la opción de usar una columna como índice temporal.

    Parámetros:
        df (pd.DataFrame): DataFrame cargado desde el archivo CSV.
        tiene_encabezado (bool): Indica si las columnas tienen nombres reales o son numéricas.
        input_func (Callable): Función para capturar entrada del usuario.
        log_msg (Callable, opcional): Función para registrar mensajes.
        modo_auto (bool): Si es True, las selecciones se realizan automáticamente.

    Retorna:
        Tuple[
            Union[str, int],  # Nombre o índice de la columna seleccionada para análisis
            str               # Nombre de la columna usada como índice temporal (o None si no se usó)
        ]
    """
    # Caso especial: solo una columna en el DataFrame
    if df.shape[1] == 1:
        col = df.columns[0]
        df.reset_index(drop=True, inplace=True)  # Se descarta cualquier índice anterior
        log_msg(f"📌 Solo hay una columna: '{col}'")
        log_msg("⏩ Se usará índice automático (0, 1, 2, ...).")
        return col, None

    # Múltiples columnas disponibles: se requiere selección
    log_msg("📊 Varias columnas encontradas." )

    # Seleccionar columna de datos (manual o automático)
    columna = _seleccionar_columna_datos(
        df,
        tiene_encabezado,
        input_func=input_func,
        log_msg=log_msg,
        modo_auto=modo_auto
    )

    # Opción: establecer índice temporal si se desea
    indice_temporal = _gestionar_indice_temporal(
        df,
        input_func=input_func,
        log_msg=log_msg,
        modo_auto=modo_auto
    )

    return columna, indice_temporal

def leer_archivo_datos_interactivo(
    ruta: str,
    separador: str,
    input_func: Callable[[str], str] = input,
    log_msg: Optional[Callable[[str], None]] = None,
    modo_auto: bool = MODO_AUTO
) -> Tuple[pd.DataFrame, str, str]:
    """
    Lee un archivo CSV, permite seleccionar una columna principal y opcionalmente establecer
    una columna como índice temporal.

    Retorna un DataFrame con solo esa columna, ya indexado correctamente.

    Parámetros:
        ruta (str): Ruta del archivo CSV.
        separador (str): Separador para `pd.read_csv`.
        input_func (Callable): Función para capturar entrada (por defecto `input`).
        log_msg (Callable, opcional): Logger opcional.
        modo_auto (bool): Si es True, selección automática de columna y sin índice temporal.

    Retorna:
        Tuple[
            pd.DataFrame,  # Solo una columna, con el índice definido o reseteado
            str,           # Hash del archivo leído
            str,           # Nombre del archiuvo elegido
        ]
    """
    tiene_encabezado = None

    # 1Leer CSV completo
    df, tiene_encabezado, hash_archivo , nombre_archivo= leer_csv(
        ruta,
        separador,
        tiene_encabezado,
        log_msg=log_msg,
        modo_auto=modo_auto
    )

    #  Selección de columna e índice temporal (si procede)
    columna, _ = seleccionar_columna_interactivamente(
        df,
        tiene_encabezado,
        input_func=input_func,
        log_msg=log_msg,
        modo_auto=modo_auto
    )

    # Extraer la columna ya con el índice que haya quedado (temporal o reseteado)
    df_resultado = df[[columna]].copy()

    # Devolver
    return df_resultado, hash_archivo, nombre_archivo

def cargar_datos_interactivamente(
    base_dir: str = BASE_DIR,
    input_func: Callable[[str], str] = input,
    log_msg: Optional[Callable[[str], None]] = None,
    modo_auto: bool = MODO_AUTO,
    indice_auto: int = INDICE_AUTO
) -> Tuple[Optional[pd.DataFrame], Optional[dict], Optional[str],Optional[str]]:
    """
    Ejecuta el flujo completo de carga interactiva (o automática) de un archivo CSV desde un directorio dado.

    Este flujo incluye:
    1. Verificación de archivos CSV en el directorio.
    2. Selección del archivo CSV y del separador (modo manual o automático).
    3. Lectura del archivo.
    4. Generación de un resumen estadístico y estructural del contenido.
    5. Devolución del DataFrame, resumen, log del proceso y hash del archivo.

    Parámetros:
        base_dir (str): Ruta del directorio base donde se buscan archivos CSV.
        input_func (Callable): Función que recibe la entrada del usuario. Por defecto, `input()`.
        log_msg (Callable, opcional): Función para registrar mensajes de log. Puede omitirse si se usa logging interno.
        modo_auto (bool): Si es True, todas las decisiones de este módulo se toman automáticamente.
        indice_auto (int): Índice del archivo a usar automáticamente (solo si modo_auto=True).

    Retorna:
        Tuple[
            Optional[pd.DataFrame],  # DataFrame cargado desde el archivo
            Optional[dict],          # Resumen estadístico generado del contenido
            Optional[str]            # Hash SHA-256 del archivo leído (truncado o completo según implementación)
            Optional[str]            # Nombre del archivo. 
        ]
    """

    # Paso 1: Verificar que existan archivos CSV en el directorio
    if not hay_archivos_csv(base_dir,log_msg=log_msg):
        log_msg(f"❌ No hay archivos .csv disponibles en '{base_dir}'.")
        return None, None, None

    # Paso 2: Seleccionar archivo y separador (interactivo o automático)
    archivo = pedir_nombre_archivo(input_func, modo_auto=modo_auto,log_msg=log_msg)
    separador = pedir_separador(input_func, modo_auto=modo_auto,log_msg=log_msg)

    # Paso 3: Selección desde lista de archivos
    ruta = seleccionar_archivo_csv(
        archivo,
        base_dir,
        input_func=input_func,
        indice_auto=indice_auto,
        log_msg=log_msg,
        modo_auto=modo_auto
    )
    if not ruta:
        log_msg(f"⚠️ No se seleccionó ningún archivo.")
        return None, None, None

    # Paso 4: Lectura y procesamiento del archivo CSV
    try:
        df, hash_archivo, nombre_archivo = leer_archivo_datos_interactivo(
            ruta,
            separador,
            input_func=input_func,
            log_msg=log_msg,
            modo_auto=modo_auto
        )

        resumen = resumen_inicial_serie(df)

        # Construcción del resumen visual
        resumen_str = "\n📊 FASE INICIAL DE LA SERIE\n" + "─" * 50 + "\n"
        resumen_str += "🧾 Resumen inicial de la serie:\n"
        resumen_str += "\n".join(f"- {k}: {v}" for k, v in resumen.items() if k != "explicaciones")

        if "explicaciones" in resumen:
            explicaciones = resumen["explicaciones"]
            resumen_str += "\n\n🧠 EXPLICACIONES DE LOS CAMPOS\n" + "─" * 50 + "\n"
            resumen_str += "\n".join(f"• {k}: {v}" for k, v in explicaciones.items())

        log_msg(resumen_str)

        return df, resumen, hash_archivo, nombre_archivo

    except pd.errors.EmptyDataError:
        log_msg(f"❌ El archivo no contiene datos legibles." )
        return None, None, None, None

    except Exception as e:
        log_msg(f"❌ Error al leer el archivo: {e}")
        return None, None, None, None

def resumen_inicial_serie(df: pd.DataFrame) -> dict:
    """
    Genera un resumen diagnóstico de una serie contenida en un DataFrame con una sola columna.

    El resumen incluye:
        - Nombre y tipo de dato de la columna
        - Cantidad y porcentaje de valores nulos
        - Tipo y naturaleza del índice (¿es temporal?)
        - Presencia de strings
        - Estadísticas básicas si los datos son numéricos

    Parámetros:
        df (pd.DataFrame): DataFrame con una única columna (columna principal de datos).

    Retorna:
        dict: Diccionario con los siguientes campos:
            {
                "columna": str,
                "tipo_dato": str,
                "longitud": int,
                "valores_nulos": int,
                "porcentaje_nulos": str,
                "indice_temporal": bool,
                "tipo_indice": str,
                "contiene_strings": bool,
                "es_numerica": bool,
                "resumen_estadistico": dict | str,
                "descripcion": str,
                "explicaciones": dict
            }

    Lanza:
        ValueError: Si el DataFrame no contiene exactamente una columna.
    """
    # Validación inicial
    if df.shape[1] != 1:
        raise ValueError("El DataFrame debe tener exactamente una columna.")

    resumen = {
        "descripcion": RESUMEN_SERIE_DESCRIPCION,
        "explicaciones": RESUMEN_SERIE_EXPLICACIONES,
    }

    # Extraer la serie
    col = df.columns[0]
    serie = df[col]

    # Información básica
    resumen["columna"] = col
    resumen["tipo_dato"] = str(serie.dtype)
    resumen["longitud"] = len(serie)

    # Valores nulos
    n_nulos = serie.isna().sum()
    resumen["valores_nulos"] = int(n_nulos)
    resumen["porcentaje_nulos"] = (
        f"{(n_nulos / len(serie)) * 100:.2f}%" if len(serie) else "N/A"
    )

    # Índice
    indice = df.index
    resumen["indice_temporal"] = pd.api.types.is_datetime64_any_dtype(indice)
    resumen["tipo_indice"] = str(indice.dtype)

    # Preprocesamiento mínimo para strings vacíos o valores tipo null
    serie_analisis = serie.replace(["", " ", "None", "NULL", "nan"], pd.NA)

    # Detección de strings
    resumen["contiene_strings"] = serie_analisis.apply(lambda x: isinstance(x, str)).any()

    # ¿Es numérica?
    resumen["es_numerica"] = pd.api.types.is_numeric_dtype(serie)

    # Estadísticas descriptivas
    if resumen["es_numerica"]:
        serie_limpia = pd.to_numeric(serie_analisis, errors="coerce").dropna()

        if not serie_limpia.empty:
            resumen["resumen_estadistico"] = {
                "media": round(serie_limpia.mean(), 3),
                "desviacion_std": round(serie_limpia.std(), 3),
                "min": round(serie_limpia.min(), 3),
                "max": round(serie_limpia.max(), 3),
                "asimetria": round(skew(serie_limpia), 3),
                "curtosis": round(kurtosis(serie_limpia), 3),
            }
        else:
            resumen["resumen_estadistico"] = "No disponible (sin datos numéricos válidos)"
    else:
        resumen["resumen_estadistico"] = "No disponible (dato no numérico)"

    return resumen
