# MIARMA_python

## 1. Introducción

Este proyecto es una implementación de código abierto que realiza la imputación de valores nulos en series temporales utilizando modelos ARMA sin diferenciación, inspirada en la metodología descrita en el paper [MIARMA: A new method of gap-filling for asteroseismic data](https://www.aanda.org/articles/aa/full_html/2015/03/aa25056-14/aa25056-14.html).

El objetivo es proporcionar una herramienta robusta para rellenar huecos en series temporales astronómicas, así como en cualquier otro tipo de serie pseudoestacionaria con tamaño suficiente y presencia de valores nulos, típicamente originados por pérdidas de datos durante observaciones estelares.

Las características principales del proyecto incluyen:

- Posibilidad de ejecución interactiva o automática bajo premisas iniciales.
- Carga y visualización de datos junto con un resumen estadístico.
- Selección de un tramo representativo y limpio para análisis.
- Búsqueda y validación de modelos ARMA mediante una técnica de doble ejecución.
- Imputación iterativa forward y backward.
- Generación de logs, gráficos, archivos CSV de resultados y análisis espectral vía FFT.

## 2. Instalación

En esta sección se explica cómo preparar el programa para su uso. El proceso básico consiste en descargar el repositorio, tener Python instalado y configurar un entorno virtual para aislar las dependencias.

### Pasos para la instalación

1. **Descargar el código**  
   Clona el repositorio con Git o descarga el ZIP desde GitHub y descomprímelo en tu equipo.

2. **Instalar Python**  
   Asegúrate de tener Python 3.8 o superior instalado. Puedes descargarlo en [python.org](https://www.python.org/downloads/).

3. **Crear y activar un entorno virtual (opcional, pero recomendado)**  
   Para evitar conflictos con otras librerías y mantener tu sistema limpio, se recomienda usar un entorno virtual:

  En Windows (CMD):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

  En macOS/Linux:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
4. **Instalar dependencias**
   Con el entorno virtual activado, si se ha creado previamente, instala las librerías necesarias mediante el siguiente comando:
  ```bash
  pip install -r requirements.txt
  ```
Con estos pasos, el proyecto estará listo para ejecutar y utilizar.

## 3. Uso

En esta sección se mostrará un vídeo explicativo sobre cómo utilizar esta herramienta para imputar valores nulos en series temporales. El vídeo detallará el flujo principal, modos de ejecución y opciones disponibles.

_El enlace al vídeo se añadirá próximamente aquí._

El uso básico consiste en ejecutar el script principal (`main.py`) y seguir las instrucciones en pantalla, o bien utilizar el modo automático para procesar múltiples archivos.

Para más detalles, consulta la documentación en el código fuente o contacta con el autor.

## 4. Estructura del proyecto

El proyecto está organizado en los siguientes archivos y carpetas principales:
```
MIARMA_python/
├── main.py                           # Script principal para ejecutar el flujo completo
├── config.py                         # Archivo de configuración global con parámetros de la herramienta
├── requirements.txt                  # Lista de dependencias para instalar con pip
├── Funciones_iniciales.py            # Funciones para carga y preprocesamiento de datos
├── Funcion_configuracion_rutas.py    # Configuración de rutas para resultados y gráficos
├── Funcion_log.py                    # Funciones para manejo de logs durante la ejecución
├── Funciones_preanalisis.py          # Visualización y análisis preliminar de la serie
├── Funcion_tramo_inteligente.py      # Selección automática del tramo para modelado ARMA
├── Funcion_arma.py                   # Búsqueda y ajuste de modelos ARMA
├── Funcion_validacion_modelos.py     # Validación estadística y selección del mejor modelo
├── Funcion_imputacion.py             # Imputación iterativa de valores faltantes con ARIMA
├── Funcion_graficas_guardado.py      # Visualización y guardado de resultados post-imputación
├── 001_datos/                        # Carpeta para colocar los archivos CSV de entrada
├── 002_resultados/                   # Carpeta generada con todos los resultados del programa
└── README.md                         # Documentación del proyecto
```

**Notas:**

- La carpeta `001_datos/` es donde se deben colocar los archivos CSV con las series temporales para su análisis.
- La carpeta `002_resultados/` se crea automáticamente al ejecutar el programa y contiene todas las salidas generadas.
- El archivo `config.py` es completamente configurable para personalizar el funcionamiento del programa.
- Todos los scripts `.py` contienen funciones para cada etapa del análisis, las cuales están debidamente documentadas para facilitar su comprensión.

## 5. Configuración

El proyecto permite una configuración flexible a través del archivo `config.py`. Este archivo contiene todos los parámetros globales que afectan el comportamiento del programa, incluyendo:

- Rutas y nombres para la carga de datos y guardado de resultados.
- Parámetros de modelado ARMA (órdenes máximos y mínimos, criterios de selección, etc.).
- Opciones para la ejecución automática o manual.
- Ajustes para validaciones estadísticas y filtros de calidad.
- Parámetros visuales para gráficos y análisis espectral.
- Configuraciones de rendimiento, como número de hilos para procesos paralelos.

Para personalizar el funcionamiento del programa, simplemente modifica los valores en `config.py` antes de ejecutar el sistema.

Para un mejor uso y comprensión de estos parámetros, se recomienda revisar tanto el video explicativo en la sección 3 (Uso), como el propio archivo `config.py` y los módulos del programa para entender cómo afectan cada una de las variables a la ejecución del proyecto.

## 6. Licencia

Este software ha sido desarrollado por el Instituto de Astrofísica de Andalucía (IAA) y es propiedad intelectual del mismo.

Se libera con el propósito de fomentar la ciencia abierta y la colaboración en la comunidad científica. 

Para consultas sobre uso, redistribución, modificación o cualquier otra cuestión legal, por favor contacte con el departamento correspondiente del Instituto de Astrofísica de Andalucía.

## 7. Contacto

Para cualquier consulta, duda o sugerencia relacionada con este proyecto, puede contactar a:

- **Raúl Díez Riesco** — desarrollador principal  
  Email: raul.diez@iaa.es

- **Javier Pascual** — contacto adicional  
  Email: javier@iaa.es
  Email(2): j.pascual@csic.es

- **Instituto de Astrofísica de Andalucía (IAA)**  
  Página web: [https://www.iaa.csic.es](https://www.iaa.csic.es)  
  Email general: info@iaa.es



