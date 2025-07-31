"""
MÓDULO: Funcion_log.py

Este módulo contiene una función auxiliar ligera para gestionar el registro
de mensajes (`log`) de manera flexible y desacoplada. Es especialmente útil
en flujos donde se desea mantener trazabilidad sin depender de `print` directo,
y donde se necesita poder almacenar los mensajes en memoria para su posterior uso.

────────────────────────────────────────────────────────────────────────────
📌 FUNCIONALIDAD PRINCIPAL:

- `crear_logger(verbose, log_buffer)`: Genera una función de log personalizada
  que puede imprimir los mensajes por consola (si `verbose=True`) y/o almacenarlos
  en un buffer de texto (`log_buffer`).

Esto permite un control centralizado y reutilizable del registro de mensajes
en diferentes módulos, incluyendo los de imputación, visualización o validación.

────────────────────────────────────────────────────────────────────────────

"""

from typing import Optional, List


def crear_logger(verbose: bool, log_buffer: Optional[List[str]] = None):
    """
    Crea y devuelve una función de log que escribe mensajes condicionalmente,
    según el modo `verbose`, y opcionalmente los acumula en un buffer de texto.

    Parámetros:
        verbose (bool): Si es True, imprime los mensajes por consola.
        log_buffer (Optional[List[str]]): Lista opcional donde se almacenarán
            los mensajes. Si es None, no se guarda nada.

    Retorna:
        Callable[[str], None]: Una función `log_msg(msg)` que puede ser llamada
        para registrar mensajes, de forma flexible y desacoplada.

    """
    def log_msg(msg: str):
        # Mostrar por consola si verbose está activo
        if verbose:
            print(msg)
        # Acumular en buffer si se proporciona
        if log_buffer is not None:
            log_buffer.append(msg)

    return log_msg
