# -*- coding: utf-8 -*-
"""
REPRODUCCIÓN COMPLETA DEL ARTÍCULO DEL AGENTE VIAJERO — VERSIÓN 2
Autor del material: Jaime Aguilar Ortiz

Un protocolo de reproducción es el programa que repite un experimento completo,
conserva sus datos y transforma sus salidas en evidencia revisable. Este archivo
incluye tanto el algoritmo genético como ese protocolo; NO ejecuta solamente la
semilla 42 y NO necesita importar el programa anterior ni abrir archivos Word.

AL EJECUTAR ESTE ARCHIVO, SIN CAMBIAR PARÁMETROS:
  1. Ejecuta una batería incluida de 26 casos funcionales.
  2. Genera UNA matriz simétrica de 20 ciudades, con semilla de datos 20260917.
  3. Ejecuta las TREINTA semillas algorítmicas 42, 43, ..., 71.
  4. Repite la semilla 42 para verificar identidad, fuera de la muestra de 30.
  5. Calcula las Tablas 4, 5, 6 y 7 y las Figuras 2, 3, 4 y 5.
  6. Exporta las Tablas 1, 2 y 3 y dibuja la Figura 1 con instrucciones de Matplotlib.
  7. Guarda las siete tablas en CSV, PNG y SVG; las cinco figuras en PNG y SVG;
     un informe HTML, datos completos, pruebas y comprobaciones contra el artículo.

PROCEDENCIA, SIN MEZCLAR TIPOS DE EVIDENCIA
- Tablas 1 y 2: descripción de los métodos y de las configuraciones del artículo.
- Tabla 3: TRANSCRIPCIÓN DOCUMENTAL de las 612 y 309 comprobaciones de la guía.
  Exportar esta tabla NO significa ejecutar esas baterías históricas.
- Figura 1: esquema redibujado con texto, recuadros y flechas. No contiene una
  imagen incrustada ni requiere archivos externos. Su SVG es vectorial. Se
  conservan las siete etapas, pero no los píxeles de la ilustración original.
- Tablas 4 a 7 y Figuras 2 a 5: resultados CALCULADOS en esta ejecución.
- Las cifras publicadas se usan SOLO para comparar después de la búsqueda;
  jamás se introducen como costos, individuos, soluciones ni curvas calculadas.
- Los 26 casos incluidos implementan las categorías descritas en el artículo.
  No se afirma identidad de código con una batería histórica no suministrada.

EJECUCIÓN EN SPYDER
Abrir este único archivo y pulsar F5. Al terminar, se abre un informe local con
TODAS las tablas y figuras. RESULTADO queda disponible en el explorador de
variables. Los resultados se guardan junto al script en una carpeta NUEVA por
cada ejecución. No se borran ni se mezclan resultados anteriores.

EJECUCIÓN EN TERMINAL
    python REPRODUCIR_ARTICULO_AGENTE_VIAJERO_30_SEMILLAS_V2.py
    python REPRODUCIR_ARTICULO_AGENTE_VIAJERO_30_SEMILLAS_V2.py --sin-abrir
    python REPRODUCIR_ARTICULO_AGENTE_VIAJERO_30_SEMILLAS_V2.py --salida "C:/Resultados"

Dependencias externas: NumPy y Matplotlib. No se instalan automáticamente.
    python -m pip install numpy matplotlib
Entorno de contraste del artículo: Python 3.13.5, NumPy 2.3.5, Matplotlib 3.10.8.
No se necesita GPU, pandas, SciPy, Excel, Word, una API ni conexión a internet.

IMPORTANTE PARA LA REPETIBILIDAD
Las 16 funciones numéricas de la versión aportada se conservan completas.
No se vectorizan de otro modo ni se reorganizan los sorteos. Crear todos los
padres de una generación ANTES de cruzar y mutar es parte del flujo original.
Los tiempos cambian con el equipo. Las coincidencias se comprueban y se informan;
si una biblioteca o plataforma produce diferencias, estas NO se sustituyen por
los valores esperados. No se garantiza identidad binaria entre todos los equipos.

FUENTES LOCALES
Artículo: ARTICULO_INVESTIGACION_AGENTE_VIAJERO_7_ETAPAS.docx, Tablas 1-7,
Figuras 1-5, apartados 3.1-3.3 y 5.1-5.4 y Apéndice A.
Guía: METODOS_ALGORITMOS_GENETICOS_ILUSTRADO_COMENTADO(2).docx.
Código base: ALGORITMO_AGENTE_VIAJERO_7_ETAPAS_COMENTADO(1).py.
La representación por claves aleatorias sigue el antecedente de Bean (1994),
https://doi.org/10.1287/ijoc.6.2.154 ; no se propone como un método nuevo.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import platform
import re
import sys
import textwrap
import traceback
import webbrowser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

# Estas variables solo afectan bibliotecas importadas después de su definición.
# En una consola de Spyder ya iniciada, NumPy podría estar cargado previamente.
# No se utiliza álgebra lineal BLAS en la búsqueda; se registra el entorno real.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

try:
    import numpy as np
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
    from matplotlib.backends.backend_agg import FigureCanvasAgg
except ImportError as error:
    raise ImportError(
        "Falta una dependencia en el entorno que ejecuta este archivo.\n"
        f'Instalar con: "{sys.executable}" -m pip install numpy matplotlib\n'
        "Después, reiniciar la consola de Spyder. No se modificó su entorno."
    ) from error

# ============================================================================
# A. NÚCLEO NUMÉRICO: LAS MISMAS OPERACIONES Y EL MISMO ORDEN ALEATORIO
# ============================================================================

METODOS = (
    (1, "Inicialización", "M01: uniforme independiente en [0, 1)"),
    (2, "Evaluación", "M05: distancia total del recorrido cerrado"),
    (3, "Selección", "M12: torneo con reemplazo"),
    (4, "Recombinación", "M14: cruce uniforme de claves reales"),
    (5, "Mutación", "M18: reinicio uniforme de claves reales"),
    (6, "Reemplazo", "M22: generacional con elitismo"),
    (7, "Criterio de paro", "M25: máximo de generaciones completas"),
)


# ============================================================================
# FUNCIONES AUXILIARES: VALIDACIÓN, DATOS Y DECODIFICACIÓN
# No son etapas genéticas adicionales: preparan o interpretan sus entradas.
# ============================================================================

def validar_entero(valor, nombre: str, minimo: int) -> int:
    """Exige un entero admisible; rechaza booleanos, decimales y negativos.

    El nombre aparece en el mensaje de error para localizar el parámetro.
    Se aceptan enteros normales de Python y enteros procedentes de NumPy.
    """
    if isinstance(valor, (bool, np.bool_)) or not isinstance(
        valor, (int, np.integer)
    ):
        raise TypeError(f"{nombre} debe ser un número entero.")
    if valor < minimo:
        raise ValueError(f"{nombre} debe ser mayor o igual que {minimo}.")
    return int(valor)


def validar_probabilidad(valor, nombre: str) -> float:
    """Comprueba una probabilidad escalar finita comprendida entre 0 y 1.

    No acepta porcentajes escritos como 90: para 90 % debe utilizarse 0.90.
    Tampoco acepta listas, cadenas ni booleanos como probabilidades.
    """
    if isinstance(valor, (bool, np.bool_)) or not isinstance(
        valor, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{nombre} debe ser un número real.")
    valor = float(valor)
    if not np.isfinite(valor) or not 0.0 <= valor <= 1.0:
        raise ValueError(f"{nombre} debe pertenecer al intervalo [0, 1].")
    return valor


def validar_claves(datos, dimensiones: int) -> np.ndarray:
    """Valida un cromosoma o una población de claves reales en [0, 1).

    dimensiones=1 significa un individuo; dimensiones=2, una población.
    Cada cromosoma necesita al menos tres ciudades. La función solo valida;
    no ordena, no modifica genes y no repara silenciosamente entradas erróneas.
    """
    claves = np.asarray(datos, dtype=float)
    if claves.ndim != dimensiones or claves.size == 0:
        raise ValueError(f"Se requiere un arreglo no vacío de {dimensiones} dimensiones.")
    if claves.shape[-1] < 3:
        raise ValueError("Se requieren al menos tres claves, una por ciudad.")
    if not np.all(np.isfinite(claves)):
        raise ValueError("Las claves no pueden contener NaN ni infinitos.")
    if np.any(claves < 0.0) or np.any(claves >= 1.0):
        raise ValueError("Todas las claves deben pertenecer a [0, 1).")
    return claves


def validar_costos(datos, cantidad: int) -> np.ndarray:
    """Exige una distancia finita y no negativa por cada individuo.

    La posición i del vector debe corresponder al individuo i de la población.
    Verificar esa cantidad ayuda a detectar desajustes al reemplazar individuos.
    """
    costos = np.asarray(datos, dtype=float)
    if costos.shape != (cantidad,):
        raise ValueError("Debe existir exactamente un costo por individuo.")
    if not np.all(np.isfinite(costos)) or np.any(costos < 0):
        raise ValueError("Las distancias deben ser finitas y no negativas.")
    return costos


def validar_matriz_distancias(datos) -> np.ndarray:
    """Valida una matriz para el agente viajero simétrico y devuelve una copia.

    Se requiere una matriz cuadrada, finita, no negativa y con diagonal cero.
    La simetría se verifica con una tolerancia numérica pequeña. No se impone
    desigualdad triangular: la instancia aleatoria original no la garantiza.
    """
    matriz = np.asarray(datos, dtype=float)
    if matriz.ndim != 2 or matriz.shape[0] != matriz.shape[1]:
        raise ValueError("La matriz de distancias debe ser cuadrada.")
    if matriz.shape[0] < 3:
        raise ValueError("Se requieren al menos tres ciudades.")
    if not np.all(np.isfinite(matriz)) or np.any(matriz < 0):
        raise ValueError("La matriz debe contener valores finitos no negativos.")
    if np.any(np.diag(matriz) != 0.0):
        raise ValueError("La diagonal de la matriz debe ser exactamente cero.")
    if not np.allclose(matriz, matriz.T, rtol=1e-12, atol=1e-12):
        raise ValueError("Esta versión requiere una matriz simétrica.")
    return matriz.copy()


def generar_instancia(
    num_ciudades: int, semilla: int, modo: str = "simetrica_aleatoria"
) -> tuple[np.ndarray, np.ndarray]:
    """Genera los datos del problema y las coordenadas para representarlos.

    simetrica_aleatoria: mantiene la construcción del programa original,
    D=(A+A.T)/2, con A uniforme en [0,100). Las coordenadas circulares son
    únicamente un esquema: la longitud de un segmento NO expresa su costo.

    euclidiana: primero genera coordenadas en [0,100)^2 y después calcula
    las distancias a partir de ellas. Aquí sí existe correspondencia geométrica.
    El generador de datos es independiente del utilizado por el algoritmo.
    """
    n = validar_entero(num_ciudades, "num_ciudades", 3)
    semilla = validar_entero(semilla, "semilla de los datos", 0)
    rng = np.random.default_rng(semilla)
    if modo == "simetrica_aleatoria":
        aleatoria = rng.random((n, n)) * 100.0
        matriz = (aleatoria + aleatoria.T) / 2.0
        angulos = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        coordenadas = np.column_stack((np.cos(angulos), np.sin(angulos)))
    elif modo == "euclidiana":
        coordenadas = rng.random((n, 2)) * 100.0
        diferencias = coordenadas[:, None, :] - coordenadas[None, :, :]
        matriz = np.sqrt(np.sum(diferencias ** 2, axis=2))
    else:
        raise ValueError("modo debe ser 'simetrica_aleatoria' o 'euclidiana'.")
    np.fill_diagonal(matriz, 0.0)
    return validar_matriz_distancias(matriz), coordenadas


def decodificar_claves(claves) -> np.ndarray:
    """Convierte prioridades reales en una ruta sin repetir ni omitir ciudades.

    La posición de cada clave identifica una ciudad. argsort devuelve esas
    posiciones ordenadas por el valor de su clave, no los valores de las claves.
    Con [0.70,0.10,0.90,0.40] se obtiene [1,3,0,2]. Se rota ese mismo ciclo
    para presentarlo desde la ciudad 0: [0,2,1,3]. No se altera el recorrido.

    El ordenamiento estable resuelve claves iguales por identificador de ciudad.
    No se añade el regreso a 0 al cromosoma: el evaluador cierra el ciclo.
    """
    claves = validar_claves(claves, 1)
    ruta = np.argsort(claves, kind="stable")
    posicion_cero = int(np.flatnonzero(ruta == 0)[0])
    return np.roll(ruta, -posicion_cero)


def evaluar_ruta(ruta, matriz_distancias) -> float:
    """Calcula y verifica la distancia de una ruta explícita de ciudades.

    Esta función auxiliar se utiliza para auditar la solución final. Cada ciudad
    debe aparecer exactamente una vez. np.roll produce la ciudad siguiente de
    cada tramo; para la última posición devuelve la primera ciudad.
    """
    matriz = validar_matriz_distancias(matriz_distancias)
    ruta = np.asarray(ruta)
    n = len(matriz)
    if ruta.shape != (n,) or not np.issubdtype(ruta.dtype, np.integer):
        raise ValueError("La ruta debe contener un identificador entero por ciudad.")
    if not np.array_equal(np.sort(ruta), np.arange(n)):
        raise ValueError("La ruta debe visitar todas las ciudades exactamente una vez.")
    distancia = float(np.sum(matriz[ruta, np.roll(ruta, -1)]))
    if not np.isfinite(distancia):
        raise FloatingPointError("La suma de distancias excedió el rango numérico.")
    return distancia


# ============================================================================
# ETAPA 1. INICIALIZACIÓN — M01: MUESTREO UNIFORME INDEPENDIENTE EN UNA CAJA
# Alternativas: M01 uniforme, M02 Bernoulli, M03 hipercubo latino, M04 Dirichlet.
# Se elige M01 porque las prioridades son reales independientes en [0,1).
# M03 también sería compatible, pero estratificar claves no asegura rutas más
# diversas. M02 representa bits; M04 impone una suma uno que aquí no hace falta.
# ============================================================================

def etapa_1_inicializar(
    tamano_poblacion: int, num_ciudades: int, rng: np.random.Generator
) -> np.ndarray:
    """Crea N cromosomas, cada uno con una clave real por ciudad.

    Una fila representa una solución; una columna representa siempre la misma
    ciudad. La fórmula general de M01, a+(b-a)*U, se reduce a U para a=0, b=1.
    Las rutas aparecen al decodificar; todavía no se eligen los mejores individuos.
    """
    n = validar_entero(tamano_poblacion, "tamano_poblacion", 2)
    d = validar_entero(num_ciudades, "num_ciudades", 3)
    return rng.random((n, d))


# ============================================================================
# ETAPA 2. EVALUACIÓN — M05: FUNCIÓN OBJETIVO ESCALAR DIRECTA
# Alternativas: M05 directa, M06 ponderada, M07 penalización, M08 factibilidad.
# Solo se minimiza la distancia. El decodificador garantiza visitas sin repetición
# y la matriz permite todas las conexiones; no hay restricciones adicionales.
# Por ello no se necesitan pesos, penalizaciones ni reglas de factibilidad.
# ============================================================================

def etapa_2_evaluar(poblacion, matriz_distancias) -> np.ndarray:
    """Asigna a cada cromosoma la distancia de su recorrido cerrado.

    Para ruta=(r_0,...,r_(d-1)), se calcula:
        f(ruta) = D[r_0,r_1] + ... + D[r_(d-1),r_0].
    Menor distancia significa mejor solución. No se calcula 1/distancia:
    el torneo y el reemplazo comparan directamente costos que se minimizan.

    Se procesa la población por filas. Aunque el cálculo sea vectorizado,
    evaluar N filas cuenta como N evaluaciones, no como una sola evaluación.
    """
    poblacion = validar_claves(poblacion, 2)
    matriz = validar_matriz_distancias(matriz_distancias)
    if poblacion.shape[1] != len(matriz):
        raise ValueError("Debe existir una clave por cada ciudad de la matriz.")
    # Cada fila contiene los identificadores de las ciudades en orden de visita.
    rutas = np.argsort(poblacion, axis=1, kind="stable")
    # Desplazar una posición cierra también el último tramo de cada recorrido.
    siguientes = np.roll(rutas, -1, axis=1)
    costos = np.sum(matriz[rutas, siguientes], axis=1)
    return validar_costos(costos, len(poblacion))


# ============================================================================
# ETAPA 3. SELECCIÓN — M12: TORNEO DETERMINISTA ENTRE PARTICIPANTES ALEATORIOS
# Alternativas: M09 ruleta, M10 SUS, M11 ranking, M12 torneo.
# Se elige torneo por su comparación directa y su presión regulable mediante k.
# Los otros métodos son alternativas válidas, no pasos que deban encadenarse.
# ============================================================================

def etapa_3_seleccionar(
    costos, cantidad: int, tamano_torneo: int, rng: np.random.Generator
) -> np.ndarray:
    """Devuelve los índices de los progenitores ganadores, no sus cromosomas.

    Cada torneo sortea k participantes CON reemplazo, tal como M12 en la guía.
    Gana el menor costo; un empate conserva la primera aparición sorteada.
    Un mismo individuo puede competir o ganar varias veces. Aumentar k aumenta
    la presión selectiva, pero no demuestra que mejore el resultado final.
    """
    costos = np.asarray(costos, dtype=float)
    if costos.ndim != 1 or costos.size == 0:
        raise ValueError("Se requiere un vector no vacío de costos.")
    costos = validar_costos(costos, len(costos))
    m = validar_entero(cantidad, "cantidad de progenitores", 1)
    k = validar_entero(tamano_torneo, "tamano_torneo", 1)
    # La fila r contiene los participantes del torneo r.
    participantes = rng.integers(0, len(costos), size=(m, k))
    posiciones_ganadoras = np.argmin(costos[participantes], axis=1)
    # Convertir la posición dentro del torneo en un índice de la población.
    return participantes[np.arange(m), posiciones_ganadoras]


# ============================================================================
# ETAPA 4. RECOMBINACIÓN — M14: CRUCE UNIFORME POR MÁSCARA
# Alternativas: M13 un punto, M14 uniforme, M15 aritmético, M16 BLX-alpha.
# Se elige M14 para heredar la prioridad de cada ciudad sin tratar su etiqueta
# como una magnitud. El cruce actúa en claves reales, NO sobre la ruta ordenada.
# M13 también es compatible con claves, pero sus bloques dependen del orden de
# etiquetas; M15 interpola prioridades y M16 puede exigir reparación de cotas.
# ============================================================================

def etapa_4_recombinar(
    padre_1, padre_2, rng: np.random.Generator, probabilidad_mascara: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Produce dos hijos con herencia complementaria a partir de una máscara.

    Si mascara[j] es verdadera, hijo_1 recibe la clave j del primer padre;
    hijo_2 recibe la del segundo. Si es falsa, se intercambian las procedencias.
    q=0.5 no favorece a ninguno de los padres en cada posición.

    Esta q NO es la probabilidad de cruzar una pareja, que se decide en el ciclo.
    La validez de las rutas viene del decodificador; M14 no preserva sus aristas.
    """
    a = validar_claves(padre_1, 1)
    b = validar_claves(padre_2, 1)
    if a.shape != b.shape:
        raise ValueError("Los progenitores deben tener la misma longitud.")
    q = validar_probabilidad(probabilidad_mascara, "probabilidad_mascara")
    mascara = rng.random(len(a)) < q
    hijo_1 = np.where(mascara, a, b)
    hijo_2 = np.where(mascara, b, a)
    return hijo_1, hijo_2, mascara


# ============================================================================
# ETAPA 5. MUTACIÓN — M18: REINICIALIZACIÓN UNIFORME DE GENES
# Alternativas: M17 inversión de bits, M18 reinicio, M19 gaussiana,
# M20 polinomial acotada. M17 no corresponde a esta representación real.
# Se elige M18 para introducir prioridades nuevas sin ajustar sigma ni eta.
# M19 y M20 permitirían usar prioridades reales definiendo cotas compatibles;
# un cambio pequeño puede mantener el orden. M18 tampoco garantiza una mejora.
# ============================================================================

def etapa_5_mutar(
    individuo, tasa_mutacion: float, rng: np.random.Generator
) -> np.ndarray:
    """Reinicia independientemente cada clave con probabilidad p.

    Una clave seleccionada recibe un nuevo uniforme en [0,1); las demás se
    conservan. Es una probabilidad POR GEN, no una probabilidad por individuo.
    Con d ciudades y p=1/d se selecciona un gen en promedio, no exactamente uno.
    Mutar una sola clave puede cambiar la posición de una ciudad al decodificar.

    La función devuelve un cromosoma nuevo; nunca modifica una élite ni un padre.
    No se realiza aquí intercambio de ciudades ni inversión de segmentos.
    """
    claves = validar_claves(individuo, 1)
    p = validar_probabilidad(tasa_mutacion, "tasa_mutacion")
    mascara = rng.random(len(claves)) < p
    nuevas_claves = rng.random(len(claves))
    return np.where(mascara, nuevas_claves, claves)


# ============================================================================
# ETAPA 6. REEMPLAZO — M22: GENERACIONAL CON ELITISMO EXPLÍCITO
# Alternativas: M21 generacional, M22 elitista, M23 (mu+lambda), M24 estacionario.
# Se elige M22 para proteger unas pocas soluciones y renovar los otros puestos.
# M21 no protege al mejor; M23 y M24 son válidos, pero aplican otra política de
# supervivencia. M22 reserva e padres y acepta TODOS los N-e hijos entregados.
# ============================================================================

def etapa_6_reemplazar(
    poblacion, costos, hijos, costos_hijos, num_elites: int
) -> tuple[np.ndarray, np.ndarray]:
    """Conserva e padres bien evaluados y completa con exactamente N-e hijos.

    Se ordenan los costos parentales y se copian sus cromosomas asociados.
    Población y costos nuevos se construyen en idéntico orden. Los empates entre
    padres mantienen el orden de sus filas. No se eliminan duplicados.

    Con al menos una élite y un objetivo fijo, el mejor costo no puede empeorar.
    Esto conserva lo encontrado; no garantiza que aparezca una solución mejor.
    """
    poblacion = validar_claves(poblacion, 2)
    hijos = validar_claves(hijos, 2)
    n = len(poblacion)
    costos = validar_costos(costos, n)
    costos_hijos = validar_costos(costos_hijos, len(hijos))
    e = validar_entero(num_elites, "num_elites", 1)
    if e >= n or hijos.shape != (n - e, poblacion.shape[1]):
        raise ValueError("Se requiere 1 <= num_elites < N y exactamente N-e hijos.")
    indices_elite = np.argsort(costos, kind="stable")[:e]
    nueva_poblacion = np.vstack((poblacion[indices_elite], hijos))
    nuevos_costos = np.concatenate((costos[indices_elite], costos_hijos))
    return nueva_poblacion, nuevos_costos


# ============================================================================
# ETAPA 7. CRITERIO DE PARO — M25: MÁXIMO DE GENERACIONES COMPLETAS
# La guía contiene SEIS alternativas aquí: M25 generaciones, M26 evaluaciones,
# M27 tiempo, M28 estancamiento, M29 dispersión y M30 meta de calidad.
# Se elige únicamente M25: fija un trabajo sencillo de reproducir y permite
# observar la evolución completa. No equivale a afirmar que el óptimo se alcanzó.
# ============================================================================

def etapa_7_parar(generacion: int, num_generaciones: int) -> bool:
    """Consulta si se completó el número permitido de transiciones.

    La población inicial es la generación 0. La función no incrementa contadores.
    Un máximo de 0 permite evaluar la población inicial, pero no crear hijos.
    No se activan silenciosamente otros criterios de terminación.
    """
    actual = validar_entero(generacion, "generacion", 0)
    maximo = validar_entero(num_generaciones, "num_generaciones", 0)
    return actual >= maximo


# ============================================================================
# INTEGRACIÓN: LAS SIETE ETAPAS EN UN ÚNICO ALGORITMO
# Orden efectivo: 1 -> 2 -> control 7 -> 3 -> 4 -> 5 -> repetir 2 -> 6 -> control 7.
# Reevaluar hijos es repetir la etapa 2; no constituye una octava etapa.
# ============================================================================

def algoritmo_genetico(
    matriz_distancias,
    tamano_poblacion: int = 100,
    tasa_mutacion: float | None = None,
    num_generaciones: int = 500,
    num_elites: int = 2,
    tamano_torneo: int = 3,
    probabilidad_cruce: float = 0.90,
    semilla: int = 42,
    mostrar_progreso: bool = True,
    intervalo_impresion: int = 25,
) -> dict:
    """Ejecuta la búsqueda y devuelve solución, costos, historial y parámetros.

    Recibe una matriz ya generada para poder comparar corridas sobre los mismos
    datos. tasa_mutacion=None utiliza 1/num_ciudades. Las élites se copian sin
    variación y sin reevaluación porque el modelo de distancias permanece fijo.

    La evaluación de búsqueda cuenta N + G*(N-e) individuos. La comprobación
    final de la ruta es una evaluación adicional, registrada por separado.
    """
    matriz = validar_matriz_distancias(matriz_distancias)
    d = len(matriz)
    n = validar_entero(tamano_poblacion, "tamano_poblacion", 2)
    g_max = validar_entero(num_generaciones, "num_generaciones", 0)
    e = validar_entero(num_elites, "num_elites", 1)
    k = validar_entero(tamano_torneo, "tamano_torneo", 1)
    semilla = validar_entero(semilla, "semilla", 0)
    intervalo = validar_entero(intervalo_impresion, "intervalo_impresion", 1)
    pc = validar_probabilidad(probabilidad_cruce, "probabilidad_cruce")
    pm = 1.0 / d if tasa_mutacion is None else validar_probabilidad(
        tasa_mutacion, "tasa_mutacion"
    )
    if e >= n:
        raise ValueError("num_elites debe ser menor que tamano_poblacion.")
    if not isinstance(mostrar_progreso, (bool, np.bool_)):
        raise TypeError("mostrar_progreso debe ser booleano.")

    # Un solo generador para toda la evolución; nunca se reinicia por generación.
    rng = np.random.default_rng(semilla)
    inicio = perf_counter()
    poblacion = etapa_1_inicializar(n, d, rng)                  # ETAPA 1: M01
    costos = etapa_2_evaluar(poblacion, matriz)                # ETAPA 2: M05
    evaluaciones_busqueda = n
    generacion = 0

    # Guardar juntos el costo y una COPIA del individuo que produjo ese costo.
    indice = int(np.argmin(costos))
    mejores_claves = poblacion[indice].copy()
    mejor_distancia = float(costos[indice])
    historial_mejor = [mejor_distancia]
    historial_promedio = [float(np.mean(costos))]
    if mostrar_progreso:
        print(f"Generación {0:4d} | mejor = {mejor_distancia:.6f}")

    # ETAPA 7: comprobar antes de autorizar la siguiente generación.
    while not etapa_7_parar(generacion, g_max):
        cantidad_hijos = n - e
        parejas = (cantidad_hijos + 1) // 2
        indices = etapa_3_seleccionar(costos, 2 * parejas, k, rng)  # ETAPA 3
        hijos = np.empty((cantidad_hijos, d), dtype=float)
        posicion = 0

        for pareja in range(parejas):
            padre_1 = poblacion[indices[2 * pareja]]
            padre_2 = poblacion[indices[2 * pareja + 1]]
            # pc decide si se aplica M14; q=0.5 decide la procedencia de cada gen.
            if rng.random() < pc:
                hijo_1, hijo_2, _ = etapa_4_recombinar(        # ETAPA 4: M14
                    padre_1, padre_2, rng, probabilidad_mascara=0.5
                )
            else:
                hijo_1, hijo_2 = padre_1.copy(), padre_2.copy()
            for hijo in (hijo_1, hijo_2):
                if posicion == cantidad_hijos:
                    break  # Permite también un número impar de hijos.
                hijos[posicion] = etapa_5_mutar(hijo, pm, rng)  # ETAPA 5: M18
                posicion += 1

        # Nunca se comparan hijos modificados usando los costos de sus padres.
        costos_hijos = etapa_2_evaluar(hijos, matriz)         # REPETIR ETAPA 2
        evaluaciones_busqueda += cantidad_hijos
        poblacion, costos = etapa_6_reemplazar(               # ETAPA 6: M22
            poblacion, costos, hijos, costos_hijos, e
        )
        generacion += 1

        # El índice se obtiene de los costos NUEVOS y se aplica a la población
        # NUEVA. Así se corrige la desalineación del mejor individuo del original.
        indice = int(np.argmin(costos))
        mejor_actual = float(costos[indice])
        if mejor_actual > mejor_distancia:
            raise RuntimeError("El mejor costo empeoró: revisar el reemplazo elitista.")
        if mejor_actual < mejor_distancia:
            mejor_distancia = mejor_actual
            mejores_claves = poblacion[indice].copy()
        historial_mejor.append(mejor_distancia)
        historial_promedio.append(float(np.mean(costos)))
        if mostrar_progreso and (
            generacion % intervalo == 0 or generacion == g_max
        ):
            print(
                f"Generación {generacion:4d} | mejor = {mejor_distancia:.6f} | "
                f"promedio = {historial_promedio[-1]:.6f}"
            )

    # Auditoría final: recuperar la ruta y recalcular su distancia directamente.
    mejor_ruta = decodificar_claves(mejores_claves)
    distancia_verificada = evaluar_ruta(mejor_ruta, matriz)
    if not np.isclose(distancia_verificada, mejor_distancia, rtol=1e-12, atol=1e-10):
        raise RuntimeError("La ruta devuelta no coincide con su distancia almacenada.")
    if evaluaciones_busqueda != n + generacion * (n - e):
        raise RuntimeError("El contador de evaluaciones no coincide con el trabajo realizado.")

    return {
        "mejor_ruta": mejor_ruta.copy(),
        "mejores_claves": mejores_claves.copy(),
        "mejor_distancia": mejor_distancia,
        "distancia_verificada": distancia_verificada,
        "matriz_distancias": matriz,
        "historial_mejor": np.asarray(historial_mejor),
        "historial_promedio": np.asarray(historial_promedio),
        "generaciones": generacion,
        "evaluaciones_busqueda": evaluaciones_busqueda,
        "evaluaciones_verificacion": 1,
        "evaluaciones_totales": evaluaciones_busqueda + 1,
        "motivo_paro": "M25: máximo de generaciones completas",
        "tiempo_segundos": perf_counter() - inicio,
        "parametros": {
            "num_ciudades": d, "tamano_poblacion": n, "tasa_mutacion": pm,
            "num_generaciones": g_max, "num_elites": e, "tamano_torneo": k,
            "probabilidad_cruce": pc, "probabilidad_mascara": 0.5,
            "semilla_algoritmo": semilla,
        },
    }


# ============================================================================
# B. PROTOCOLO DEL ARTÍCULO, ARCHIVOS Y PRUEBAS FUNCIONALES
# ============================================================================

# Estos parámetros fijan el experimento publicado. No hay un modo rápido oculto.
# Cambiarlos significa realizar OTRO experimento; la comparación contra las tablas
# publicadas podría dejar de coincidir. Las referencias no alteran estos valores.
SEMILLA_DATOS = 20260917
SEMILLAS_ALGORITMICAS = tuple(range(42, 72))  # 72 se excluye: 42 a 71 = 30 semillas.
SEMILLA_PRINCIPAL = 42
NUM_CIUDADES = 20
PARAMETROS_AG = {
    "tamano_poblacion": 100,
    "tasa_mutacion": 0.05,
    "num_generaciones": 500,
    "num_elites": 2,
    "tamano_torneo": 3,
    "probabilidad_cruce": 0.90,
}
CONTROLES_TABLA_4 = (0, 25, 50, 100, 113, 120, 250, 500)
TOLERANCIA_DIAGNOSTICA = 1e-10  # Diagnóstico posterior; NO altera la evolución.
ABRIR_INFORME_AL_TERMINAR = True
CARPETA_BASE_PERSONALIZADA = None  # Ejemplo: r"C:\Resultados_AG"; None usa el script.


def convertir_json(objeto):
    """Convierte valores NumPy y rutas a tipos JSON, sin perder las matrices.

    Este conversor se usa solo al exportar. No redondea valores de búsqueda ni
    cambia los arreglos del algoritmo. Los archivos numéricos grandes se guardan
    además en NPZ, que conserva sus tipos y precisión sin convertirlos a texto.
    """
    if isinstance(objeto, np.ndarray):
        return objeto.tolist()
    if isinstance(objeto, np.generic):
        return objeto.item()
    if isinstance(objeto, Path):
        return str(objeto)
    raise TypeError(f"Tipo no serializable: {type(objeto).__name__}")


def guardar_json(ruta: Path, contenido) -> None:
    """Escribe JSON legible y rechaza NaN o infinito en lugar de ocultarlos."""
    ruta.write_text(json.dumps(contenido, ensure_ascii=False, indent=2,
                               default=convertir_json, allow_nan=False), encoding="utf-8")


def guardar_csv(ruta: Path, encabezados, filas) -> None:
    """Escribe datos tabulares con precisión completa y caracteres españoles.

    UTF-8 con marca de inicio permite abrir las etiquetas en aplicaciones que
    detectan la codificación por esa marca. Se usa coma como separador y punto
    decimal. Los textos con comas se entrecomillan mediante el módulo csv.
    Las seis cifras visibles del artículo se aplican SOLO a la presentación.
    """
    with ruta.open("w", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(encabezados)
        for fila in filas:
            escritor.writerow([x.item() if isinstance(x, np.generic) else x for x in fila])


def exigir(condicion: bool, mensaje: str) -> None:
    """Interrumpe una prueba fallida incluso cuando Python se ejecuta con -O.

    No se usa assert para las comprobaciones esenciales: la optimización de
    Python puede desactivar asserts. Aquí una condición falsa siempre se informa.
    """
    if not condicion:
        raise AssertionError(mensaje)


def esperar_error(funcion, tipos=(TypeError, ValueError)) -> None:
    """Aprueba solo si la entrada inválida produce el tipo de error esperado."""
    try:
        funcion()
    except tipos:
        return
    raise AssertionError("La entrada inválida no fue rechazada como se esperaba.")


def ejecutar_pruebas_funcionales() -> list[dict]:
    """Ejecuta 26 casos distribuidos como describe el apartado 3.3 del artículo.

    Cada función registrada cuenta como UN caso, aunque examine varias
    propiedades relacionadas. Sus resultados se calculan de nuevo. Esta batería
    es independiente de las 612 y 309 comprobaciones documentales de la Tabla 3.
    Se conservan nombres, categorías, duración y errores de cada caso para poder
    revisar la cobertura; aprobarlos no demuestra ausencia universal de errores.
    """
    casos = []

    def caso(grupo: str, descripcion: str):
        def registrar(funcion):
            casos.append((grupo, descripcion, funcion))
            return funcion
        return registrar

    @caso("Validaciones", "Enteros: valor admisible y rechazo de booleanos, decimales y negativos")
    def probar_enteros():
        exigir(validar_entero(np.int64(3), "n", 1) == 3, "Falló entero NumPy.")
        for valor in (True, 2.5, -1):
            esperar_error(lambda v=valor: validar_entero(v, "n", 1))

    @caso("Validaciones", "Probabilidades: extremos válidos y entradas inadmisibles")
    def probar_probabilidades():
        for p in (0, 0.5, 1):
            exigir(validar_probabilidad(p, "p") == p, "Probabilidad válida rechazada.")
        for p in (True, -0.1, 1.1, float("nan"), float("inf"), "0.5"):
            esperar_error(lambda v=p: validar_probabilidad(v, "p"))

    @caso("Validaciones", "Matriz: rechazo de dimensiones no cuadradas o insuficientes")
    def probar_dimensiones():
        for matriz in (np.zeros((3, 4)), np.zeros((2, 2)), np.zeros(4)):
            esperar_error(lambda m=matriz: validar_matriz_distancias(m))

    @caso("Validaciones", "Matriz: rechazo de costos negativos")
    def probar_negativos():
        matriz = np.zeros((3, 3)); matriz[0, 1] = matriz[1, 0] = -1
        esperar_error(lambda: validar_matriz_distancias(matriz))

    @caso("Validaciones", "Matriz: rechazo de NaN e infinitos")
    def probar_no_finitos():
        for valor in (float("nan"), float("inf")):
            matriz = np.zeros((3, 3)); matriz[0, 1] = matriz[1, 0] = valor
            esperar_error(lambda m=matriz: validar_matriz_distancias(m))

    @caso("Validaciones", "Matriz: rechazo de diagonal no nula")
    def probar_diagonal():
        esperar_error(lambda: validar_matriz_distancias(np.eye(3)))

    @caso("Validaciones", "Matriz: rechazo de asimetría y protección de la copia")
    def probar_simetria():
        matriz = np.zeros((3, 3)); matriz[0, 1] = 2
        esperar_error(lambda: validar_matriz_distancias(matriz))
        matriz[1, 0] = 2
        copia = validar_matriz_distancias(matriz)
        exigir(not np.shares_memory(matriz, copia), "No se protegió la matriz original.")

    @caso("Inicialización", "M01: dimensiones y pertenencia de las claves a [0,1)")
    def probar_poblacion():
        poblacion = etapa_1_inicializar(11, 7, np.random.default_rng(11))
        exigir(poblacion.shape == (11, 7), "Forma poblacional incorrecta.")
        exigir(bool(np.all((poblacion >= 0) & (poblacion < 1))), "Claves fuera del dominio.")

    @caso("Inicialización", "M01: repetición exacta con dos generadores de la misma semilla")
    def probar_poblacion_repetible():
        a = etapa_1_inicializar(8, 5, np.random.default_rng(17))
        b = etapa_1_inicializar(8, 5, np.random.default_rng(17))
        exigir(np.array_equal(a, b), "La inicialización no se reprodujo.")

    @caso("Decodificación y evaluación", "Orden de claves del ejemplo y desempate estable")
    def probar_decodificador():
        ruta = decodificar_claves([0.70, 0.10, 0.90, 0.40])
        exigir(np.array_equal(ruta, [0, 2, 1, 3]), "Decodificación del ejemplo incorrecta.")
        exigir(np.array_equal(decodificar_claves([0.5]*4), [0, 1, 2, 3]),
               "El desempate no conserva el orden de los identificadores.")

    @caso("Decodificación y evaluación", "Costo manual con regreso y rechazo de ruta repetida")
    def probar_cierre():
        matriz = np.array([[0, 1, 4, 3], [1, 0, 2, 5], [4, 2, 0, 6], [3, 5, 6, 0]])
        exigir(evaluar_ruta([0, 1, 2, 3], matriz) == 12, "Falta el regreso al origen.")
        esperar_error(lambda: evaluar_ruta([0, 1, 1, 3], matriz))

    @caso("Decodificación y evaluación", "Evaluación por lotes frente al cálculo ruta por ruta")
    def probar_lote():
        matriz, _ = generar_instancia(6, 19)
        poblacion = etapa_1_inicializar(9, 6, np.random.default_rng(21))
        esperado = [evaluar_ruta(decodificar_claves(x), matriz) for x in poblacion]
        observado = etapa_2_evaluar(poblacion, matriz)
        exigir(np.allclose(observado, esperado, rtol=1e-12, atol=1e-10), "Lote desalineado.")

    @caso("Selección", "M12: ganador comprobado con los mismos participantes sorteados")
    def probar_torneo():
        costos = np.array([9.0, 1.0, 4.0, 6.0])
        participantes = np.random.default_rng(23).integers(0, 4, size=(13, 3))
        esperado = participantes[np.arange(13), np.argmin(costos[participantes], axis=1)]
        observado = etapa_3_seleccionar(costos, 13, 3, np.random.default_rng(23))
        exigir(np.array_equal(esperado, observado), "Torneo incorrecto.")

    @caso("Selección", "M12: torneo unitario y primera aparición ante empate")
    def probar_torneo_extremos():
        participantes = np.random.default_rng(25).integers(0, 4, size=(12, 1))
        observado = etapa_3_seleccionar([1, 2, 3, 4], 12, 1, np.random.default_rng(25))
        exigir(np.array_equal(observado, participantes[:, 0]), "k=1 no coincide con el sorteo.")
        participantes = np.random.default_rng(27).integers(0, 4, size=(12, 3))
        observado = etapa_3_seleccionar([2]*4, 12, 3, np.random.default_rng(27))
        exigir(np.array_equal(observado, participantes[:, 0]), "Empate resuelto indebidamente.")

    @caso("Recombinación", "M14: q=0 intercambia las procedencias")
    def probar_cruce_cero():
        a, b = np.array([0.1, 0.2, 0.3]), np.array([0.8, 0.7, 0.6])
        h1, h2, _ = etapa_4_recombinar(a, b, np.random.default_rng(29), 0.0)
        exigir(np.array_equal(h1, b) and np.array_equal(h2, a), "Cruce q=0 incorrecto.")

    @caso("Recombinación", "M14: q=1 conserva las procedencias")
    def probar_cruce_uno():
        a, b = np.array([0.1, 0.2, 0.3]), np.array([0.8, 0.7, 0.6])
        h1, h2, _ = etapa_4_recombinar(a, b, np.random.default_rng(31), 1.0)
        exigir(np.array_equal(h1, a) and np.array_equal(h2, b), "Cruce q=1 incorrecto.")

    @caso("Recombinación", "M14: máscara complementaria, cotas y padres sin alteración")
    def probar_cruce_mascara():
        a, b = np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.8, 0.7, 0.6, 0.5])
        copia_a, copia_b = a.copy(), b.copy()
        h1, h2, mascara = etapa_4_recombinar(a, b, np.random.default_rng(33))
        exigir(np.array_equal(h1, np.where(mascara, a, b)), "Hijo 1 no respeta la máscara.")
        exigir(np.array_equal(h2, np.where(mascara, b, a)), "Hijo 2 no es complementario.")
        exigir(np.array_equal(a, copia_a) and np.array_equal(b, copia_b), "Se alteraron padres.")
        validar_claves(h1, 1); validar_claves(h2, 1)

    @caso("Mutación", "M18: p=0 conserva valores y devuelve una copia independiente")
    def probar_mutacion_cero():
        x = np.array([0.1, 0.2, 0.3, 0.4])
        y = etapa_5_mutar(x, 0, np.random.default_rng(35))
        exigir(np.array_equal(x, y) and not np.shares_memory(x, y), "Mutación nula incorrecta.")

    @caso("Mutación", "M18: p=1 usa todos los nuevos uniformes después de sortear la máscara")
    def probar_mutacion_uno():
        x = np.array([0.1, 0.2, 0.3, 0.4])
        rng = np.random.default_rng(37)
        rng.random(4)  # La primera extracción corresponde a la máscara.
        esperado = rng.random(4)
        observado = etapa_5_mutar(x, 1, np.random.default_rng(37))
        exigir(np.array_equal(esperado, observado), "Se alteró el flujo de mutación.")
        validar_claves(observado, 1)

    def datos_reemplazo():
        # Los costos son datos de contrato explícitos, no resultados del TSP.
        p = np.array([[0.1, 0.2, 0.3], [0.2, 0.4, 0.6], [0.3, 0.6, 0.9]])
        h = np.array([[0.4, 0.3, 0.2], [0.7, 0.5, 0.1]])
        return p, np.array([9.0, 1.0, 4.0]), h, np.array([8.0, 2.0])

    @caso("Reemplazo", "M22: conservación del mejor progenitor")
    def probar_elite():
        p, c, h, ch = datos_reemplazo()
        p2, c2 = etapa_6_reemplazar(p, c, h, ch, 1)
        exigir(c2.min() == 1 and np.array_equal(p2[0], p[1]), "Se perdió la élite.")

    @caso("Reemplazo", "M22: tamaño, alineación de costos y aceptación de todos los hijos")
    def probar_alineacion():
        p, c, h, ch = datos_reemplazo()
        p2, c2 = etapa_6_reemplazar(p, c, h, ch, 1)
        exigir(p2.shape == p.shape and np.array_equal(c2, [1, 8, 2]), "Alineación incorrecta.")
        exigir(np.array_equal(p2[1:], h), "No se incorporó la descendencia completa.")
        exigir(not np.shares_memory(p2, p), "El reemplazo comparte memoria con padres.")

    @caso("Reemplazo", "M22: rechazo de descendencia con cantidad incorrecta")
    def probar_reemplazo_invalido():
        p, c, h, ch = datos_reemplazo()
        esperar_error(lambda: etapa_6_reemplazar(p, c, h[:1], ch[:1], 1))
        esperar_error(lambda: etapa_6_reemplazar(p, c, h, ch, 3))

    @caso("Paro", "M25: frontera del máximo de generaciones")
    def probar_paro():
        exigir(not etapa_7_parar(4, 5) and etapa_7_parar(5, 5), "Frontera incorrecta.")
        exigir(etapa_7_parar(6, 5), "El paro no reconoce un contador superior al máximo.")

    @caso("Paro", "M25: máximo cero y rechazo de contadores negativos")
    def probar_paro_cero():
        exigir(etapa_7_parar(0, 0), "G=0 debe impedir crear descendencia.")
        esperar_error(lambda: etapa_7_parar(-1, 5))
        esperar_error(lambda: etapa_7_parar(0, -1))

    @caso("Integración", "G=0: población inicial, un control y contabilidad independiente")
    def probar_integracion_cero():
        matriz, _ = generar_instancia(5, 39)
        r = algoritmo_genetico(matriz, tamano_poblacion=8, num_generaciones=0,
                               num_elites=1, semilla=41, mostrar_progreso=False)
        exigir(r["generaciones"] == 0 and len(r["historial_mejor"]) == 1, "Historial incorrecto.")
        exigir(r["evaluaciones_busqueda"] == 8 and r["evaluaciones_totales"] == 9,
               "Contabilidad de G=0 incorrecta.")

    @caso("Integración", "Descendencia impar: auditoría, monotonía, presupuesto y repetición")
    def probar_integracion_impar():
        matriz, _ = generar_instancia(5, 43)
        argumentos = dict(tamano_poblacion=8, num_generaciones=7, num_elites=1,
                          semilla=45, mostrar_progreso=False)
        a = algoritmo_genetico(matriz, **argumentos)
        b = algoritmo_genetico(matriz, **argumentos)
        exigir(a["evaluaciones_busqueda"] == 8 + 7*7, "Presupuesto incorrecto.")
        exigir(len(a["historial_mejor"]) == 8, "Historial de longitud incorrecta.")
        exigir(bool(np.all(np.diff(a["historial_mejor"]) <= 0)), "Se perdió la monotonía.")
        for clave in ("mejor_ruta", "mejores_claves", "historial_mejor", "historial_promedio"):
            exigir(np.array_equal(a[clave], b[clave]), f"Repetición fallida: {clave}.")

    exigir(len(casos) == 26, "La batería no tiene los 26 casos declarados.")
    resultados = []
    for numero, (grupo, descripcion, funcion) in enumerate(casos, start=1):
        inicio = perf_counter()
        try:
            funcion()
            estado, detalle = "CORRECTA", ""
        except Exception as error:
            estado, detalle = "FALLIDA", f"{type(error).__name__}: {error}"
        resultados.append({"caso": numero, "grupo": grupo, "descripcion": descripcion,
                           "estado": estado, "detalle": detalle,
                           "segundos": perf_counter() - inicio})
    return resultados


def diagnosticar_corrida(resultado: dict) -> dict:
    """Describe los descensos del historial SIN cambiar la búsqueda realizada.

    Un descenso estricto puede ser menor que 1e-10. Se conservan todos en el
    archivo y se separan los superiores al umbral diagnóstico del artículo.
    La última mejora se determina por la generación del historial, no por una
    selección retrospectiva del número de generaciones que se ejecutará.
    """
    historial = resultado["historial_mejor"]
    descensos = historial[:-1] - historial[1:]
    estrictos = np.flatnonzero(descensos > 0) + 1
    sustantivos = np.flatnonzero(descensos > TOLERANCIA_DIAGNOSTICA) + 1
    diminutos = np.flatnonzero((descensos > 0) & (descensos <= TOLERANCIA_DIAGNOSTICA)) + 1
    ultima = int(sustantivos[-1]) if sustantivos.size else 0
    p = resultado["parametros"]
    return {
        "semilla": p["semilla_algoritmo"], "tolerancia": TOLERANCIA_DIAGNOSTICA,
        "numero_descensos_estrictos": len(estrictos),
        "numero_avances_superiores_al_umbral": len(sustantivos),
        "ultima_generacion_con_avance": ultima,
        "generaciones_descensos_diminutos": diminutos,
        "magnitudes_descensos_diminutos": descensos[diminutos-1],
        "evaluaciones_posteriores_ultimo_avance":
            (resultado["generaciones"]-ultima)*(p["tamano_poblacion"]-p["num_elites"]),
        "eventos": [{"generacion": int(t), "descenso": float(descensos[t-1]),
                     "supera_umbral": bool(descensos[t-1] > TOLERANCIA_DIAGNOSTICA)}
                    for t in estrictos],
    }


def resumir_experimento(resultados: list[dict], matriz: np.ndarray) -> dict:
    """Calcula resúmenes de treinta corridas, sin cargar cifras de referencia.

    Los cuartiles utilizan interpolación lineal y la desviación muestral ddof=1.
    La mediana de reducción se calcula de las treinta reducciones individuales:
    NO es el porcentaje entre la mediana inicial y la mediana final.
    """
    if len(resultados) != 30:
        raise ValueError("El protocolo del artículo requiere exactamente treinta corridas.")
    iniciales = np.array([r["historial_mejor"][0] for r in resultados])
    finales = np.array([r["mejor_distancia"] for r in resultados])
    reducciones = np.zeros_like(finales)
    np.divide(100*(iniciales-finales), iniciales, out=reducciones, where=iniciales > 0)
    q1, mediana, q3 = np.quantile(finales, [0.25, 0.5, 0.75], method="linear")
    return {
        "numero_corridas": len(resultados),
        "minimo": float(finales.min()), "q1": float(q1), "mediana": float(mediana),
        "q3": float(q3), "maximo": float(finales.max()), "media": float(finales.mean()),
        "desviacion_muestral": float(finales.std(ddof=1)),
        "mediana_reduccion_porcentual": float(np.median(reducciones)),
        "minima_reduccion_porcentual": float(reducciones.min()),
        "maxima_reduccion_porcentual": float(reducciones.max()),
        "corridas_con_reduccion": int(np.count_nonzero(finales < iniciales)),
        "evaluaciones_busqueda": sum(r["evaluaciones_busqueda"] for r in resultados),
        "verificaciones_finales": sum(r["evaluaciones_verificacion"] for r in resultados),
        "evaluaciones_totales": sum(r["evaluaciones_totales"] for r in resultados),
        "registros_generacionales": sum(len(r["historial_mejor"]) for r in resultados),
        "mediana_tiempo_busqueda_segundos": float(np.median([r["tiempo_segundos"] for r in resultados])),
        "esperanza_costo_ciclo_uniforme_individual":
            float(2*np.triu(matriz, 1).sum()/(len(matriz)-1)),
        "semilla_menor_costo": resultados[int(finales.argmin())]["parametros"]["semilla_algoritmo"],
        "reducciones_individuales": reducciones,
    }


def ejecutar_treinta_semillas(matriz: np.ndarray, carpeta: Path) -> list[dict]:
    """Realiza las treinta búsquedas y guarda cada una al terminar.

    La matriz se recibe ya generada y NO cambia dentro de este ciclo. Cada llamada
    crea su propio generador con la semilla indicada. Ninguna figura se abre entre
    corridas: cerrar una ventana no es un requisito para ejecutar la siguiente.
    Guardar al terminar cada corrida deja evidencia de lo completado si el usuario
    interrumpe el experimento; una ejecución incompleta NO se declara reproducida.
    """
    if len(SEMILLAS_ALGORITMICAS) != 30 or len(set(SEMILLAS_ALGORITMICAS)) != 30:
        raise ValueError("Se requieren treinta semillas distintas.")
    resultados = []
    for orden, semilla in enumerate(SEMILLAS_ALGORITMICAS, start=1):
        print(f"[{orden:02d}/30] Ejecutando semilla {semilla}...", flush=True)
        r = algoritmo_genetico(matriz, **PARAMETROS_AG, semilla=semilla, mostrar_progreso=False)
        resultados.append(r)
        np.savez_compressed(
            carpeta / f"semilla_{semilla}.npz",
            semilla=np.array(semilla), mejor_ruta=r["mejor_ruta"],
            mejores_claves=r["mejores_claves"], mejor_distancia=np.array(r["mejor_distancia"]),
            historial_mejor=r["historial_mejor"], historial_promedio=r["historial_promedio"],
            evaluaciones_busqueda=np.array(r["evaluaciones_busqueda"]),
            evaluaciones_verificacion=np.array(r["evaluaciones_verificacion"]),
            distancia_verificada=np.array(r["distancia_verificada"]),
            tiempo_segundos=np.array(r["tiempo_segundos"]),
        )
        print(f"        Costo final = {r['mejor_distancia']:.6f}; "
              f"evaluaciones = {r['evaluaciones_busqueda']:,}", flush=True)
    return resultados


def verificar_repetibilidad(matriz: np.ndarray, principal: dict) -> dict:
    """Repite la semilla principal y compara exactamente cuatro arreglos.

    Es la corrida adicional del apartado 3.1 del artículo. No se añade a las
    treinta observaciones y no interviene en medias, cuartiles ni figuras.
    El tiempo se registra, pero no se exige que sea igual entre ejecuciones.
    """
    print("[Auditoría] Repitiendo la semilla 42 fuera de la muestra...", flush=True)
    repetida = algoritmo_genetico(matriz, **PARAMETROS_AG, semilla=SEMILLA_PRINCIPAL,
                                 mostrar_progreso=False)
    comparaciones = {clave: bool(np.array_equal(principal[clave], repetida[clave]))
                    for clave in ("mejor_ruta", "mejores_claves", "historial_mejor", "historial_promedio")}
    return {"semilla": SEMILLA_PRINCIPAL, "igualdades_exactas": comparaciones,
            "correcta": all(comparaciones.values()),
            "incluida_en_muestra_de_30": False,
            "evaluaciones_busqueda_adicionales": repetida["evaluaciones_busqueda"],
            "verificaciones_finales_adicionales": repetida["evaluaciones_verificacion"],
            "tiempo_segundos": repetida["tiempo_segundos"]}

# ============================================================================
# C. LAS SIETE TABLAS: VALORES DOCUMENTALES Y VALORES CALCULADOS, SEPARADOS
# ============================================================================

TITULOS_TABLAS = {
    1: "Método de la guía seleccionado en cada etapa y responsabilidad en el programa.",
    2: "Configuración del ejemplo integrado de la guía y del estudio del agente viajero. Son problemas distintos.",
    3: "Comprobaciones reportadas en las Tablas 3 y 4 de la guía. No son pruebas nuevas del agente viajero.",
    4: "Controles del caso principal, semilla 42. La auditoría final no se incluye en las evaluaciones de búsqueda.",
    5: "Resumen de los treinta costos terminales y de las reducciones relativas. Una instancia y una configuración.",
    6: "Costos de las veinte conexiones del caso principal, incluido el regreso. La suma se verifica antes del redondeo.",
    7: "Resultados por semilla: N=100; G=500; e=2; k=3; p_c=0.90; p_m=0.05.",
}
TITULOS_FIGURAS = {
    1: "Arquitectura de las siete etapas. Esquema redibujado con código a partir del artículo.",
    2: "Mejor costo histórico y promedio poblacional de la corrida principal, semilla 42.",
    3: "Mediana y banda entre los cuartiles 25% y 75% del mejor costo histórico de treinta semillas.",
    4: "Costos finales de todas las semillas. La línea discontinua representa su mediana.",
    5: "Ruta de la semilla 42. La longitud dibujada de una conexión no representa su costo.",
}
NOTAS_TABLAS = {
    1: "Fuente: artículo, Tabla 1. Documentación de los métodos M01, M05, M12, M14, M18, M22 y M25.",
    2: "La columna de la guía es documental. La columna del agente viajero describe el protocolo ejecutado; no se ejecuta la función esfera.",
    3: "Transcripción del antecedente documental: 612 comprobaciones históricas y 309 de la edición ampliada. Estas baterías NO se vuelven a ejecutar aquí.",
    4: "Calculada desde el historial de la corrida 42. B(t)=100+98t. La generación 0 es la población inicial.",
    5: "Calculada con 30 valores terminales. Cuartiles lineales; desviación estándar muestral con denominador 29; cada reducción usa su propio costo inicial.",
    6: "Calculada a partir de la ruta devuelta y de la matriz de costos. El total usa los costos sin redondear; no se suman las cifras impresas.",
    7: "Calculada desde las treinta búsquedas; 49,100 evaluaciones de búsqueda por fila. La repetición adicional de la semilla 42 se excluye.",
}


def construir_tablas(resultados: list[dict], principal: dict, resumen: dict,
                     matriz: np.ndarray) -> dict:
    """Construye los cuerpos de las siete tablas conservando su numeración.

    Solo las tres tablas documentales leen sus filas del texto incorporado del
    artículo. Las filas empíricas 4-7 se construyen exclusivamente con resultados
    de las búsquedas y con la matriz. La comparación con las filas publicadas
    ocurre en OTRA función, después de terminar todos estos cálculos.
    """
    tablas = {}
    for numero in (1, 2):
        original = REFERENCIAS_ARTICULO[str(numero)]
        tablas[numero] = {"encabezados": list(original[0]),
                          "filas": [list(fila) for fila in original[1:]],
                          "procedencia": "Documentación del artículo"}

    original = REFERENCIAS_ARTICULO["3"]
    filas_historicas = [[fila[0], int(fila[1]), int(fila[2])] for fila in original[1:-1]]
    filas_historicas.append(["Total informado", sum(f[1] for f in filas_historicas),
                             sum(f[2] for f in filas_historicas)])
    tablas[3] = {"encabezados": list(original[0]), "filas": filas_historicas,
                 "procedencia": "Antecedente documental; no reejecutado"}

    n, e = PARAMETROS_AG["tamano_poblacion"], PARAMETROS_AG["num_elites"]
    tablas[4] = {
        "encabezados": ["Generación", "Mejor costo", "Promedio poblacional", "Evaluaciones de búsqueda"],
        "filas": [[g, float(principal["historial_mejor"][g]),
                   float(principal["historial_promedio"][g]), n+g*(n-e)] for g in CONTROLES_TABLA_4],
        "procedencia": "Calculada en esta ejecución",
    }
    tablas[5] = {
        "encabezados": ["Estadístico", "Valor"],
        "filas": [
            ["Mínimo del costo final", resumen["minimo"]],
            ["Primer cuartil", resumen["q1"]],
            ["Mediana del costo final", resumen["mediana"]],
            ["Tercer cuartil", resumen["q3"]],
            ["Máximo del costo final", resumen["maximo"]],
            ["Media del costo final", resumen["media"]],
            ["Desviación estándar muestral", resumen["desviacion_muestral"]],
            ["Mediana de reducción relativa", resumen["mediana_reduccion_porcentual"]],
            ["Corridas con reducción del mejor costo inicial",
             f"{resumen['corridas_con_reduccion']} de {resumen['numero_corridas']}"],
        ], "procedencia": "Calculada en esta ejecución",
    }

    ruta = principal["mejor_ruta"]
    siguiente = np.roll(ruta, -1)
    tramos = [[i+1, f"{int(a)} → {int(b)}", float(matriz[a, b])]
              for i, (a, b) in enumerate(zip(ruta, siguiente))]
    exigir(len(tramos) == 20, "La Tabla 6 publicada corresponde a veinte ciudades.")
    costo_sin_redondeo = float(np.sum(matriz[ruta, siguiente]))
    filas_conexiones = [tramos[i] + tramos[i+10] for i in range(10)]
    filas_conexiones.append(["Costo total del ciclo", "", "", "", "", costo_sin_redondeo])
    tablas[6] = {"encabezados": ["Tramo", "Conexión", "Costo", "Tramo", "Conexión", "Costo"],
                 "filas": filas_conexiones, "procedencia": "Calculada en esta ejecución"}

    tablas[7] = {
        "encabezados": ["Semilla", "Mejor costo inicial", "Mejor costo final", "Reducción (%)"],
        "filas": [[r["parametros"]["semilla_algoritmo"], float(r["historial_mejor"][0]),
                   float(r["mejor_distancia"]), float(reduccion)]
                  for r, reduccion in zip(resultados, resumen["reducciones_individuales"])],
        "procedencia": "Calculada en esta ejecución",
    }
    for numero, tabla in tablas.items():
        tabla.update(numero=numero, titulo=TITULOS_TABLAS[numero], nota=NOTAS_TABLAS[numero])
        tabla["filas_presentacion"] = formatear_filas(numero, tabla["filas"])
    return tablas


def formatear_filas(numero: int, filas: list) -> list[list[str]]:
    """Aplica las cifras decimales del artículo solamente al texto visible."""
    presentacion = []
    for indice, fila in enumerate(filas):
        if numero == 4:
            salida = [str(int(fila[0])), f"{fila[1]:.6f}", f"{fila[2]:.6f}", f"{fila[3]:,}"]
        elif numero == 5:
            valor = str(fila[1]) if indice == 8 else f"{fila[1]:.6f}" + ("%" if indice == 7 else "")
            salida = [str(fila[0]), valor]
        elif numero == 6:
            salida = [f"{valor:.6f}" if columna in (2, 5) and isinstance(valor, (float, np.floating))
                      else str(valor) for columna, valor in enumerate(fila)]
        elif numero == 7:
            salida = [str(int(fila[0])), f"{fila[1]:.6f}", f"{fila[2]:.6f}", f"{fila[3]:.3f}"]
        else:
            salida = [str(valor) for valor in fila]
        presentacion.append(salida)
    return presentacion


def comparar_con_articulo(tablas: dict, principal: dict, resumen: dict, diagnostico: dict) -> dict:
    """Contrasta lo calculado con lo publicado sin corregir ni sustituir salidas.

    Las celdas se comparan después de aplicar el mismo redondeo de presentación.
    El CSV de evidencia mantiene más cifras. Una celda discrepante queda marcada;
    jamás se reemplaza para forzar concordancia. Las tablas documentales 1-3 no
    cuentan como resultados empíricos reproducidos.
    """
    comparaciones = []
    for numero in (4, 5, 6, 7):
        publicado = [list(fila) for fila in REFERENCIAS_ARTICULO[str(numero)][1:]]
        # Word almacena repetido el texto de la celda fusionada del total.
        # La normalización elimina esas copias de presentación, no cambia cifras.
        if numero == 6:
            publicado[-1] = [publicado[-1][0], "", "", "", "", publicado[-1][-1]]
        calculado = tablas[numero]["filas_presentacion"]
        exigir(len(publicado) == len(calculado), f"Diferencia de filas en Tabla {numero}.")
        for i, (esperada, observada) in enumerate(zip(publicado, calculado), start=1):
            for j, (a, b) in enumerate(zip(esperada, observada), start=1):
                comparaciones.append({"tabla": numero, "fila": i, "columna": j,
                                      "articulo": a, "recalculado": b, "coincide": a == b})
    # Estos contrastes amplían la revisión a afirmaciones numéricas del texto.
    ruta_publicada = [0, 17, 1, 6, 14, 9, 8, 5, 4, 18, 7, 16, 2, 11, 12, 10, 3, 15, 19, 13]
    extras = {
        "ruta_principal": bool(np.array_equal(principal["mejor_ruta"], ruta_publicada)),
        "evaluaciones_busqueda_30": resumen["evaluaciones_busqueda"] == 1473000,
        "verificaciones_finales_30": resumen["verificaciones_finales"] == 30,
        "registros_generacionales": resumen["registros_generacionales"] == 15030,
        "esperanza_ciclo_uniforme": f"{resumen['esperanza_costo_ciclo_uniforme_individual']:.6f}" == "1020.815520",
        "descensos_estrictos_principal": diagnostico["numero_descensos_estrictos"] == 19,
        "avances_superiores_umbral": diagnostico["numero_avances_superiores_al_umbral"] == 16,
        "ultima_generacion_con_avance": diagnostico["ultima_generacion_con_avance"] == 113,
        "generaciones_descensos_diminutos": bool(np.array_equal(diagnostico["generaciones_descensos_diminutos"], [33, 49, 120])),
        "evaluaciones_posteriores_ultimo_avance": diagnostico["evaluaciones_posteriores_ultimo_avance"] == 37926,
    }
    return {"criterio_celdas": "Coincidencia al redondeo publicado; incluye textos y cifras",
            "celdas_contrastadas": len(comparaciones),
            "celdas_coincidentes": sum(c["coincide"] for c in comparaciones),
            "todas_las_celdas_coinciden": all(c["coincide"] for c in comparaciones),
            "contrastes_adicionales": extras, "todos_los_adicionales_coinciden": all(extras.values()),
            "detalle_celdas": comparaciones,
            "advertencia": "Las referencias no se utilizan para generar los resultados calculados."}


# ============================================================================
# D. FIGURAS Y TABLAS COMO IMÁGENES: CADA UNA EN SU PROPIO ARCHIVO
# ============================================================================


def nueva_figura(ancho: float, alto: float):
    """Crea un lienzo independiente sin abrir una ventana durante el experimento.

    FigureCanvasAgg dibuja en memoria. Esto permite guardar las figuras también
    en equipos sin escritorio gráfico y evita bloquear las treinta corridas.
    Cada figura contiene un único gráfico; no se reúnen en subpaneles.
    """
    figura = Figure(figsize=(ancho, alto))
    FigureCanvasAgg(figura)
    eje = figura.add_subplot(111)
    return figura, eje


def guardar_figura(figura: Figure, carpeta: Path, nombre: str) -> None:
    """Guarda un gráfico calculado en PNG a 300 ppp y SVG vectorial.

    Guarda tanto el esquema construido por código como los gráficos calculados.
    No lee imágenes del artículo. Los datos de curvas, puntos y bandas proceden
    de las funciones de cálculo; la arquitectura es un esquema explicativo.
    """
    figura.savefig(carpeta / f"{nombre}.png", dpi=300)
    figura.savefig(carpeta / f"{nombre}.svg")
    figura.clear()


def dibujar_figura_1(carpeta: Path, nombre: str = "figura_1_arquitectura") -> None:
    """Dibuja la arquitectura mediante texto, recuadros y flechas de Matplotlib.

    No lee una imagen externa, no descarga recursos y no contiene una imagen
    codificada. Cada elemento visible se define aquí con instrucciones legibles.
    El PNG se renderiza al ejecutar; el SVG contiene trazos y texto vectoriales.

    Se conserva la numeración de las siete responsabilidades y se explicita
    debajo la secuencia efectiva, incluida la evaluación de los descendientes.
    Es una nueva representación del esquema, no una copia píxel a píxel de la
    ilustración del artículo. No consume azar ni modifica resultados numéricos.
    """
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    if not isinstance(nombre, str) or not nombre or Path(nombre).name != nombre:
        raise ValueError("nombre debe ser un nombre de archivo simple, sin carpetas.")

    # Las coordenadas siguientes sitúan objetos sobre el lienzo. No representan
    # distancias entre ciudades ni parámetros adicionales de la búsqueda.
    figura, eje = nueva_figura(15.0, 7.4)
    figura.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.98)
    eje.set_xlim(0.0, 14.4)
    eje.set_ylim(0.0, 7.0)
    eje.set_aspect("equal")
    eje.axis("off")
    eje.text(7.2, 6.65, "Figura 1. Arquitectura del algoritmo genético de siete etapas",
             ha="center", va="center", fontsize=16, fontweight="bold")
    eje.text(7.2, 6.22, "Numeración de responsabilidades; la secuencia de ejecución se detalla debajo.",
             ha="center", va="center", fontsize=11)

    # Un registro por etapa: título, método de la guía y responsabilidad local.
    # Las etiquetas documentan la configuración publicada; no alimentan el AG.
    etapas = [
        ("Inicialización", "M01 · Uniforme", "Claves en [0, 1)\nPoblación inicial\nN = 100"),
        ("Evaluación", "M05 · Directa", "Ordenar claves\nCerrar cada ruta\nCalcular su costo"),
        ("Selección", "M12 · Torneo", "Muestreo con\nreemplazo\nk = 3"),
        ("Recombinación", "M14 · Uniforme", "Cruce por máscara\np_c = 0.90\nq = 0.50"),
        ("Mutación", "M18 · Reinicio", "Por cada gen\np_m = 0.05\nClaves en [0, 1)"),
        ("Reemplazo", "M22 · Elitismo", "Conservar 2 élites\nIncorporar 98 hijos\nMantener N = 100"),
        ("Criterio de paro", "M25 · Generaciones", "500 transiciones\nSin paro anticipado\nG = 500"),
    ]
    ancho, alto, separacion = 1.78, 2.18, 2.02
    izquierda, abajo = 0.25, 3.35
    for indice, (titulo, metodo, detalle) in enumerate(etapas):
        x = izquierda + indice * separacion
        centro = x + ancho / 2.0
        # fill=False deja los recuadros sin relleno; no se importa ningún icono.
        eje.add_patch(FancyBboxPatch((x, abajo), ancho, alto,
                      boxstyle="round,pad=0.02,rounding_size=0.10", fill=False,
                      linewidth=1.2))
        eje.add_patch(Circle((centro, abajo + alto + 0.16), radius=0.18,
                      fill=False, linewidth=1.2))
        eje.text(centro, abajo + alto + 0.16, str(indice + 1),
                 ha="center", va="center", fontsize=11, fontweight="bold")
        eje.text(centro, 5.17, titulo, ha="center", va="center",
                 fontsize=10.5, fontweight="bold")
        eje.text(centro, 4.72, metodo, ha="center", va="center", fontsize=10)
        eje.text(centro, 4.03, detalle, ha="center", va="center",
                 fontsize=10.2, linespacing=1.6)
        if indice < len(etapas) - 1:
            # Las flechas de esta fila enlazan las responsabilidades numeradas.
            # No sustituyen la secuencia efectiva, que repite la evaluación.
            eje.add_patch(FancyArrowPatch((x + ancho + 0.035, 4.43),
                          (x + separacion - 0.035, 4.43), arrowstyle="->",
                          mutation_scale=11, linewidth=1.1, shrinkA=0, shrinkB=0))

    eje.text(7.2, 2.87, "Secuencia efectiva del programa", ha="center", va="center",
             fontsize=12, fontweight="bold")
    eje.add_patch(FancyBboxPatch((0.25, 0.99), 13.90, 1.55,
                  boxstyle="round,pad=0.02,rounding_size=0.10", fill=False,
                  linewidth=1.1))
    # Consultar el paro antes de crear hijos evita una generación excedente.
    # Reevaluar los hijos impide reutilizar costos anteriores a cruce o mutación.
    eje.text(7.2, 2.18,
             "Inicio: 1. Inicializar → 2. Evaluar población → 7. Comprobar paro",
             ha="center", va="center", fontsize=11)
    eje.text(7.2, 1.76,
             "Si continúa: 3. Seleccionar → 4. Recombinar → 5. Mutar → 2. Evaluar hijos → 6. Reemplazar",
             ha="center", va="center", fontsize=11)
    eje.text(7.2, 1.34,
             "Después: actualizar generación e historial → 7. Comprobar paro → repetir o finalizar",
             ha="center", va="center", fontsize=11)
    eje.text(7.2, 0.55,
             "Reevaluar hijos es repetir la etapa 2. El elitismo conserva lo encontrado, pero no certifica optimalidad.",
             ha="center", va="center", fontsize=10.5)
    # La opción local mantiene texto editable en el SVG de esta figura sin
    # modificar el estilo o los archivos producidos para las Figuras 2 a 5.
    with matplotlib.rc_context({"svg.fonttype": "none"}):
        guardar_figura(figura, carpeta, nombre)


def generar_figuras(resultados: list[dict], principal: dict,
                     coordenadas: np.ndarray, carpeta: Path) -> dict[int, str]:
    """Reproduce las cinco figuras manteniendo su significado y numeración.

    Figura 1 se dibuja con instrucciones legibles mediante dibujar_figura_1.
    Su diseño no es idéntico al original. Figuras 2-5 se vuelven a dibujar con
    datos de las treinta corridas actuales. La banda de la
    Figura 3 es intercuartílica: NO es un intervalo de confianza ni diversidad
    de individuos de una misma población.
    """
    nombres = {
        1: "figura_1_arquitectura",
        2: "figura_2_trayectoria_principal",
        3: "figura_3_mediana_cuartiles",
        4: "figura_4_costos_finales",
        5: "figura_5_ruta_principal",
    }
    dibujar_figura_1(carpeta, nombres[1])

    generaciones = np.arange(len(principal["historial_mejor"]))
    figura, eje = nueva_figura(8, 4.1)
    eje.plot(generaciones, principal["historial_mejor"], label="Mejor costo histórico")
    eje.plot(generaciones, principal["historial_promedio"], label="Promedio poblacional")
    eje.set(xlabel="Generación", ylabel="Costo del recorrido",
            title="Caso principal: semilla algorítmica 42")
    eje.grid(alpha=0.25); eje.legend(); figura.tight_layout()
    guardar_figura(figura, carpeta, nombres[2])

    historiales = np.stack([r["historial_mejor"] for r in resultados])
    q1, mediana, q3 = np.quantile(historiales, [0.25, 0.50, 0.75], axis=0, method="linear")
    figura, eje = nueva_figura(8, 4.1)
    eje.fill_between(generaciones, q1, q3, alpha=0.2, label="Intervalo entre cuartiles")
    eje.plot(generaciones, mediana, label="Mediana de 30 semillas")
    eje.set(xlabel="Generación", ylabel="Mejor costo histórico",
            title="Una matriz de distancias; treinta ejecuciones")
    eje.grid(alpha=0.25); eje.legend(); figura.tight_layout()
    guardar_figura(figura, carpeta, nombres[3])

    semillas = [r["parametros"]["semilla_algoritmo"] for r in resultados]
    finales = [r["mejor_distancia"] for r in resultados]
    figura, eje = nueva_figura(8, 3.8)
    eje.scatter(semillas, finales, label="Costo final por semilla")
    eje.axhline(np.median(finales), linestyle="--", label="Mediana de los costos finales")
    eje.set(xlabel="Semilla algorítmica", ylabel="Costo final",
            title="Variabilidad terminal: semillas 42 a 71")
    eje.grid(alpha=0.25); eje.legend(); figura.tight_layout()
    guardar_figura(figura, carpeta, nombres[4])

    ruta = principal["mejor_ruta"]
    cerrada = np.append(ruta, ruta[0])
    figura, eje = nueva_figura(6.2, 5.5)
    eje.plot(coordenadas[cerrada, 0], coordenadas[cerrada, 1], "-o",
             linewidth=1.25, markersize=4)
    for ciudad, (x, y) in enumerate(coordenadas):
        eje.annotate(str(ciudad), (x, y), xytext=(5, 5), textcoords="offset points")
    eje.scatter(coordenadas[0, 0], coordenadas[0, 1], marker="*", s=150,
                label="Inicio y regreso: ciudad 0")
    eje.set_title("Recorrido de la semilla 42\nEsquema circular, no mapa geográfico")
    eje.set_aspect("equal"); eje.margins(0.17); eje.axis("off")
    eje.legend(loc="lower center", bbox_to_anchor=(0.5, -0.04))
    figura.tight_layout()
    guardar_figura(figura, carpeta, nombres[5])
    return nombres


def generar_imagen_tabla(tabla: dict, carpeta: Path) -> str:
    """Exporta una tabla completa, incluidas las treinta filas de la Tabla 7.

    Se ajusta la altura de cada fila al texto que contiene para evitar recortes.
    Las tablas no sustituyen sus CSV: la imagen es presentación y el CSV conserva
    los valores numéricos. El diseño usa reglas horizontales como el artículo.
    """
    numero = tabla["numero"]
    anchos = {1: [0.19, 0.24, 0.57], 2: [0.22, 0.39, 0.39], 3: [0.58, 0.21, 0.21],
              4: [0.15, 0.24, 0.29, 0.32], 5: [0.75, 0.25],
              6: [0.085, 0.225, 0.19, 0.085, 0.225, 0.19], 7: [0.12, 0.30, 0.30, 0.28]}[numero]
    ancho_figura = 11.5
    tamanio_fuente = 10.5
    capacidad = [max(8, int(155*ancho)) for ancho in anchos]

    def envolver(fila):
        return ["\n".join(textwrap.wrap(str(texto), width=capacidad[j],
                                         break_long_words=False, break_on_hyphens=False))
                for j, texto in enumerate(fila)]

    filas = [envolver(tabla["encabezados"])] + [envolver(f) for f in tabla["filas_presentacion"]]
    if numero == 6:
        filas[-1][0] = "Costo total del ciclo"  # Equivale al rótulo de la celda fusionada.
    alturas = [0.18*max(texto.count("\n")+1 for texto in fila)+0.16 for fila in filas]
    titulo = textwrap.fill(f"Tabla {numero}. {tabla['titulo']}", width=115)
    nota = textwrap.fill(tabla["nota"], width=132)
    alto_titulo = 0.23*(titulo.count("\n")+1)+0.28
    alto_nota = 0.18*(nota.count("\n")+1)+0.26
    alto_cuerpo = sum(alturas)
    alto_figura = alto_titulo + alto_cuerpo + alto_nota
    figura = Figure(figsize=(ancho_figura, alto_figura))
    FigureCanvasAgg(figura)
    eje = figura.add_axes([0.025, alto_nota/alto_figura, 0.95, alto_cuerpo/alto_figura])
    eje.axis("off")
    tabla_mpl = eje.table(cellText=filas, colWidths=anchos, cellLoc="left", bbox=[0, 0, 1, 1], edges="open")
    tabla_mpl.auto_set_font_size(False)
    tabla_mpl.set_fontsize(tamanio_fuente)
    columnas_numericas = {1: (), 2: (), 3: (1, 2), 4: (0, 1, 2, 3), 5: (1,),
                          6: (0, 2, 3, 5), 7: (0, 1, 2, 3)}[numero]
    for (fila, columna), celda in tabla_mpl.get_celld().items():
        celda.set_height(alturas[fila]/alto_cuerpo)
        celda.PAD = 0.055
        texto = celda.get_text()
        texto.set_fontfamily("DejaVu Serif")
        texto.set_va("center")
        if columna in columnas_numericas:
            texto.set_ha("right")
        if fila == 0:
            texto.set_weight("bold"); celda.visible_edges = "BT"; celda.set_linewidth(1.1)
        elif fila == len(filas)-1:
            celda.visible_edges = "B"; celda.set_linewidth(1.1)
            if numero in (3, 6):
                celda.visible_edges = "TB"; texto.set_weight("bold")
    # El total de Tabla 6 ocupa varias columnas en Word. Aquí se deja el espacio
    # libre equivalente sin repetir el texto de la celda fusionada.
    if numero == 6:
        texto_total = tabla_mpl[(len(filas)-1, 0)].get_text()
        texto_total.set_text("Costo total del ciclo")
        texto_total.set_ha("left")
    figura.text(0.025, 1-0.18/alto_figura, titulo, va="top", fontsize=11,
                fontfamily="DejaVu Serif", fontweight="bold")
    figura.text(0.025, 0.12/alto_figura, nota, va="bottom", fontsize=9,
                fontfamily="DejaVu Serif")
    nombre = f"tabla_{numero}"
    guardar_figura(figura, carpeta, nombre)
    return nombre

# ============================================================================
# E. DATOS COMPLETOS, INFORME LOCAL Y REGISTRO DE PROCEDENCIA
# ============================================================================


def exportar_datos(resultados: list[dict], principal: dict, resumen: dict,
                   matriz: np.ndarray, coordenadas: np.ndarray, carpeta: Path) -> None:
    """Guarda datos suficientes para recalcular tablas y volver a trazar curvas.

    El CSV de historiales contiene 30*501=15,030 filas. El NPZ guarda las mismas
    trayectorias como matrices y conserva las rutas y claves de cada corrida.
    Los archivos no contienen objetos Python serializados: pueden abrirse con
    np.load(..., allow_pickle=False). Los costos no se redondean a seis decimales.
    """
    semillas = np.array([r["parametros"]["semilla_algoritmo"] for r in resultados], dtype=int)
    mejores = np.stack([r["historial_mejor"] for r in resultados])
    promedios = np.stack([r["historial_promedio"] for r in resultados])
    rutas = np.stack([r["mejor_ruta"] for r in resultados])
    claves = np.stack([r["mejores_claves"] for r in resultados])
    np.savetxt(carpeta / "matriz_distancias.csv", matriz, delimiter=",", fmt="%.17g")
    np.savetxt(carpeta / "coordenadas_esquema.csv", coordenadas, delimiter=",", fmt="%.17g")
    np.savez_compressed(carpeta / "historiales_30_semillas.npz", semillas=semillas,
                        mejor=mejores, promedio=promedios, rutas=rutas, claves=claves,
                        matriz_distancias=matriz, coordenadas=coordenadas,
                        reducciones=resumen["reducciones_individuales"])
    np.savez_compressed(carpeta / "caso_principal.npz", semilla=np.array(SEMILLA_PRINCIPAL),
                        mejor_ruta=principal["mejor_ruta"], mejores_claves=principal["mejores_claves"],
                        historial_mejor=principal["historial_mejor"],
                        historial_promedio=principal["historial_promedio"])
    n, e = PARAMETROS_AG["tamano_poblacion"], PARAMETROS_AG["num_elites"]
    filas_historial = ((int(s), g, float(mejor), float(promedio), n+g*(n-e))
                      for s, hm, hp in zip(semillas, mejores, promedios)
                      for g, (mejor, promedio) in enumerate(zip(hm, hp)))
    guardar_csv(carpeta / "historiales_30_semillas.csv",
                ["semilla", "generacion", "mejor_costo", "promedio_poblacional", "evaluaciones_busqueda"],
                filas_historial)
    q1, mediana, q3 = np.quantile(mejores, [0.25, 0.50, 0.75], axis=0, method="linear")
    guardar_csv(carpeta / "datos_figura_3.csv", ["generacion", "q25", "mediana", "q75"],
                ((g, float(a), float(b), float(c)) for g, (a, b, c) in enumerate(zip(q1, mediana, q3))))
    filas_corridas = []
    filas_tramos = []
    diagnosticos = []
    for r, reduccion in zip(resultados, resumen["reducciones_individuales"]):
        semilla = r["parametros"]["semilla_algoritmo"]
        d = diagnosticar_corrida(r)
        diagnosticos.append(d)
        filas_corridas.append([
            semilla, float(r["historial_mejor"][0]), r["mejor_distancia"], float(reduccion),
            r["generaciones"], r["evaluaciones_busqueda"], r["evaluaciones_verificacion"],
            r["evaluaciones_totales"], r["tiempo_segundos"], r["distancia_verificada"],
            d["numero_descensos_estrictos"], d["numero_avances_superiores_al_umbral"],
            d["ultima_generacion_con_avance"], " -> ".join(map(str, np.append(r["mejor_ruta"], 0))),
        ])
        for indice, (a, b) in enumerate(zip(r["mejor_ruta"], np.roll(r["mejor_ruta"], -1)), start=1):
            filas_tramos.append([semilla, indice, int(a), int(b), float(matriz[a, b])])
    guardar_csv(carpeta / "resumen_por_semilla.csv",
                ["semilla", "mejor_costo_inicial", "mejor_costo_final", "reduccion_porcentaje",
                 "generaciones", "evaluaciones_busqueda", "evaluaciones_verificacion", "evaluaciones_totales",
                 "tiempo_segundos", "distancia_verificada", "descensos_estrictos", "avances_superiores_umbral",
                 "ultima_generacion_con_avance", "ruta_cerrada"], filas_corridas)
    guardar_csv(carpeta / "tramos_todas_las_rutas.csv", ["semilla", "tramo", "origen", "destino", "costo"], filas_tramos)
    guardar_json(carpeta / "diagnosticos_todas_las_semillas.json", diagnosticos)
    guardar_json(carpeta / "estadisticas.json", resumen)
    guardar_json(carpeta / "antecedentes_documentales_guia.json", {
        "procedencia": "Guía de Aguilar Ortiz; apartado 9.3; retomada en el apartado 5.1 del artículo",
        "estado": "REPORTADO POR LA FUENTE; NO REEJECUTADO POR ESTE ARCHIVO",
        "tabla_3_del_articulo": REFERENCIAS_ARTICULO["3"],
        "ejemplo_esfera_reportado": {"semilla": 20260917, "costo": 1.6124001222436235e-5,
                                     "generaciones": 28, "evaluaciones": 928, "meta": 1e-6,
                                     "motivo": "dispersión poblacional pequeña"},
    })


def tabla_html(tabla: dict) -> str:
    """Crea una tabla HTML accesible con el mismo contenido que su PNG."""
    encabezado = "".join(f"<th scope='col'>{html.escape(t)}</th>" for t in tabla["encabezados"])
    filas = []
    for indice, fila in enumerate(tabla["filas_presentacion"]):
        if tabla["numero"] == 6 and indice == len(tabla["filas_presentacion"])-1:
            cuerpo = f"<th colspan='5' scope='row'>{html.escape(fila[0])}</th><td>{html.escape(fila[-1])}</td>"
        else:
            cuerpo = "".join(f"<td>{html.escape(celda)}</td>" for celda in fila)
        filas.append(f"<tr>{cuerpo}</tr>")
    return f"<div class='desplazable'><table><thead><tr>{encabezado}</tr></thead><tbody>{''.join(filas)}</tbody></table></div>"


def escribir_informe(carpeta: Path, tablas: dict, figuras: dict, resumen: dict,
                      pruebas: list[dict], repetibilidad: dict, contraste: dict,
                      principal: dict, diagnostico: dict, entorno: dict) -> Path:
    """Reúne todas las salidas en un informe que se abre localmente sin servidor.

    Los enlaces son relativos a esta carpeta. Para transportar el informe basta
    conservarla completa. Las figuras y tablas siguen teniendo sus archivos
    independientes para insertarlas en Word o utilizarlas en otra aplicación.
    """
    coincide = contraste["todas_las_celdas_coinciden"] and contraste["todos_los_adicionales_coinciden"]
    estado = "COINCIDE CON LAS CIFRAS DEL ARTÍCULO" if coincide else "SE DETECTARON DIFERENCIAS CON EL ARTÍCULO"
    navegacion = "".join(f"<a href='#tabla-{n}'>Tabla {n}</a>" for n in range(1, 8))
    navegacion += "".join(f"<a href='#figura-{n}'>Figura {n}</a>" for n in range(1, 6))
    secciones = {}
    for n, tabla in tablas.items():
        secciones[f"t{n}"] = (
            f"<section id='tabla-{n}'><div class='etiqueta'>{html.escape(tabla['procedencia'])}</div>"
            f"<h2>Tabla {n}. {html.escape(tabla['titulo'])}</h2>{tabla_html(tabla)}"
            f"<p class='nota'>{html.escape(tabla['nota'])}</p>"
            f"<p class='archivos'><a href='tablas/tabla_{n}.csv'>CSV: precisión completa</a> · "
            f"<a href='tablas/tabla_{n}_presentacion.csv'>CSV: redondeo del artículo</a> · "
            f"<a href='tablas/tabla_{n}.png'>Imagen PNG</a> · <a href='tablas/tabla_{n}.svg'>SVG</a></p></section>"
        )
    for n, nombre in figuras.items():
        tipo = "Esquema redibujado con código" if n == 1 else "Figura calculada con esta ejecución"
        nota = {1: "Esquema generado con texto, recuadros y flechas de Matplotlib, sin imágenes incrustadas. Conserva las siete responsabilidades, pero no es una copia visual exacta de la ilustración del artículo. El SVG es vectorial. La evaluación también se aplica a los hijos después de la variación.",
                2: "La generación cero incluye la población inicial. La mejora del costo no certifica optimalidad.",
                3: "Banda intercuartílica entre corridas, no intervalo de confianza ni diversidad intrapoblacional.",
                4: "Se muestran las treinta semillas sin excluir costos desfavorables. La corrida 42 no fue seleccionada por ser la mejor.",
                5: "Esquema circular, no mapa geográfico. La Tabla 6 contiene los costos reales de la instancia sintética."}[n]
        secciones[f"f{n}"] = (
            f"<section id='figura-{n}'><div class='etiqueta'>{tipo}</div>"
            f"<h2>Figura {n}. {html.escape(TITULOS_FIGURAS[n])}</h2>"
            f"<figure><img src='figuras/{nombre}.png' alt='{html.escape(TITULOS_FIGURAS[n], quote=True)}'/>"
            f"<figcaption>{html.escape(nota)}</figcaption></figure>"
            f"<p class='archivos'><a href='figuras/{nombre}.png'>PNG</a> · <a href='figuras/{nombre}.svg'>SVG</a></p></section>"
        )
    orden = ("f1", "t1", "t2", "t3", "f2", "t4", "t5", "f3", "f4", "f5", "t6", "t7")
    cuerpo = "".join(secciones[k] for k in orden)
    num_correctas = sum(p["estado"] == "CORRECTA" for p in pruebas)
    ruta = " → ".join(map(str, np.append(principal["mejor_ruta"], 0)))
    inicio = float(principal["historial_mejor"][0])
    reduccion = 100*(inicio-principal["mejor_distancia"])/inicio if inicio else 0.0
    documento = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reproducción completa · Agente viajero · 30 semillas</title>
<style>
:root {{ color-scheme:light; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:#f4f3f6; color:#23212a; font:17px/1.55 Georgia,"Times New Roman",serif; }}
main {{ max-width:1160px; margin:24px auto; padding:36px 42px; background:white; }}
h1,h2,h3 {{ color:#542781; line-height:1.25; }}
h1 {{ font-size:34px; margin:10px 0; }} h2 {{ font-size:22px; margin:8px 0 18px; }}
.autor,.etiqueta,.archivos,nav,.tarjetas {{ font-family:Arial,sans-serif; }}
.etiqueta {{ text-transform:uppercase; letter-spacing:.05em; font-size:12px; color:#5e566a; }}
.subtitulo {{ font-size:20px; }}
nav {{ display:flex; flex-wrap:wrap; gap:8px; padding:18px 0; border-bottom:1px solid #dcd6e4; }}
nav a {{ padding:6px 10px; border:1px solid #ded7e8; text-decoration:none; font-size:14px; }}
a {{ color:#542781; }}
.tarjetas {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:22px 0; }}
.tarjeta {{ padding:14px 16px; background:#f4eff9; border-top:3px solid #542781; }}
.tarjeta strong {{ display:block; font-size:23px; }} .tarjeta span {{ font-size:13px; }}
.aviso {{ border-left:4px solid #542781; background:#f8f5fb; padding:16px 20px; margin:20px 0; }}
section {{ margin-top:36px; padding-top:16px; border-top:1px solid #ded7e8; scroll-margin-top:15px; }}
table {{ width:100%; border-collapse:collapse; font-size:15px; border-top:2px solid #222; border-bottom:2px solid #222; }}
thead {{ border-bottom:1px solid #222; }} th,td {{ padding:8px 10px; text-align:left; vertical-align:middle; }}
th {{ font-weight:bold; }} td {{ font-variant-numeric:tabular-nums; }}
#tabla-3 td:not(:first-child),#tabla-4 td,#tabla-5 td:last-child,#tabla-6 td,#tabla-7 td {{ text-align:right; }}
#tabla-6 td:nth-child(2),#tabla-6 td:nth-child(5) {{ text-align:center; }}
.desplazable {{ overflow-x:auto; }}
figure {{ margin:8px 0; text-align:center; }} figure img {{ max-width:100%; height:auto; max-height:690px; }}
figcaption,.nota {{ font-size:14px; color:#514959; }} .archivos {{ font-size:14px; }}
.codigo {{ font-family:Consolas,monospace; overflow-wrap:anywhere; font-size:14px; background:#f6f5f7; padding:12px; }}
.pie {{ margin-top:35px; padding-top:15px; border-top:1px solid #ded7e8; font-size:14px; }}
@media(max-width:700px) {{ main {{ margin:0; padding:22px 16px; }} h1 {{ font-size:27px; }} .tarjetas {{ grid-template-columns:1fr; }} }}
@media print {{ body {{ background:white; }} main {{ margin:0; padding:0; }} nav,.archivos {{ display:none; }} section {{ break-inside:avoid; }} table {{ font-size:11px; }} }}
</style></head><body><main>
<div class="etiqueta">Artículo de investigación computacional · Evidencia reproducible</div>
<h1>Agente viajero: las treinta semillas, las siete tablas y las cinco figuras</h1>
<p class="autor">Jaime Aguilar Ortiz</p>
<p class="subtitulo">Una matriz sintética de veinte ciudades. Siete etapas genéticas. Resultados recalculados.</p>
<nav aria-label="Tablas y figuras">{navegacion}</nav>
<div class="tarjetas">
<div class="tarjeta"><strong>{resumen['numero_corridas']}</strong><span>corridas: semillas 42 a 71</span></div>
<div class="tarjeta"><strong>{resumen['mediana']:.6f}</strong><span>mediana del costo terminal</span></div>
<div class="tarjeta"><strong>{resumen['evaluaciones_busqueda']:,}</strong><span>evaluaciones de búsqueda en la muestra</span></div>
</div>
<div class="aviso"><strong>{estado}</strong><br>
Se compararon {contraste['celdas_contrastadas']} celdas, incluidos textos y cifras, de las Tablas 4–7:
{contraste['celdas_coincidentes']} coinciden al redondeo publicado.
Los diez contrastes adicionales del texto se conservan en la auditoría.
La comparación no altera los datos calculados.</div>
<p><strong>Separación de evidencia.</strong> Las Tablas 1 y 2 describen métodos y configuraciones.
La Tabla 3 transcribe las comprobaciones históricas de la guía: no se ejecutaron de nuevo las baterías
612 y 309. La Figura 1 redibuja las siete etapas mediante código, con una composición nueva.
Las Tablas 4–7 y las Figuras 2–5 proceden de
las treinta búsquedas ejecutadas para este informe, no de valores copiados del artículo.</p>
<p><strong>Caso principal.</strong> Semilla 42: {inicio:.6f} → {principal['mejor_distancia']:.6f},
reducción de {reduccion:.6f}% respecto del mejor costo inicial. Ruta: {html.escape(ruta)}.</p>
<p><strong>Verificación actual.</strong> {num_correctas} de 26 casos funcionales correctos.
Repetición exacta de ruta, claves e historiales: {'CORRECTA' if repetibilidad['correcta'] else 'FALLIDA'}.
La repetición adicional de 42 se excluye de las estadísticas y requiere otras
{repetibilidad['evaluaciones_busqueda_adicionales']:,} evaluaciones de búsqueda y una auditoría final.</p>
<p><strong>Diagnóstico de la meseta.</strong> Se observaron {diagnostico['numero_descensos_estrictos']} descensos estrictos;
{diagnostico['numero_avances_superiores_al_umbral']} superaron 10⁻¹⁰. El último avance superior al umbral
ocurrió en la generación {diagnostico['ultima_generacion_con_avance']}. Este diagnóstico no detuvo la búsqueda.</p>
<p><strong>Entorno de esta ejecución.</strong> Python {html.escape(entorno['python'])};
NumPy {html.escape(entorno['numpy'])}; Matplotlib {html.escape(entorno['matplotlib'])};
{html.escape(entorno['sistema'])}. Mediana del tiempo medido por la búsqueda:
{resumen['mediana_tiempo_busqueda_segundos']:.3f} s. El tiempo no incluye generación de tablas o figuras
ni se exige que coincida con el del artículo.</p>
{cuerpo}
<section id="auditoria"><h2>Datos y auditoría</h2>
<p><a href="datos/historiales_30_semillas.csv">15,030 controles generacionales</a> ·
<a href="datos/historiales_30_semillas.npz">Matrices y trayectorias NPZ</a> ·
<a href="datos/resumen_por_semilla.csv">Datos completos por semilla</a> ·
<a href="datos/datos_figura_3.csv">Cuartiles de la Figura 3</a></p>
<p><a href="auditoria/comparacion_con_articulo.csv">Comparación celda por celda</a> ·
<a href="auditoria/pruebas_funcionales_26.csv">Detalle de los 26 casos</a> ·
<a href="auditoria/repetibilidad.json">Repetición de la semilla 42</a> ·
<a href="manifiesto.json">Manifiesto y huellas SHA-256</a></p>
<p class="nota">Los 26 casos incluidos se han construido según las categorías descritas en el artículo.
No se afirma identidad de código con una batería histórica que no acompaña al documento.</p>
</section>
<div class="pie">Alcance: una instancia sintética y una configuración. No se certifica optimalidad,
no se establece superioridad frente a otros algoritmos y no se miden efectos educativos.
Las referencias documentales proceden del artículo y de la guía aportados por el autor.</div>
</main></body></html>"""
    ruta_salida = carpeta / "INFORME_REPRODUCCION.html"
    ruta_salida.write_text(documento, encoding="utf-8")
    return ruta_salida


def crear_manifiesto(carpeta: Path, entorno: dict, resumen: dict,
                     contraste: dict, repetibilidad: dict) -> None:
    """Registra procedencia, configuración y huellas de los archivos entregados.

    La huella identifica bytes, no calidad científica. El propio manifiesto se
    excluye de la lista para evitar una dependencia circular. Las fechas, tiempos,
    versiones y SVG pueden variar aunque los resultados numéricos coincidan.
    """
    archivos = []
    for ruta in sorted(carpeta.rglob("*")):
        if ruta.is_file() and ruta.name != "manifiesto.json":
            archivos.append({"archivo": ruta.relative_to(carpeta).as_posix(),
                             "bytes": ruta.stat().st_size,
                             "sha256": hashlib.sha256(ruta.read_bytes()).hexdigest()})
    archivo_codigo = Path(__file__).resolve() if "__file__" in globals() else None
    guardar_json(carpeta / "manifiesto.json", {
        "creado_utc": datetime.now(timezone.utc).isoformat(), "entorno": entorno,
        "codigo_reproductor": archivo_codigo.name if archivo_codigo else "sesion_interactiva",
        "sha256_codigo_reproductor": hashlib.sha256(archivo_codigo.read_bytes()).hexdigest() if archivo_codigo and archivo_codigo.is_file() else None,
        "sha256_codigo_base_del_articulo": SHA256_CODIGO_BASE,
        "sha256_articulo_fuente": SHA256_ARTICULO,
        "semillas_algoritmicas": SEMILLAS_ALGORITMICAS, "semilla_datos": SEMILLA_DATOS,
        "parametros": PARAMETROS_AG, "numero_ciudades": NUM_CIUDADES,
        "muestra": {"corridas": resumen["numero_corridas"],
                    "registros_generacionales": resumen["registros_generacionales"],
                    "evaluaciones_busqueda": resumen["evaluaciones_busqueda"],
                    "verificaciones_finales": resumen["verificaciones_finales"]},
        "repeticion_fuera_de_muestra": repetibilidad,
        "pruebas_funcionales": "26 casos incluidos; separados de las baterías históricas 612 y 309",
        "coincidencia_celdas_empiricas": contraste["todas_las_celdas_coinciden"],
        "coincidencia_contrastes_adicionales": contraste["todos_los_adicionales_coinciden"],
        "figura_1": "Esquema redibujado con Matplotlib; PNG renderizado y SVG vectorial. No es copia visual exacta del original.",
        "figuras_2_a_5": "Gráficos calculados; PNG a 300 ppp y SVG vectorial",
        "tablas_1_a_3": "Documentación o antecedentes; no resultados de treinta búsquedas",
        "tablas_4_a_7": "Datos calculados en esta ejecución",
        "archivos": archivos,
    })


def validar_protocolo_publicado() -> None:
    """Evita etiquetar como reproducción un experimento de parámetros distintos.

    El algoritmo genético puede reutilizarse con otros parámetros llamando a su
    función directamente. Este controlador, en cambio, se dedica al artículo y
    exige su configuración para que las Tablas 1 y 2 no documenten otro estudio.
    """
    esperado = {"tamano_poblacion": 100, "tasa_mutacion": 0.05, "num_generaciones": 500,
                "num_elites": 2, "tamano_torneo": 3, "probabilidad_cruce": 0.90}
    if (PARAMETROS_AG != esperado or NUM_CIUDADES != 20 or SEMILLA_DATOS != 20260917
            or SEMILLAS_ALGORITMICAS != tuple(range(42, 72)) or SEMILLA_PRINCIPAL != 42):
        raise ValueError("Para reproducir el artículo deben conservarse su configuración y sus treinta semillas 42–71.")
    if CONTROLES_TABLA_4 != (0, 25, 50, 100, 113, 120, 250, 500) or TOLERANCIA_DIAGNOSTICA != 1e-10:
        raise ValueError("Los controles de la Tabla 4 o el umbral diagnóstico difieren del artículo.")


def ejecutar_reproduccion(carpeta_base=None, abrir_informe=True, mostrar_figuras=False) -> dict:
    """Orquesta las ejecuciones, comprueba la evidencia y produce todos los archivos.

    Esta función es la ampliación que faltaba en el código anterior. Se conserva
    el núcleo numérico y se añade alrededor de él el experimento de treinta
    semillas, la estadística y la producción de las siete tablas y cinco figuras.
    """
    validar_protocolo_publicado()
    base = Path(carpeta_base).expanduser() if carpeta_base else (
        Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd())
    base.mkdir(parents=True, exist_ok=True)
    carpeta = base / ("AG_30_SEMILLAS_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    carpeta.mkdir(exist_ok=False)
    for nombre in ("tablas", "figuras", "datos", "auditoria", "corridas"):
        (carpeta / nombre).mkdir()
    carpeta = carpeta.resolve()
    print("\nREPRODUCCIÓN COMPLETA DEL ARTÍCULO DEL AGENTE VIAJERO", flush=True)
    print("Carpeta de salida:", carpeta, flush=True)
    print("Tabla 3: antecedente documental. Tablas 4–7: cálculo nuevo.\n", flush=True)
    inicio_completo = perf_counter()
    guardar_json(carpeta / "ESTADO.json", {"estado": "EN_EJECUCION", "completo": False})
    try:
        entorno = {"python": platform.python_version(), "numpy": np.__version__,
                   "matplotlib": matplotlib.__version__, "sistema": platform.system()+" "+platform.machine(),
                   "generador": type(np.random.default_rng(0).bit_generator).__name__,
                   "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                   "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")}
        guardar_json(carpeta / "datos" / "configuracion.json", {
            "semilla_datos": SEMILLA_DATOS, "semillas_algoritmicas": SEMILLAS_ALGORITMICAS,
            "semilla_principal": SEMILLA_PRINCIPAL, "numero_ciudades": NUM_CIUDADES,
            "parametros": PARAMETROS_AG, "modo_instancia": "simetrica_aleatoria", "entorno": entorno})
        pruebas = ejecutar_pruebas_funcionales()
        guardar_json(carpeta / "auditoria" / "pruebas_funcionales_26.json", pruebas)
        guardar_csv(carpeta / "auditoria" / "pruebas_funcionales_26.csv", list(pruebas[0]),
                    ([p[k] for k in pruebas[0]] for p in pruebas))
        if any(p["estado"] != "CORRECTA" for p in pruebas):
            raise RuntimeError("Falló alguna prueba funcional. Consulte auditoria/pruebas_funcionales_26.json.")
        print("[Pruebas] 26 de 26 casos funcionales correctos.\n", flush=True)

        matriz, coordenadas = generar_instancia(NUM_CIUDADES, SEMILLA_DATOS, "simetrica_aleatoria")
        resultados = ejecutar_treinta_semillas(matriz, carpeta / "corridas")
        principal = next(r for r in resultados if r["parametros"]["semilla_algoritmo"] == SEMILLA_PRINCIPAL)
        repetibilidad = verificar_repetibilidad(matriz, principal)
        guardar_json(carpeta / "auditoria" / "repetibilidad.json", repetibilidad)
        if not repetibilidad["correcta"]:
            raise RuntimeError("La repetición de la semilla 42 no produjo identidad exacta; revise la auditoría.")

        resumen = resumir_experimento(resultados, matriz)
        diagnostico = diagnosticar_corrida(principal)
        guardar_json(carpeta / "auditoria" / "diagnostico_semilla_42.json", diagnostico)
        exportar_datos(resultados, principal, resumen, matriz, coordenadas, carpeta / "datos")
        tablas = construir_tablas(resultados, principal, resumen, matriz)
        contraste = comparar_con_articulo(tablas, principal, resumen, diagnostico)
        guardar_json(carpeta / "auditoria" / "comparacion_con_articulo.json", contraste)
        detalle = contraste["detalle_celdas"]
        guardar_csv(carpeta / "auditoria" / "comparacion_con_articulo.csv", list(detalle[0]),
                    ([fila[k] for k in detalle[0]] for fila in detalle))
        guardar_json(carpeta / "auditoria" / "referencias_transcritas_del_articulo.json", REFERENCIAS_ARTICULO)

        print("[Exportación] Generando las siete tablas y las cinco figuras...", flush=True)
        for numero, tabla in tablas.items():
            guardar_csv(carpeta / "tablas" / f"tabla_{numero}.csv", tabla["encabezados"], tabla["filas"])
            guardar_csv(carpeta / "tablas" / f"tabla_{numero}_presentacion.csv", tabla["encabezados"], tabla["filas_presentacion"])
            generar_imagen_tabla(tabla, carpeta / "tablas")
        guardar_json(carpeta / "datos" / "tablas_completas.json", tablas)
        figuras = generar_figuras(resultados, principal, coordenadas, carpeta / "figuras")
        informe = escribir_informe(carpeta, tablas, figuras, resumen, pruebas, repetibilidad,
                                   contraste, principal, diagnostico, entorno)
        (carpeta / "LEAME.txt").write_text(TEXTO_LEAME, encoding="utf-8")

        # El control final verifica presencia, no solo que se imprimió un mensaje.
        esperados = [carpeta / "tablas" / f"tabla_{n}.{ext}" for n in range(1, 8) for ext in ("csv", "png", "svg")]
        esperados += [carpeta / "figuras" / f"{figuras[n]}.{ext}" for n in range(1, 6) for ext in ("png", "svg")]
        esperados.append(informe)
        if any(not p.is_file() or p.stat().st_size == 0 for p in esperados):
            raise RuntimeError("Falta un archivo de tabla, figura o informe; no se declara completa la exportación.")
        coincide = contraste["todas_las_celdas_coinciden"] and contraste["todos_los_adicionales_coinciden"]
        estado = "COMPLETADO_Y_COINCIDENTE" if coincide else "COMPLETADO_CON_DIFERENCIAS"
        guardar_json(carpeta / "ESTADO.json", {"estado": estado, "completo": True,
                                              "segundos_proceso_completo": perf_counter()-inicio_completo,
                                              "informe": informe.name})
        crear_manifiesto(carpeta, entorno, resumen, contraste, repetibilidad)

        print("\nRESULTADOS DEL EXPERIMENTO COMPLETO", flush=True)
        for nombre, valor in tablas[5]["filas_presentacion"]:
            print(f"  {nombre}: {valor}")
        print(f"\nComparación de Tablas 4–7: {contraste['celdas_coincidentes']}/{contraste['celdas_contrastadas']} celdas coincidentes.")
        print("Diez contrastes adicionales:", "COINCIDENTES" if contraste["todos_los_adicionales_coinciden"] else "REVISAR DIFERENCIAS")
        print("Repetición de 42: idéntica; no añadida a la muestra de treinta.")
        print("Siete tablas CSV/PNG/SVG y cinco figuras PNG/SVG guardadas.")
        print("Informe:", informe, flush=True)
        if not coincide:
            print("ADVERTENCIA: consulte auditoria/comparacion_con_articulo.json. No se modificaron resultados para forzar coincidencia.")
        if abrir_informe:
            try:
                abierto = webbrowser.open(informe.as_uri())
                if not abierto:
                    print("El navegador no se abrió automáticamente. Abra el archivo HTML indicado.")
            except Exception as error:
                print("No se abrió el navegador, pero los archivos se guardaron:", error)
        if mostrar_figuras:
            # Solo después de guardar todo se presentan las imágenes en ventanas.
            import matplotlib.pyplot as plt
            for numero, nombre in figuras.items():
                figura = plt.figure(figsize=(10, 6))
                plt.imshow(plt.imread(carpeta / "figuras" / f"{nombre}.png"))
                plt.axis("off"); plt.title(f"Figura {numero}"); plt.tight_layout()
            plt.show()
        return {"carpeta_salida": str(carpeta), "informe": str(informe), "resumen": resumen,
                "tablas": tablas, "resultados_30_semillas": resultados,
                "comparacion": contraste, "repetibilidad": repetibilidad, "pruebas": pruebas}
    except BaseException as error:
        # Conservar resultados parciales es útil; presentarlos como completos no.
        guardar_json(carpeta / "ESTADO.json", {"estado": "INTERRUMPIDO_O_FALLIDO", "completo": False,
                                              "error": f"{type(error).__name__}: {error}"})
        (carpeta / "ERROR.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print("La ejecución no se completó. Evidencia parcial:", carpeta, flush=True)
        raise


def main(argumentos=None) -> dict:
    """Punto de entrada: sin argumentos ejecuta el protocolo completo de 30 semillas.

    En Spyder se utiliza F5. En terminal, --salida elige una carpeta base;
    --sin-abrir impide abrir el navegador y --mostrar-figuras abre las imágenes
    solo después de finalizar. Ninguna opción reduce semillas o generaciones.
    """
    parser = argparse.ArgumentParser(description="Reproduce las siete tablas y cinco figuras del artículo del agente viajero.")
    parser.add_argument("--salida", default=CARPETA_BASE_PERSONALIZADA, help="Carpeta base para guardar una nueva ejecución.")
    parser.add_argument("--sin-abrir", action="store_true", help="No abrir el informe HTML al terminar.")
    parser.add_argument("--mostrar-figuras", action="store_true", help="Mostrar las cinco figuras al final, después de exportarlas.")
    opciones = parser.parse_args(argumentos)
    return ejecutar_reproduccion(opciones.salida,
                                abrir_informe=ABRIR_INFORME_AL_TERMINAR and not opciones.sin_abrir,
                                mostrar_figuras=opciones.mostrar_figuras)


# ============================================================================
# F. REFERENCIAS TEXTUALES PARA DOCUMENTACIÓN Y CONTRASTE
# ============================================================================
# Esta sección NO contiene resultados para alimentar al algoritmo genético.
# Las cifras empíricas transcritas se consultan únicamente DESPUÉS del experimento
# para construir la comparación celda por celda. Las Tablas 1–3 son documentales.
# No hay recursos gráficos incrustados. La Figura 1 se dibuja con código;
# las Figuras 2–5 se construyen a partir de las salidas numéricas recién calculadas.

TEXTO_LEAME = """REPRODUCCIÓN DEL ARTÍCULO DEL AGENTE VIAJERO — JAIME AGUILAR ORTIZ

ABRIR PRIMERO: INFORME_REPRODUCCION.html
Contiene las SIETE tablas y las CINCO figuras con la misma numeración del artículo.
No requiere servidor, internet, Excel, Word ni una biblioteca de algoritmos genéticos.

QUÉ SE EJECUTÓ
Una matriz simétrica de 20 ciudades (semilla 20260917), treinta búsquedas con semillas
42–71, N=100, G=500, e=2, k=3, cruce=0.90, máscara=0.50, mutación=0.05 por clave.
Cada búsqueda usa 49,100 evaluaciones más una auditoría final. Se conserva el flujo
aleatorio y las operaciones del programa estudiado en el artículo.
Se repite la semilla 42 fuera de la muestra y se ejecutan 26 casos funcionales
incluidos, construidos a partir de las categorías descritas en el artículo.

PROCEDENCIA
Tablas 1 y 2: documentación de métodos y configuraciones.
Tabla 3: antecedente reportado por la guía. NO se vuelven a ejecutar las 612 ni las
309 comprobaciones históricas. Se exportan con su procedencia, sin sumarlas a las
26 pruebas de esta ejecución. La función esfera se conserva como antecedente
reportado, no se ejecuta como parte del experimento del agente viajero.
Figura 1: esquema redibujado con texto, recuadros y flechas de Matplotlib.
No se requiere una imagen externa ni hay una imagen codificada dentro del script.
El SVG es vectorial. Se conserva la arquitectura, no una copia visual exacta.
Tablas 4–7 y Figuras 2–5: calculadas en esta ejecución, no copiadas del artículo.
Las Figuras 2–5 se exportan como PNG de 300 ppp y SVG vectoriales.

ARCHIVOS
- tablas/tabla_N.csv: tabla con precisión numérica completa.
- tablas/tabla_N_presentacion.csv: redondeo empleado en el artículo.
- tablas/tabla_N.png y .svg: imagen de cada tabla, incluidas las 30 filas de Tabla 7.
- figuras/: las cinco imágenes numeradas en PNG y SVG.
- datos/historiales_30_semillas.csv: 15,030 registros de evolución.
- datos/historiales_30_semillas.npz: semillas, mejores, promedios, rutas, claves,
  matriz y coordenadas. Usar np.load(..., allow_pickle=False).
- datos/datos_figura_3.csv: cuartiles y mediana para cada generación.
- datos/resumen_por_semilla.csv: costos, reducciones, presupuesto, tiempo y rutas.
- datos/matriz_distancias.csv: sin encabezado; cargar con np.loadtxt(..., delimiter=',').
- datos/coordenadas_esquema.csv: esquema circular; NO coordenadas geográficas.
- datos/tramos_todas_las_rutas.csv: las veinte conexiones de cada una de las 30 rutas.
- corridas/: resultado independiente de cada semilla en NPZ.
- auditoria/: comparación con cifras publicadas, pruebas y repetibilidad.
- ESTADO.json: indica si el proceso terminó y si las comparaciones coincidieron.
- manifiesto.json: versiones, parámetros, semillas, procedencia y huellas de archivos.

PRECISIÓN Y LÍMITES
Los cuartiles usan interpolación lineal; la desviación estándar usa ddof=1.
La mediana de las reducciones se calcula sobre los 30 porcentajes individuales.
Los descensos <=1e-10 se diagnostican después, sin cambiar los operadores ni el paro.
La comparación usa el mismo redondeo del artículo y no altera valores discrepantes.
Los tiempos y la apariencia tipográfica pueden cambiar según el equipo.
Entorno numérico del artículo: Python 3.13.5, NumPy 2.3.5 y Matplotlib 3.10.8, Linux.
Las políticas de compatibilidad no garantizan identidad entre todas las versiones,
plataformas y tamaños de llamadas aleatorias; consulte la auditoría de cada corrida.
No se demuestra optimalidad, superioridad comparativa ni eficacia educativa.

REPETIR
Abrir REPRODUCIR_ARTICULO_AGENTE_VIAJERO_30_SEMILLAS_V2.py en Spyder y pulsar F5.
En terminal: python REPRODUCIR_ARTICULO_AGENTE_VIAJERO_30_SEMILLAS_V2.py --sin-abrir
Cada ejecución crea otra carpeta. Conserve la carpeta completa al copiar el informe.
"""

# Cifras transcritas del artículo para contraste, NO resultados simulados.
REFERENCIAS_ARTICULO = {'1': [['Etapa', 'Método', 'Aplicación y motivo de elección'], ['1. Inicialización', 'M01. Uniforme en caja', 'Prioridades reales independientes en [0, 1); no se requiere suma fija.'], ['2. Evaluación', 'M05. Objetivo directo', 'Costo único del recorrido cerrado; no necesita ponderación ni penalización.'], ['3. Selección', 'M12. Torneo', 'Compara costos de minimización sin transformarlos en probabilidades.'], ['4. Recombinación', 'M14. Cruce uniforme', 'Hereda claves por coordenada; dos hijos con procedencias complementarias.'], ['5. Mutación', 'M18. Reinicio uniforme', 'Introduce prioridades nuevas sin ajustar una escala local de perturbación.'], ['6. Reemplazo', 'M22. Elitismo', 'Protege dos progenitores y acepta los 98 hijos restantes.'], ['7. Paro', 'M25. Generaciones', 'Completa 500 transiciones; no incorpora otras condiciones de paro.']], '2': [['Componente', 'Guía: función esfera', 'Agente viajero'], ['Variables y población', '4 variables; N = 32', '20 claves; N = 100'], ['Objetivo', 'Suma de cuadrados', 'Costo del ciclo cerrado'], ['Inicialización', 'M03. Hipercubo latino', 'M01. Uniforme independiente'], ['Evaluación y selección', 'M05; M12, torneo k = 3', 'M05; M12, torneo k = 3'], ['Recombinación', 'M15. Aritmética; p_c = 0.90', 'M14. Uniforme; p_c = 0.90; q = 0.50'], ['Mutación', 'M19. Gaussiana; p_m = 0.25; σ = 0.15', 'M18. Reinicio; p_m = 0.05'], ['Reemplazo', 'M23. Unión de padres e hijos', 'M22. Dos élites y 98 hijos'], ['Límites de trabajo', '200 transiciones; 4,096 evaluaciones; 30 s', '500 transiciones; 49,100 evaluaciones de búsqueda'], ['Terminación', 'Seis controles combinados; meta 10⁻⁶', 'Solo M25; auditoría final adicional'], ['Semillas relevantes', 'Ejemplo integrado: 20260917', 'Datos: 20260917; algoritmo: 42–71']], '3': [['Grupo de comprobaciones', 'Registro histórico', 'Edición ampliada'], ['Ejemplos de los treinta métodos', '30', '30'], ['Inicialización', '90', '40'], ['Evaluación', '8', '6'], ['Selección', '30', '50'], ['Recombinación', '60', '50'], ['Mutación', '180', '50'], ['Reemplazo', '121', '21'], ['Criterios de paro', '29', '17'], ['Entradas inválidas', '38', '24'], ['Integración y repetibilidad / regresión', '26', '21'], ['Total informado', '612', '309']], '4': [['Generación', 'Mejor costo', 'Promedio poblacional', 'Evaluaciones de búsqueda'], ['0', '798.391379', '1013.287426', '100'], ['25', '552.159158', '697.717577', '2,550'], ['50', '537.774027', '636.265950', '5,000'], ['100', '514.224848', '621.548624', '9,900'], ['113', '499.843695', '655.523536', '11,174'], ['120', '499.843695', '617.304743', '11,860'], ['250', '499.843695', '595.910623', '24,600'], ['500', '499.843695', '598.019995', '49,100']], '5': [['Estadístico', 'Valor'], ['Mínimo del costo final', '449.828174'], ['Primer cuartil', '466.320771'], ['Mediana del costo final', '474.777123'], ['Tercer cuartil', '491.643464'], ['Máximo del costo final', '533.380587'], ['Media del costo final', '479.301013'], ['Desviación estándar muestral', '20.870978'], ['Mediana de reducción relativa', '41.387690%'], ['Corridas con reducción del mejor costo inicial', '30 de 30']], '6': [['Tramo', 'Conexión', 'Costo', 'Tramo', 'Conexión', 'Costo'], ['1', '0 → 17', '5.231906', '11', '7 → 16', '36.909015'], ['2', '17 → 1', '18.142151', '12', '16 → 2', '6.215857'], ['3', '1 → 6', '26.259631', '13', '2 → 11', '43.520926'], ['4', '6 → 14', '38.405249', '14', '11 → 12', '35.062505'], ['5', '14 → 9', '19.765422', '15', '12 → 10', '35.063092'], ['6', '9 → 8', '36.578216', '16', '10 → 3', '28.368831'], ['7', '8 → 5', '13.576871', '17', '3 → 15', '36.480597'], ['8', '5 → 4', '25.396680', '18', '15 → 19', '12.262501'], ['9', '4 → 18', '11.966798', '19', '19 → 13', '27.434680'], ['10', '18 → 7', '37.895779', '20', '13 → 0', '5.306989'], ['Costo total del ciclo', 'Costo total del ciclo', 'Costo total del ciclo', 'Costo total del ciclo', 'Costo total del ciclo', '499.843695']], '7': [['Semilla', 'Mejor costo inicial', 'Mejor costo final', 'Reducción (%)'], ['42', '798.391379', '499.843695', '37.394'], ['43', '842.517921', '452.857147', '46.250'], ['44', '830.815878', '471.096723', '43.297'], ['45', '746.781766', '456.820813', '38.828'], ['46', '780.445415', '472.354958', '39.476'], ['47', '815.986729', '452.876963', '44.499'], ['48', '799.018210', '493.377141', '38.252'], ['49', '810.260728', '474.633630', '41.422'], ['50', '794.457428', '480.180963', '39.559'], ['51', '798.357081', '517.787947', '35.143'], ['52', '734.286995', '477.585756', '34.959'], ['53', '802.042214', '505.943244', '36.918'], ['54', '806.829820', '465.885896', '42.257'], ['55', '799.973654', '469.158393', '41.353'], ['56', '850.155731', '492.841274', '42.029'], ['57', '847.431333', '471.090075', '44.410'], ['58', '813.898930', '453.662466', '44.261'], ['59', '830.162762', '467.333482', '43.706'], ['60', '846.956048', '485.852964', '42.635'], ['61', '833.516512', '474.920617', '43.022'], ['62', '837.569947', '459.823853', '45.100'], ['63', '803.831556', '488.050036', '39.285'], ['64', '816.102085', '533.380587', '34.643'], ['65', '809.456172', '493.944192', '38.978'], ['66', '822.946769', '518.505648', '36.994'], ['67', '741.579369', '485.948142', '34.471'], ['68', '866.555994', '466.267430', '46.193'], ['69', '862.182265', '466.480793', '45.895'], ['70', '803.411669', '480.697389', '40.168'], ['71', '844.193285', '449.828174', '46.715']]}

SHA256_CODIGO_BASE = 'e3ab1da8d674eb7bad3c79abc4187ce2a4e6e0cf0c2a2031cfcb8521058ece24'
SHA256_ARTICULO = 'e758e9b409c1fde3b7881176cafb72913c61d4271cdd01025ec3cc5c50b8e93e'

# %% Ejecutar el experimento completo (sin argumentos: treinta semillas).
if __name__ == "__main__":
    try:
        RESULTADO = main()
    except KeyboardInterrupt:
        print("\nEjecución interrumpida por el usuario. Los resultados parciales no reproducen el artículo completo.")
    except Exception as error:
        print(f"\nNo se completó el protocolo: {type(error).__name__}: {error}")
        raise
