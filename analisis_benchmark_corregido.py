"""Analisis reproducible del benchmark DuckDB, Polars y Apache Spark.

El programa lee resultados/benchmark_detalle.csv, conserva solo las 270
mediciones definitivas y genera tablas, pruebas y figuras en
resultados/analisis. El archivo fuente nunca se modifica.

Uso:
    python analisis_benchmark_corregido.py
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


RAIZ = Path(__file__).resolve().parent
FUENTE = RAIZ / "resultados" / "benchmark_detalle.csv"
SALIDA = RAIZ / "resultados" / "analisis_corregido"
FIGURAS = SALIDA / "figuras"

ALFA = 0.05
BOOTSTRAP_EFECTOS = 9_999
BOOTSTRAP_CONTRASTES = 9_999
SEMILLA_BOOTSTRAP = 20260925
SEMILLA_BOOTSTRAP_CONTRASTES = 20260926

VERSIONES_ESPERADAS = {
    "pandas": "2.3.3",
    "numpy": "2.4.6",
    "scipy": "1.17.1",
    "statsmodels": "0.15.0",
    "matplotlib": "3.11.1",
    "seaborn": "0.13.2",
}

TECNOLOGIAS = ["duckdb", "polars", "spark"]
OPERACIONES = ["filtrado", "agregacion", "join"]
VOLUMENES = ["sf1", "sf5", "sf10"]

ETIQUETAS_TECNOLOGIA = {
    "duckdb": "DuckDB",
    "polars": "Polars",
    "spark": "Apache Spark",
}
ETIQUETAS_OPERACION = {
    "filtrado": "Filtrado",
    "agregacion": "Agregación",
    "join": "JOIN",
}
ETIQUETAS_VOLUMEN = {"sf1": "SF1", "sf5": "SF5", "sf10": "SF10"}

RESPUESTAS_PRINCIPALES = {
    "tiempo_s": "Tiempo de ejecución (s)",
    "memoria_pico_mb": "Pico absoluto de memoria USS (MB)",
}
RESPUESTA_COMPLEMENTARIA = {
    "incremento_memoria_mb": "Incremento de memoria USS (MB)"
}
TODAS_RESPUESTAS = {**RESPUESTAS_PRINCIPALES, **RESPUESTA_COMPLEMENTARIA}

TERMINOS = [
    "tecnologia",
    "operacion",
    "volumen",
    "tecnologia:operacion",
    "tecnologia:volumen",
    "operacion:volumen",
    "tecnologia:operacion:volumen",
]

ETIQUETAS_TERMINOS = {
    "tecnologia": "Tecnología",
    "operacion": "Operación",
    "volumen": "Volumen",
    "tecnologia:operacion": "Tecnología × operación",
    "tecnologia:volumen": "Tecnología × volumen",
    "operacion:volumen": "Operación × volumen",
    "tecnologia:operacion:volumen": "Tecnología × operación × volumen",
}


def ahora_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def es_verdadero(serie: pd.Series) -> pd.Series:
    return serie.astype(str).str.strip().str.lower().isin(
        ["true", "1", "si", "sí"]
    )


def version_paquete(nombre: str) -> str:
    try:
        return importlib.metadata.version(nombre)
    except importlib.metadata.PackageNotFoundError:
        return "NO INSTALADO"


def comprobar_versiones() -> dict[str, str]:
    detectadas = {
        nombre: version_paquete(nombre) for nombre in VERSIONES_ESPERADAS
    }
    diferencias = [
        f"{nombre}: esperada {esperada}, detectada {detectadas[nombre]}"
        for nombre, esperada in VERSIONES_ESPERADAS.items()
        if detectadas[nombre] != esperada
    ]
    if diferencias and os.environ.get("PERMITIR_VERSIONES_DISTINTAS") != "1":
        raise RuntimeError(
            "Las versiones no coinciden con las declaradas en Metodos:\n  - "
            + "\n  - ".join(diferencias)
        )
    if diferencias:
        warnings.warn(
            "La comprobacion se ejecutara con versiones distintas de las "
            "declaradas porque PERMITIR_VERSIONES_DISTINTAS=1: "
            + " | ".join(diferencias),
            RuntimeWarning,
        )
    return detectadas


def sha256(ruta: Path) -> str:
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not FUENTE.is_file():
        raise FileNotFoundError(f"No se encontro el archivo fuente: {FUENTE}")

    datos = pd.read_csv(FUENTE, encoding="utf-8-sig")
    requeridas = {
        "sesion_id",
        "tarea_id",
        "tipo",
        "bloque",
        "orden_en_bloque",
        "incluir_analisis",
        "tecnologia",
        "volumen",
        "operacion",
        "resultado_correcto",
        "tiempo_s",
        "memoria_base_mb",
        "memoria_pico_mb",
        "incremento_memoria_mb",
        "muestras_operacion",
        "intervalo_nominal_ms",
        "intervalo_real_mediana_ms",
        "errores_muestreo",
        "tiene_python",
        "tiene_java",
        "estado",
    }
    faltantes = sorted(requeridas - set(datos.columns))
    if faltantes:
        raise RuntimeError("Faltan columnas: " + ", ".join(faltantes))

    for columna in [
        "bloque",
        "orden_en_bloque",
        "tiempo_s",
        "memoria_base_mb",
        "memoria_pico_mb",
        "incremento_memoria_mb",
        "muestras_operacion",
        "intervalo_nominal_ms",
        "intervalo_real_mediana_ms",
        "errores_muestreo",
    ]:
        datos[columna] = pd.to_numeric(datos[columna], errors="raise")

    datos["incluir_analisis_bool"] = es_verdadero(datos["incluir_analisis"])
    datos["resultado_correcto_bool"] = es_verdadero(datos["resultado_correcto"])
    datos["tiene_python_bool"] = es_verdadero(datos["tiene_python"])
    datos["tiene_java_bool"] = es_verdadero(datos["tiene_java"])
    mediciones = datos.loc[datos["incluir_analisis_bool"]].copy()

    mediciones["tecnologia"] = pd.Categorical(
        mediciones["tecnologia"], categories=TECNOLOGIAS, ordered=True
    )
    mediciones["operacion"] = pd.Categorical(
        mediciones["operacion"], categories=OPERACIONES, ordered=True
    )
    mediciones["volumen"] = pd.Categorical(
        mediciones["volumen"], categories=VOLUMENES, ordered=True
    )
    mediciones["bloque"] = mediciones["bloque"].astype(int)
    mediciones["orden_en_bloque"] = mediciones["orden_en_bloque"].astype(int)
    mediciones["posicion_centrada"] = (
        mediciones["orden_en_bloque"]
        - mediciones["orden_en_bloque"].mean()
    ) / mediciones["orden_en_bloque"].std(ddof=0)
    mediciones = mediciones.sort_values(
        ["bloque", "orden_en_bloque"]
    ).reset_index(drop=True)
    return datos, mediciones


def control_integridad(
    datos: pd.DataFrame, mediciones: pd.DataFrame
) -> pd.DataFrame:
    conteos = mediciones.groupby(
        ["tecnologia", "operacion", "volumen"], observed=True
    ).size()
    calentamientos = datos.loc[~datos["incluir_analisis_bool"]]
    conteos_calentamiento = calentamientos.groupby(
        ["tecnologia", "operacion", "volumen"], observed=True
    ).size()
    spark = datos.loc[datos["tecnologia"].astype(str) == "spark"]

    controles = [
        ("Filas totales", len(datos), 324),
        ("Tareas únicas", datos["tarea_id"].nunique(), 324),
        ("Sesiones experimentales", datos["sesion_id"].nunique(), 1),
        ("Calentamientos", len(calentamientos), 54),
        ("Mediciones analizables", len(mediciones), 270),
        ("Escenarios medidos", len(conteos), 27),
        ("Mínimo de repeticiones por escenario", int(conteos.min()), 10),
        ("Máximo de repeticiones por escenario", int(conteos.max()), 10),
        (
            "Mínimo de calentamientos por escenario",
            int(conteos_calentamiento.min()),
            2,
        ),
        (
            "Máximo de calentamientos por escenario",
            int(conteos_calentamiento.max()),
            2,
        ),
        (
            "Resultados incorrectos",
            int((~datos["resultado_correcto_bool"]).sum()),
            0,
        ),
        (
            "Estados distintos de correcto",
            int((datos["estado"] != "correcto").sum()),
            0,
        ),
        ("Errores de muestreo", int(datos["errores_muestreo"].sum()), 0),
        (
            "Filas Spark sin Python",
            int((~spark["tiene_python_bool"]).sum()),
            0,
        ),
        (
            "Filas Spark sin Java",
            int((~spark["tiene_java_bool"]).sum()),
            0,
        ),
        (
            "Intervalos nominales distintos de 2 ms",
            int((datos["intervalo_nominal_ms"] != 2).sum()),
            0,
        ),
        (
            "Valores faltantes en variables principales",
            int(
                mediciones[
                    [
                        "tiempo_s",
                        "memoria_pico_mb",
                        "incremento_memoria_mb",
                    ]
                ].isna().sum().sum()
            ),
            0,
        ),
        (
            "Tiempos no positivos",
            int((mediciones["tiempo_s"] <= 0).sum()),
            0,
        ),
        (
            "Picos de memoria menores que la línea de base",
            int(
                (
                    mediciones["memoria_pico_mb"]
                    < mediciones["memoria_base_mb"]
                ).sum()
            ),
            0,
        ),
    ]
    tabla = pd.DataFrame(
        controles, columns=["control", "valor_observado", "valor_esperado"]
    )
    tabla["estado"] = np.where(
        tabla["valor_observado"] == tabla["valor_esperado"],
        "correcto",
        "error",
    )
    if (tabla["estado"] != "correcto").any():
        errores = tabla.loc[tabla["estado"] != "correcto", "control"].tolist()
        raise RuntimeError(
            "El control de integridad no fue superado: " + ", ".join(errores)
        )
    return tabla


def resumen_descriptivo(mediciones: pd.DataFrame) -> pd.DataFrame:
    filas: list[dict[str, Any]] = []
    grupos = mediciones.groupby(
        ["tecnologia", "operacion", "volumen"],
        observed=True,
        sort=False,
    )
    for (tecnologia, operacion, volumen), grupo in grupos:
        for variable, etiqueta in TODAS_RESPUESTAS.items():
            valores = grupo[variable].to_numpy(dtype=float)
            q1, q3 = np.quantile(valores, [0.25, 0.75])
            filas.append(
                {
                    "tecnologia": str(tecnologia),
                    "operacion": str(operacion),
                    "volumen": str(volumen),
                    "variable": variable,
                    "variable_etiqueta": etiqueta,
                    "n": len(valores),
                    "media": np.mean(valores),
                    "desviacion_estandar": np.std(valores, ddof=1),
                    "mediana": np.median(valores),
                    "q1": q1,
                    "q3": q3,
                    "rango_intercuartil": q3 - q1,
                    "minimo": np.min(valores),
                    "maximo": np.max(valores),
                }
            )
    return pd.DataFrame(filas)


def detectar_atipicos(mediciones: pd.DataFrame) -> pd.DataFrame:
    filas: list[dict[str, Any]] = []
    grupos = mediciones.groupby(
        ["tecnologia", "operacion", "volumen"], observed=True
    )
    for (tecnologia, operacion, volumen), grupo in grupos:
        for variable in TODAS_RESPUESTAS:
            q1, q3 = grupo[variable].quantile([0.25, 0.75])
            ric = q3 - q1
            inferior = q1 - 1.5 * ric
            superior = q3 + 1.5 * ric
            seleccion = grupo.loc[
                (grupo[variable] < inferior) | (grupo[variable] > superior)
            ]
            for _, fila in seleccion.iterrows():
                filas.append(
                    {
                        "tarea_id": fila["tarea_id"],
                        "bloque": int(fila["bloque"]),
                        "orden_en_bloque": int(fila["orden_en_bloque"]),
                        "tecnologia": str(tecnologia),
                        "operacion": str(operacion),
                        "volumen": str(volumen),
                        "variable": variable,
                        "valor": fila[variable],
                        "limite_inferior_tukey": inferior,
                        "limite_superior_tukey": superior,
                        "accion": "conservar y revisar; no eliminado",
                    }
                )
    columnas = [
        "tarea_id",
        "bloque",
        "orden_en_bloque",
        "tecnologia",
        "operacion",
        "volumen",
        "variable",
        "valor",
        "limite_inferior_tukey",
        "limite_superior_tukey",
        "accion",
    ]
    return pd.DataFrame(filas, columns=columnas)


def codificacion_suma(
    serie: pd.Series, niveles: list[str], prefijo: str
) -> tuple[np.ndarray, list[str]]:
    valores = serie.astype(str).to_numpy()
    matriz = np.zeros((len(serie), len(niveles) - 1), dtype=float)
    for indice, nivel in enumerate(niveles[:-1]):
        matriz[:, indice] = np.where(
            valores == nivel, 1.0, np.where(valores == niveles[-1], -1.0, 0.0)
        )
    nombres = [f"{prefijo}[{nivel}]" for nivel in niveles[:-1]]
    return matriz, nombres


@dataclass
class Diseno:
    x_completo: np.ndarray
    nombres_columnas: list[str]
    indices_terminos: dict[str, list[int]]
    bloques: np.ndarray


def construir_diseno(mediciones: pd.DataFrame) -> Diseno:
    n = len(mediciones)
    partes: list[np.ndarray] = [np.ones((n, 1))]
    nombres = ["intercepto"]

    bloques = pd.get_dummies(
        mediciones["bloque"].astype(str), drop_first=True, dtype=float
    )
    partes.append(bloques.to_numpy())
    nombres.extend([f"bloque[{col}]" for col in bloques.columns])
    partes.append(mediciones[["posicion_centrada"]].to_numpy(dtype=float))
    nombres.append("posicion_centrada")

    t, nt = codificacion_suma(mediciones["tecnologia"], TECNOLOGIAS, "tec")
    o, no = codificacion_suma(mediciones["operacion"], OPERACIONES, "op")
    v, nv = codificacion_suma(mediciones["volumen"], VOLUMENES, "vol")

    indices: dict[str, list[int]] = {}

    def agregar(termino: str, matriz: np.ndarray, nombres_nuevos: list[str]):
        inicio = sum(parte.shape[1] for parte in partes)
        partes.append(matriz)
        nombres.extend(nombres_nuevos)
        indices[termino] = list(range(inicio, inicio + matriz.shape[1]))

    agregar("tecnologia", t, nt)
    agregar("operacion", o, no)
    agregar("volumen", v, nv)

    def interaccion(
        a: np.ndarray,
        na: list[str],
        b: np.ndarray,
        nb: list[str],
    ) -> tuple[np.ndarray, list[str]]:
        columnas = [a[:, i] * b[:, j] for i in range(a.shape[1]) for j in range(b.shape[1])]
        nombres_i = [f"{na[i]}:{nb[j]}" for i in range(a.shape[1]) for j in range(b.shape[1])]
        return np.column_stack(columnas), nombres_i

    to, nto = interaccion(t, nt, o, no)
    tv, ntv = interaccion(t, nt, v, nv)
    ov, nov = interaccion(o, no, v, nv)
    agregar("tecnologia:operacion", to, nto)
    agregar("tecnologia:volumen", tv, ntv)
    agregar("operacion:volumen", ov, nov)

    columnas_tov = []
    nombres_tov = []
    for i, j, k in itertools.product(
        range(t.shape[1]), range(o.shape[1]), range(v.shape[1])
    ):
        columnas_tov.append(t[:, i] * o[:, j] * v[:, k])
        nombres_tov.append(f"{nt[i]}:{no[j]}:{nv[k]}")
    agregar(
        "tecnologia:operacion:volumen",
        np.column_stack(columnas_tov),
        nombres_tov,
    )

    x = np.column_stack(partes)
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise RuntimeError("La matriz factorial completa no tiene rango completo.")
    return Diseno(x, nombres, indices, mediciones["bloque"].to_numpy())


def base_ortonormal(x: np.ndarray) -> tuple[np.ndarray, int]:
    u, singulares, _ = np.linalg.svd(x, full_matrices=False)
    tolerancia = np.max(x.shape) * np.finfo(float).eps * singulares[0]
    rango = int(np.sum(singulares > tolerancia))
    return u[:, :rango], rango


def sse_desde_q(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return np.maximum(np.dot(y, y) - np.sum((q.T @ y) ** 2), 0.0)
    return np.maximum(np.sum(y * y, axis=0) - np.sum((q.T @ y) ** 2, axis=0), 0.0)


def matrices_reducidas(diseno: Diseno) -> dict[str, np.ndarray]:
    reducidas = {}
    todas = np.arange(diseno.x_completo.shape[1])
    for termino, quitar in diseno.indices_terminos.items():
        conservar = np.setdiff1d(todas, quitar)
        reducidas[termino] = diseno.x_completo[:, conservar]
    return reducidas


def eta_parcial(sse_reducido: float, sse_completo: float) -> float:
    numerador = max(sse_reducido - sse_completo, 0.0)
    denominador = numerador + sse_completo
    return numerador / denominador if denominador > 0 else 0.0


def ajustar_ols_hc3(y: np.ndarray, x: np.ndarray) -> dict[str, Any]:
    """Ajusta MCO y calcula la matriz de covarianza robusta HC3."""
    xtx_inversa = np.linalg.pinv(x.T @ x, rcond=1e-12, hermitian=True)
    beta = xtx_inversa @ (x.T @ y)
    ajustados = x @ beta
    residuos = y - ajustados
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inversa, x)
    divisor = np.maximum(1.0 - leverage, np.finfo(float).eps)
    x_ponderada = x * (residuos / divisor)[:, None]
    carne = x_ponderada.T @ x_ponderada
    cov_hc3 = xtx_inversa @ carne @ xtx_inversa
    cov_hc3 = (cov_hc3 + cov_hc3.T) / 2.0
    errores = np.sqrt(np.maximum(np.diag(cov_hc3), 0.0))
    gl_error = len(y) - np.linalg.matrix_rank(x)
    estadisticos = np.divide(
        beta,
        errores,
        out=np.full_like(beta, np.nan),
        where=errores > 0,
    )
    p_coeficientes = 2.0 * stats.t.sf(np.abs(estadisticos), gl_error)
    critico = stats.t.ppf(0.975, gl_error)
    return {
        "beta": beta,
        "ajustados": ajustados,
        "residuos": residuos,
        "cov_hc3": cov_hc3,
        "errores_hc3": errores,
        "estadisticos_t": estadisticos,
        "p_coeficientes": p_coeficientes,
        "ci_inferior": beta - critico * errores,
        "ci_superior": beta + critico * errores,
        "sse": float(residuos @ residuos),
        "gl_error": int(gl_error),
        "leverage": leverage,
    }


def prueba_wald_hc3(
    ajuste: dict[str, Any], indices: list[int]
) -> tuple[float, float, int]:
    """Prueba conjunta bilateral H0: todos los coeficientes del término = 0."""
    beta = ajuste["beta"][indices]
    cov = ajuste["cov_hc3"][np.ix_(indices, indices)]
    rango = int(np.linalg.matrix_rank(cov))
    if rango == 0:
        return math.nan, math.nan, 0
    wald = float(beta @ np.linalg.pinv(cov, rcond=1e-12) @ beta)
    f_robusto = wald / rango
    p = float(stats.f.sf(f_robusto, rango, ajuste["gl_error"]))
    return f_robusto, p, rango


def _sse_bootstrap_para_diseno(
    x: np.ndarray,
    respuestas: dict[str, np.ndarray],
    grupos_indices: list[np.ndarray],
    conteos: np.ndarray,
    lote: int = 400,
) -> dict[str, np.ndarray]:
    """Calcula SSE de remuestreos de bloques mediante productos cruzados."""
    xtx_bloques = np.stack(
        [x[indices].T @ x[indices] for indices in grupos_indices]
    )
    xty_bloques = {
        variable: np.stack(
            [x[indices].T @ y[indices] for indices in grupos_indices]
        )
        for variable, y in respuestas.items()
    }
    yty_bloques = {
        variable: np.array([y[indices] @ y[indices] for indices in grupos_indices])
        for variable, y in respuestas.items()
    }
    resultado = {
        variable: np.empty(len(conteos), dtype=float) for variable in respuestas
    }
    for inicio in range(0, len(conteos), lote):
        fin = min(inicio + lote, len(conteos))
        pesos = conteos[inicio:fin]
        xtx = np.einsum("sb,bij->sij", pesos, xtx_bloques, optimize=True)
        inversas = np.linalg.pinv(xtx, rcond=1e-12, hermitian=True)
        for variable in respuestas:
            xty = pesos @ xty_bloques[variable]
            beta = np.einsum("sij,sj->si", inversas, xty, optimize=True)
            yty = pesos @ yty_bloques[variable]
            sse = yty - np.einsum("si,si->s", xty, beta, optimize=True)
            resultado[variable][inicio:fin] = np.maximum(sse, 0.0)
    return resultado


def bootstrap_eta_todos(
    mediciones: pd.DataFrame, diseno: Diseno
) -> dict[tuple[str, str], tuple[float, float]]:
    """IC percentiles de eta parcial mediante remuestreo de bloques completos."""
    bloques_unicos = np.unique(diseno.bloques)
    grupos_indices = [
        np.flatnonzero(diseno.bloques == bloque) for bloque in bloques_unicos
    ]
    rng = np.random.default_rng(SEMILLA_BOOTSTRAP)
    selecciones = rng.integers(
        0,
        len(grupos_indices),
        size=(BOOTSTRAP_EFECTOS, len(grupos_indices)),
    )
    conteos = np.stack(
        [np.bincount(fila, minlength=len(grupos_indices)) for fila in selecciones]
    ).astype(float)
    respuestas = {
        variable: mediciones[variable].to_numpy(dtype=float)
        for variable in RESPUESTAS_PRINCIPALES
    }
    sse_completo = _sse_bootstrap_para_diseno(
        diseno.x_completo, respuestas, grupos_indices, conteos
    )
    intervalos: dict[tuple[str, str], tuple[float, float]] = {}
    for termino, x_reducido in matrices_reducidas(diseno).items():
        sse_reducido = _sse_bootstrap_para_diseno(
            x_reducido, respuestas, grupos_indices, conteos
        )
        for variable in RESPUESTAS_PRINCIPALES:
            numerador = np.maximum(
                sse_reducido[variable] - sse_completo[variable], 0.0
            )
            denominador = numerador + sse_completo[variable]
            valores = np.divide(
                numerador,
                denominador,
                out=np.zeros_like(numerador),
                where=denominador > 0,
            )
            intervalos[(variable, termino)] = tuple(
                np.quantile(valores, [0.025, 0.975])
            )
    return intervalos


def efectos_factoriales_hc3(
    mediciones: pd.DataFrame, diseno: Diseno
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Efectos factoriales mediante pruebas conjuntas de Wald con HC3."""
    reducidas = matrices_reducidas(diseno)
    intervalos_eta = bootstrap_eta_todos(mediciones, diseno)
    filas_efectos: list[dict[str, Any]] = []
    filas_coeficientes: list[dict[str, Any]] = []

    for variable, etiqueta in RESPUESTAS_PRINCIPALES.items():
        y = mediciones[variable].to_numpy(dtype=float)
        ajuste = ajustar_ols_hc3(y, diseno.x_completo)
        for indice, nombre in enumerate(diseno.nombres_columnas):
            filas_coeficientes.append(
                {
                    "variable": variable,
                    "variable_etiqueta": etiqueta,
                    "parametro": nombre,
                    "coeficiente": ajuste["beta"][indice],
                    "error_estandar_hc3": ajuste["errores_hc3"][indice],
                    "estadistico_t": ajuste["estadisticos_t"][indice],
                    "gl_error": ajuste["gl_error"],
                    "p_sin_ajustar": ajuste["p_coeficientes"][indice],
                    "ci95_hc3_inferior": ajuste["ci_inferior"][indice],
                    "ci95_hc3_superior": ajuste["ci_superior"][indice],
                }
            )

        sse_completo = ajuste["sse"]
        for termino in TERMINOS:
            indices = diseno.indices_terminos[termino]
            f_robusto, p, gl_efecto = prueba_wald_hc3(ajuste, indices)
            x_reducido = reducidas[termino]
            beta_reducido = np.linalg.lstsq(x_reducido, y, rcond=None)[0]
            residuo_reducido = y - x_reducido @ beta_reducido
            sse_reducido = float(residuo_reducido @ residuo_reducido)
            eta = eta_parcial(sse_reducido, sse_completo)
            ci_inf, ci_sup = intervalos_eta[(variable, termino)]
            filas_efectos.append(
                {
                    "variable": variable,
                    "variable_etiqueta": etiqueta,
                    "termino": termino,
                    "termino_etiqueta": ETIQUETAS_TERMINOS[termino],
                    "metodo": "MCO factorial; prueba conjunta de Wald con covarianza HC3",
                    "estadistico_f_robusto": f_robusto,
                    "gl_efecto": gl_efecto,
                    "gl_error": ajuste["gl_error"],
                    "p_sin_ajustar": p,
                    "eta_cuadrado_parcial": eta,
                    "eta_ci95_inferior": ci_inf,
                    "eta_ci95_superior": ci_sup,
                    "metodo_ic_eta": "bootstrap no parametrico de bloques completos; percentiles 2,5 y 97,5",
                    "remuestreos_eta": BOOTSTRAP_EFECTOS,
                    "semilla_eta": SEMILLA_BOOTSTRAP,
                }
            )
    return pd.DataFrame(filas_efectos), pd.DataFrame(filas_coeficientes)


def componentes_diseno_mixto(
    diseno: Diseno,
) -> tuple[np.ndarray, list[str], dict[str, list[int]]]:
    """Quita los bloques fijos para intentar el modelo con bloque aleatorio.

    El modelo mixto conserva el intercepto, la posición y la codificación de
    suma del factorial; el bloque se incorpora como intercepto aleatorio.
    """
    conservar = [
        indice
        for indice, nombre in enumerate(diseno.nombres_columnas)
        if not nombre.startswith("bloque[")
    ]
    equivalencias = {
        indice_original: indice_nuevo
        for indice_nuevo, indice_original in enumerate(conservar)
    }
    indices = {
        termino: [equivalencias[indice] for indice in indices_originales]
        for termino, indices_originales in diseno.indices_terminos.items()
    }
    return (
        diseno.x_completo[:, conservar],
        [diseno.nombres_columnas[indice] for indice in conservar],
        indices,
    )


def ajustar_mixto(
    y: np.ndarray,
    x: np.ndarray,
    nombres: list[str],
    bloques: np.ndarray,
):
    from statsmodels.regression.mixed_linear_model import MixedLM

    exog = pd.DataFrame(x, columns=nombres)
    modelo = MixedLM(y, exog, groups=bloques)
    with warnings.catch_warnings(record=True) as capturadas:
        warnings.simplefilter("always")
        resultado = modelo.fit(
            reml=False,
            method=["lbfgs", "powell"],
            maxiter=2_000,
            disp=False,
        )
    textos = sorted({str(item.message) for item in capturadas})
    return resultado, " | ".join(textos)


def modelos_y_supuestos(
    mediciones: pd.DataFrame, diseno: Diseno
) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnosticos = []
    supuestos = []
    q_completo, _ = base_ortonormal(diseno.x_completo)
    x_mixto, nombres_mixto, _ = componentes_diseno_mixto(diseno)

    for variable, etiqueta in RESPUESTAS_PRINCIPALES.items():
        convergio = False
        advertencias = ""
        error_ajuste = ""
        varianza_bloque = math.nan
        y = mediciones[variable].to_numpy(dtype=float)
        try:
            resultado, advertencias = ajustar_mixto(
                y, x_mixto, nombres_mixto, diseno.bloques
            )
            convergio = bool(resultado.converged)
            cov_re = np.asarray(resultado.cov_re, dtype=float)
            if cov_re.size:
                varianza_bloque = float(cov_re.ravel()[0])
        except Exception as exc:
            error_ajuste = f"{type(exc).__name__}: {exc}"

        diagnosticos.append(
            {
                "variable": variable,
                "variable_etiqueta": etiqueta,
                "formula": (
                    "respuesta ~ posicion_centrada + tecnologia * operacion * "
                    "volumen + (1 | bloque)"
                ),
                "codificacion_factores": "contrastes de suma",
                "estimador": "maxima verosimilitud (REML=False)",
                "intercepto_aleatorio": "bloque",
                "optimizadores": "lbfgs; powell",
                "max_iteraciones": 2_000,
                "convergio": convergio,
                "varianza_intercepto_bloque": varianza_bloque,
                "advertencias": advertencias,
                "error": error_ajuste,
                "decision": (
                    "no utilizar para inferencia; se obtuvo una matriz singular "
                    "o no se demostro convergencia. El resultado puede reflejar "
                    "varianza de bloque cercana a cero o dificultades de ajuste, "
                    "sin atribuirle una causa unica."
                ),
            }
        )

        residuos = y - q_completo @ (q_completo.T @ y)

        residuos_est = (residuos - np.mean(residuos)) / np.std(
            residuos, ddof=1
        )
        shapiro_w, shapiro_p = stats.shapiro(residuos_est)
        grupos = []
        for _, grupo in mediciones.assign(
            residuo_estandarizado=residuos_est
        ).groupby(["tecnologia", "operacion", "volumen"], observed=True):
            grupos.append(grupo["residuo_estandarizado"].to_numpy())
        levene_w, levene_p = stats.levene(*grupos, center="median")
        supuestos.append(
            {
                "variable": variable,
                "variable_etiqueta": etiqueta,
                "modelo_mixto_convergio": convergio,
                "shapiro_w": shapiro_w,
                "shapiro_p": shapiro_p,
                "normalidad_residuos": shapiro_p >= ALFA,
                "levene_w": levene_w,
                "levene_p": levene_p,
                "homogeneidad_varianzas": levene_p >= ALFA,
                "supuestos_clasicos_cumplidos": (
                    shapiro_p >= ALFA and levene_p >= ALFA
                ),
                "residuos_evaluados": (
                    "modelo factorial completo con bloque fijo y posicion centrada"
                ),
                "metodo_inferencial_seleccionado": (
                    "MCO factorial con pruebas conjuntas de Wald y covarianza HC3"
                ),
                "motivo_seleccion": (
                    "modelo mixto singular/no convergente y evidencia de "
                    "heterogeneidad; HC3 se definio como analisis de respaldo"
                ),
            }
        )
    return pd.DataFrame(diagnosticos), pd.DataFrame(supuestos)


def ajustar_holm(valores: list[float]) -> list[float]:
    arreglo = np.asarray(valores, dtype=float)
    orden = np.argsort(arreglo)
    ajustados_orden = np.empty(len(arreglo), dtype=float)
    acumulado = 0.0
    m = len(arreglo)
    for posicion, indice in enumerate(orden):
        candidato = (m - posicion) * arreglo[indice]
        acumulado = max(acumulado, candidato)
        ajustados_orden[posicion] = min(acumulado, 1.0)
    resultado = np.empty(len(arreglo), dtype=float)
    resultado[orden] = ajustados_orden
    return resultado.tolist()


def corregir_efectos(efectos: pd.DataFrame) -> pd.DataFrame:
    efectos = efectos.copy()
    efectos["familia_holm"] = efectos["termino"].map(
        lambda termino: f"efecto:{termino}"
    )
    efectos["cantidad_hipotesis_familia"] = 2
    efectos["p_holm_familia_efecto"] = np.nan
    for termino, indices in efectos.groupby("termino").groups.items():
        valores = efectos.loc[indices, "p_sin_ajustar"].tolist()
        if len(valores) != 2:
            raise RuntimeError(
                f"La familia factorial {termino} no contiene dos respuestas."
            )
        efectos.loc[indices, "p_holm_familia_efecto"] = ajustar_holm(valores)
    efectos["significativo_0_05"] = (
        efectos["p_holm_familia_efecto"] < ALFA
    )
    return efectos


def p_signos_exacta(diferencias: np.ndarray) -> float:
    diferencias = diferencias[np.isfinite(diferencias)]
    n = len(diferencias)
    signos = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    distribucion = np.mean(signos * diferencias[None, :], axis=1)
    observado = abs(float(np.mean(diferencias)))
    return float(np.mean(np.abs(distribucion) >= observado - 1e-15))


def efecto_dz_con_ci(
    diferencias: np.ndarray, rng: np.random.Generator
) -> tuple[float, float, float]:
    desviacion = np.std(diferencias, ddof=1)
    dz = float(np.mean(diferencias) / desviacion) if desviacion > 0 else math.inf
    indices = rng.integers(
        0,
        len(diferencias),
        size=(BOOTSTRAP_CONTRASTES, len(diferencias)),
    )
    muestras = diferencias[indices]
    medias = np.mean(muestras, axis=1)
    desvios = np.std(muestras, axis=1, ddof=1)
    efectos = np.divide(
        medias,
        desvios,
        out=np.full_like(medias, np.nan),
        where=desvios > 0,
    )
    validos = efectos[np.isfinite(efectos)]
    if len(validos) == 0:
        return dz, math.nan, math.nan
    ci_inf, ci_sup = np.quantile(validos, [0.025, 0.975])
    return dz, float(ci_inf), float(ci_sup)


def comparaciones_tecnologias(
    mediciones: pd.DataFrame, efectos: pd.DataFrame
) -> pd.DataFrame:
    filas = []
    rng = np.random.default_rng(SEMILLA_BOOTSTRAP_CONTRASTES)
    terminos_tecnologia = {
        "tecnologia",
        "tecnologia:operacion",
        "tecnologia:volumen",
        "tecnologia:operacion:volumen",
    }
    significativos = {
        variable: bool(
            grupo.loc[
                grupo["termino"].isin(terminos_tecnologia),
                "significativo_0_05",
            ].any()
        )
        for variable, grupo in efectos.groupby("variable")
    }

    for variable, etiqueta in RESPUESTAS_PRINCIPALES.items():
        for operacion in OPERACIONES:
            for volumen in VOLUMENES:
                grupo = mediciones.loc[
                    (mediciones["operacion"].astype(str) == operacion)
                    & (mediciones["volumen"].astype(str) == volumen)
                ]
                tabla = grupo.pivot(
                    index="bloque", columns="tecnologia", values=variable
                ).sort_index()
                if tabla.shape != (10, 3) or tabla.isna().any().any():
                    raise RuntimeError(
                        "No se pudieron formar diez pares completos por bloque "
                        f"para {variable}, {operacion}, {volumen}."
                    )
                for tecnologia_a, tecnologia_b in itertools.combinations(
                    TECNOLOGIAS, 2
                ):
                    diferencias = (
                        tabla[tecnologia_a] - tabla[tecnologia_b]
                    ).to_numpy(dtype=float)
                    p = p_signos_exacta(diferencias)
                    dz, ci_inf, ci_sup = efecto_dz_con_ci(diferencias, rng)
                    filas.append(
                        {
                            "variable": variable,
                            "variable_etiqueta": etiqueta,
                            "operacion": operacion,
                            "volumen": volumen,
                            "tecnologia_a": tecnologia_a,
                            "tecnologia_b": tecnologia_b,
                            "n_pares": len(diferencias),
                            "media_a": tabla[tecnologia_a].mean(),
                            "media_b": tabla[tecnologia_b].mean(),
                            "diferencia_media_a_menos_b": np.mean(diferencias),
                            "direccion_diferencia": (
                                f"{tecnologia_a} menos {tecnologia_b}"
                            ),
                            "estadistico_abs_media_diferencias": abs(
                                float(np.mean(diferencias))
                            ),
                            "hipotesis_nula": (
                                "etiquetas intercambiables dentro del bloque; "
                                "diferencias invariantes ante cambio de signo"
                            ),
                            "hipotesis_alternativa": "bilateral",
                            "mecanismo_permutacion": (
                                "enumeracion exhaustiva de cambios de signo"
                            ),
                            "permutaciones_signos": 2 ** len(diferencias),
                            "semilla_permutacion": "no aplica",
                            "p_permutacion_sin_ajustar": p,
                            "d_z_pareado": dz,
                            "d_z_ci95_inferior": ci_inf,
                            "d_z_ci95_superior": ci_sup,
                            "metodo_ic_d_z": (
                                "bootstrap no parametrico de diferencias "
                                "pareadas; percentiles 2,5 y 97,5"
                            ),
                            "remuestreos_d_z": BOOTSTRAP_CONTRASTES,
                            "semilla_bootstrap_d_z": (
                                SEMILLA_BOOTSTRAP_CONTRASTES
                            ),
                            "posthoc_corresponde": significativos[variable],
                        }
                    )

    tabla = pd.DataFrame(filas)
    tabla["familia_holm"] = (
        "escenario:"
        + tabla["operacion"].astype(str)
        + ":"
        + tabla["volumen"].astype(str)
    )
    tabla["cantidad_hipotesis_familia"] = 6
    tabla["p_holm_familia_escenario"] = np.nan
    familia_escenario = ["operacion", "volumen"]
    for _, indices in tabla.groupby(familia_escenario).groups.items():
        p = tabla.loc[indices, "p_permutacion_sin_ajustar"].tolist()
        if len(p) != 6:
            raise RuntimeError(
                "Cada familia posterior debe contener tres pares por dos "
                "variables de respuesta."
            )
        tabla.loc[indices, "p_holm_familia_escenario"] = ajustar_holm(p)
    tabla["significativo_0_05"] = (
        tabla["posthoc_corresponde"]
        & (tabla["p_holm_familia_escenario"] < ALFA)
    )
    return tabla


def resumen_intervalos(mediciones: pd.DataFrame) -> pd.DataFrame:
    return (
        mediciones.groupby("tecnologia", observed=True)[
            "intervalo_real_mediana_ms"
        ]
        .agg(
            n="count",
            mediana_intervalo_real_ms="median",
            minimo_intervalo_real_ms="min",
            maximo_intervalo_real_ms="max",
        )
        .reset_index()
    )


def preparar_graficos(mediciones: pd.DataFrame) -> pd.DataFrame:
    graf = mediciones.copy()
    graf["Tecnología"] = pd.Categorical(
        graf["tecnologia"].astype(str).map(ETIQUETAS_TECNOLOGIA),
        categories=["DuckDB", "Polars", "Apache Spark"],
        ordered=True,
    )
    graf["Operación"] = graf["operacion"].astype(str).map(ETIQUETAS_OPERACION)
    graf["Volumen"] = pd.Categorical(
        graf["volumen"].astype(str).map(ETIQUETAS_VOLUMEN),
        categories=["SF1", "SF5", "SF10"],
        ordered=True,
    )
    return graf


def grafico_lineas(
    graf: pd.DataFrame,
    variable: str,
    etiqueta_y: str,
    ruta: Path,
    logaritmica: bool,
) -> None:
    fig, ejes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for eje, operacion in zip(ejes, OPERACIONES):
        subconjunto = graf.loc[
            graf["operacion"].astype(str) == operacion
        ]
        sns.lineplot(
            data=subconjunto,
            x="Volumen",
            y=variable,
            sort=True,
            hue="Tecnología",
            hue_order=["DuckDB", "Polars", "Apache Spark"],
            estimator="median",
            errorbar=None,
            marker="o",
            linewidth=2,
            ax=eje,
        )
        eje.set_title(ETIQUETAS_OPERACION[operacion])
        eje.set_xlabel("Volumen")
        eje.grid(axis="y", alpha=0.25)
        if logaritmica:
            eje.set_yscale("log")
        if eje is ejes[0]:
            eje.set_ylabel(etiqueta_y)
        else:
            eje.set_ylabel("")
        if eje.get_legend() is not None:
            eje.get_legend().remove()
    handles, labels = ejes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)


def grafico_cajas(
    graf: pd.DataFrame,
    variable: str,
    etiqueta_y: str,
    ruta: Path,
    logaritmica: bool,
) -> None:
    fig, ejes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for eje, operacion in zip(ejes, OPERACIONES):
        subconjunto = graf.loc[
            graf["operacion"].astype(str) == operacion
        ]
        sns.boxplot(
            data=subconjunto,
            x="Volumen",
            y=variable,
            order=["SF1", "SF5", "SF10"],
            hue="Tecnología",
            hue_order=["DuckDB", "Polars", "Apache Spark"],
            showfliers=True,
            ax=eje,
        )
        eje.set_title(ETIQUETAS_OPERACION[operacion])
        eje.set_xlabel("Volumen")
        eje.grid(axis="y", alpha=0.25)
        if logaritmica:
            eje.set_yscale("log")
        if eje is ejes[0]:
            eje.set_ylabel(etiqueta_y)
        else:
            eje.set_ylabel("")
        if eje.get_legend() is not None:
            eje.get_legend().remove()
    handles, labels = ejes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)


def grafico_barras(
    graf: pd.DataFrame,
    variable: str,
    etiqueta_y: str,
    ruta: Path,
    logaritmica: bool,
) -> None:
    fig, ejes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for eje, operacion in zip(ejes, OPERACIONES):
        subconjunto = graf.loc[
            graf["operacion"].astype(str) == operacion
        ]
        sns.barplot(
            data=subconjunto,
            x="Volumen",
            y=variable,
            order=["SF1", "SF5", "SF10"],
            hue="Tecnología",
            hue_order=["DuckDB", "Polars", "Apache Spark"],
            estimator="mean",
            errorbar="sd",
            capsize=0.08,
            ax=eje,
        )
        eje.set_title(ETIQUETAS_OPERACION[operacion])
        eje.set_xlabel("Volumen")
        eje.grid(axis="y", alpha=0.25)
        if logaritmica:
            eje.set_yscale("log")
        if eje is ejes[0]:
            eje.set_ylabel(etiqueta_y)
        else:
            eje.set_ylabel("")
        if eje.get_legend() is not None:
            eje.get_legend().remove()
    handles, labels = ejes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generar_figuras(mediciones: pd.DataFrame) -> None:
    FIGURAS.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    graf = preparar_graficos(mediciones)
    grafico_lineas(
        graf,
        "tiempo_s",
        "Mediana del tiempo (s, escala logarítmica)",
        FIGURAS / "01_lineas_mediana_tiempo.png",
        True,
    )
    grafico_lineas(
        graf,
        "memoria_pico_mb",
        "Mediana del pico de memoria USS (MB)",
        FIGURAS / "02_lineas_mediana_memoria.png",
        False,
    )
    grafico_cajas(
        graf,
        "tiempo_s",
        "Tiempo de ejecución (s, escala logarítmica)",
        FIGURAS / "03_cajas_tiempo.png",
        True,
    )
    grafico_cajas(
        graf,
        "memoria_pico_mb",
        "Pico absoluto de memoria USS (MB)",
        FIGURAS / "04_cajas_memoria.png",
        False,
    )
    grafico_barras(
        graf,
        "tiempo_s",
        "Media del tiempo (s, escala logarítmica)",
        FIGURAS / "05_barras_media_tiempo.png",
        True,
    )
    grafico_barras(
        graf,
        "memoria_pico_mb",
        "Media del pico de memoria USS (MB)",
        FIGURAS / "06_barras_media_memoria.png",
        False,
    )


def guardar_csv(tabla: pd.DataFrame, nombre: str) -> Path:
    ruta = SALIDA / nombre
    tabla.to_csv(ruta, index=False, encoding="utf-8-sig")
    return ruta


def resumen_familias_holm(
    efectos: pd.DataFrame, comparaciones: pd.DataFrame
) -> pd.DataFrame:
    filas: list[dict[str, Any]] = []
    for familia, grupo in efectos.groupby("familia_holm", sort=False):
        filas.append(
            {
                "tipo": "factorial",
                "familia_holm": familia,
                "cantidad_hipotesis": len(grupo),
                "contenido": (
                    "dos respuestas principales para un mismo efecto: "
                    "tiempo y pico absoluto de memoria"
                ),
                "columna_p_original": "p_sin_ajustar",
                "columna_p_ajustada": "p_holm_familia_efecto",
                "orden_ajuste": "Holm aplicado una sola vez a valores p originales",
                "alfa_familiar": ALFA,
            }
        )
    for familia, grupo in comparaciones.groupby("familia_holm", sort=False):
        filas.append(
            {
                "tipo": "comparaciones_pareadas",
                "familia_holm": familia,
                "cantidad_hipotesis": len(grupo),
                "contenido": (
                    "tres pares de tecnologias por dos respuestas dentro "
                    "de una combinacion de operacion y volumen"
                ),
                "columna_p_original": "p_permutacion_sin_ajustar",
                "columna_p_ajustada": "p_holm_familia_escenario",
                "orden_ajuste": "Holm aplicado una sola vez a valores p originales",
                "alfa_familiar": ALFA,
            }
        )
    return pd.DataFrame(filas)


def guardar_especificacion_metodos() -> Path:
    lineas = [
        "ESPECIFICACION REPRODUCIBLE DEL ANALISIS INFERENCIAL",
        "",
        "Modelo factorial de respaldo",
        "- Formula: respuesta ~ bloque + posicion_centrada + tecnologia * operacion * volumen.",
        "- Bloque: efecto fijo mediante nueve indicadores, con el bloque 1 como referencia.",
        "- Posicion: orden dentro del bloque, centrado y dividido por su desviacion estandar poblacional.",
        "- Tecnologia, operacion y volumen: factores de tres niveles codificados con contrastes de suma.",
        "- Estimador: minimos cuadrados ordinarios sobre las 270 mediciones.",
        "- Inferencia: pruebas conjuntas bilaterales de Wald con matriz de covarianza HC3.",
        "- Para cada termino se contrastaron conjuntamente sus coeficientes, manteniendo todos los demas terminos, incluidas las interacciones de orden superior.",
        "- Grados de libertad residuales: n menos el rango de la matriz completa.",
        "",
        "Tamano del efecto factorial",
        "- eta_p2 = (SSE_restringido - SSE_completo) / [(SSE_restringido - SSE_completo) + SSE_completo].",
        "- El modelo restringido fija en cero solamente los coeficientes del termino evaluado y conserva los demas.",
        f"- IC del 95 %: {BOOTSTRAP_EFECTOS:,} remuestreos de los diez bloques completos con reemplazo; percentiles 2,5 y 97,5; semilla {SEMILLA_BOOTSTRAP}.",
        "- Los intervalos son no parametricos y aproximados porque solo se observaron diez bloques.",
        "",
        "Comparaciones entre tecnologias",
        "- Unidad apareada: mediciones de las dos tecnologias pertenecientes al mismo bloque, operacion y volumen.",
        "- Direcciones: DuckDB-Polars, DuckDB-Spark y Polars-Spark.",
        "- Estadistico: valor absoluto de la media de las diez diferencias pareadas.",
        "- Hipotesis nula: las etiquetas son intercambiables dentro de cada bloque y las diferencias son invariantes ante cambios de signo.",
        "- Hipotesis alternativa: bilateral.",
        "- Distribucion nula: enumeracion exhaustiva de 2^10 = 1.024 cambios de signo; no se usa semilla.",
        "- d_z: media de las diferencias dividida por su desviacion estandar muestral.",
        f"- IC de d_z: {BOOTSTRAP_CONTRASTES:,} remuestreos de diferencias pareadas; percentiles 2,5 y 97,5; semilla {SEMILLA_BOOTSTRAP_CONTRASTES}.",
        "",
        "Familias de Holm",
        "- Factorial: siete familias separadas. Cada familia contiene los dos valores p originales, tiempo y memoria, correspondientes a un mismo efecto.",
        "- Posteriores: nueve familias separadas, una por operacion y volumen. Cada familia contiene seis valores p originales: tres pares de tecnologias por dos respuestas.",
        "- Holm se aplica una sola vez dentro de cada familia. No se vuelven a ajustar valores p previamente corregidos.",
        "- El nivel de significacion familiar es 0,05.",
    ]
    ruta = SALIDA / "10_especificacion_metodos.txt"
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return ruta


def guardar_metadatos(
    versiones: dict[str, str],
    datos: pd.DataFrame,
    mediciones: pd.DataFrame,
    supuestos: pd.DataFrame,
) -> Path:
    contenido = {
        "fecha_utc": ahora_iso(),
        "archivo_fuente": str(FUENTE),
        "sha256_fuente": sha256(FUENTE),
        "sesion_id": datos["sesion_id"].iloc[0],
        "filas_fuente": len(datos),
        "mediciones_analizadas": len(mediciones),
        "versiones": versiones,
        "alfa": ALFA,
        "formula_modelo_factorial": (
            "respuesta ~ bloque + posicion_centrada + tecnologia * "
            "operacion * volumen"
        ),
        "estimador": "minimos cuadrados ordinarios",
        "covarianza": "HC3",
        "prueba_factorial": "Wald conjunta con aproximacion F",
        "familias_holm_factoriales": 7,
        "hipotesis_por_familia_factorial": 2,
        "familias_holm_posthoc": 9,
        "hipotesis_por_familia_posthoc": 6,
        "ajuste_holm": "una sola vez sobre valores p originales",
        "permutaciones_pareadas": "enumeracion exhaustiva de 2^10 cambios de signo",
        "semilla_permutaciones_pareadas": None,
        "bootstrap_efectos": BOOTSTRAP_EFECTOS,
        "bootstrap_contrastes": BOOTSTRAP_CONTRASTES,
        "semilla_bootstrap_eta": SEMILLA_BOOTSTRAP,
        "semilla_bootstrap_d_z": SEMILLA_BOOTSTRAP_CONTRASTES,
        "metodo_por_variable": dict(
            zip(
                supuestos["variable"],
                supuestos["metodo_inferencial_seleccionado"],
            )
        ),
    }
    ruta = SALIDA / "00_metadatos_analisis.json"
    with ruta.open("w", encoding="utf-8") as archivo:
        json.dump(contenido, archivo, ensure_ascii=False, indent=2)
        archivo.write("\n")
    return ruta


def guardar_resumen_texto(
    control: pd.DataFrame,
    atipicos: pd.DataFrame,
    supuestos: pd.DataFrame,
    efectos: pd.DataFrame,
    comparaciones: pd.DataFrame,
) -> Path:
    lineas = [
        "RESUMEN AUTOMATICO DEL ANALISIS",
        "",
        f"Controles de integridad correctos: {(control['estado'] == 'correcto').sum()}/{len(control)}",
        f"Valores atipicos señalados y conservados: {len(atipicos)}",
        "",
        "METODO SELECCIONADO POR VARIABLE",
    ]
    for _, fila in supuestos.iterrows():
        lineas.append(
            f"- {fila['variable_etiqueta']}: "
            f"{fila['metodo_inferencial_seleccionado']} "
            f"(Shapiro p={fila['shapiro_p']:.6g}; "
            f"Levene p={fila['levene_p']:.6g})"
        )
    lineas.extend(["", "EFECTOS FACTORIALES"])
    for _, fila in efectos.iterrows():
        lineas.append(
            f"- {fila['variable_etiqueta']} | {fila['termino_etiqueta']}: "
            f"p ajustada={fila['p_holm_familia_efecto']:.6g}; "
            f"eta parcial={fila['eta_cuadrado_parcial']:.6f}"
        )
    lineas.extend(
        [
            "",
            "COMPARACIONES POSTERIORES",
            f"Comparaciones calculadas: {len(comparaciones)}",
            "Comparaciones significativas después de Holm: "
            f"{int(comparaciones['significativo_0_05'].sum())}",
            "",
            "Este archivo informa resultados numéricos. La interpretación "
            "arquitectónica corresponde a la Discusión.",
        ]
    )
    ruta = SALIDA / "resumen_automatico.txt"
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return ruta


def main() -> int:
    print("ANALISIS DEL BENCHMARK")
    print("El archivo benchmark_detalle.csv se utilizara en modo de solo lectura.\n")
    versiones = comprobar_versiones()
    SALIDA.mkdir(parents=True, exist_ok=True)

    datos, mediciones = cargar_datos()
    control = control_integridad(datos, mediciones)
    guardar_csv(control, "01_control_integridad.csv")
    print("[1/6] Integridad: 324 registros y 270 mediciones validas.")

    descriptivos = resumen_descriptivo(mediciones)
    atipicos = detectar_atipicos(mediciones)
    intervalos = resumen_intervalos(mediciones)
    guardar_csv(descriptivos, "02_resumen_descriptivo.csv")
    guardar_csv(atipicos, "03_valores_atipicos_revision.csv")
    guardar_csv(intervalos, "04_intervalos_reales_muestreo.csv")
    print(
        f"[2/6] Descriptivos: 27 escenarios; {len(atipicos)} valores "
        "atipicos señalados sin eliminar."
    )

    diseno = construir_diseno(mediciones)
    diagnostico_mixto, supuestos = modelos_y_supuestos(mediciones, diseno)
    guardar_csv(diagnostico_mixto, "05a_modelo_mixto_diagnostico.csv")
    guardar_csv(supuestos, "06_pruebas_supuestos.csv")
    print("[3/6] Diagnostico del modelo mixto y supuestos completados.")

    print(
        "[4/6] Se ajustaran los modelos factoriales con covarianza HC3 y "
        "9.999 remuestreos de bloques para los intervalos de eta parcial; "
        "esta etapa puede demorar varios minutos.",
        flush=True,
    )
    efectos, coeficientes_robustos = efectos_factoriales_hc3(
        mediciones, diseno
    )
    efectos = corregir_efectos(efectos)
    guardar_csv(
        coeficientes_robustos, "05b_modelo_robusto_coeficientes.csv"
    )
    guardar_csv(efectos, "07_efectos_factoriales.csv")

    comparaciones = comparaciones_tecnologias(mediciones, efectos)
    guardar_csv(comparaciones, "08_comparaciones_tecnologias.csv")
    familias = resumen_familias_holm(efectos, comparaciones)
    guardar_csv(familias, "09_familias_holm.csv")
    guardar_especificacion_metodos()
    print("[5/6] Contrastes pareados y correcciones de Holm completados.")

    generar_figuras(mediciones)
    guardar_metadatos(versiones, datos, mediciones, supuestos)
    guardar_resumen_texto(
        control, atipicos, supuestos, efectos, comparaciones
    )
    print("[6/6] Tablas, metadatos y seis figuras guardados.")

    print("\nRESUMEN")
    for _, fila in supuestos.iterrows():
        print(
            f"{fila['variable_etiqueta']}: "
            f"{fila['metodo_inferencial_seleccionado']}"
        )
    print(f"Mediciones analizadas: {len(mediciones)}")
    print(f"Resultados: {SALIDA}")
    print("\nANALISIS COMPLETADO.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR DEL ANALISIS: {exc}", file=sys.stderr)
        raise SystemExit(1)
