"""Benchmark reproducible de DuckDB, Polars y Apache Spark.

Primera etapa: validacion de archivos, entorno y equivalencia funcional.
Las ejecuciones de este modo NO miden rendimiento y NO forman parte de la
muestra experimental.

Uso:
    python benchmark_definitivo.py validar
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import threading
import time
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


RAIZ = Path(__file__).resolve().parent
CARPETA_DATOS = RAIZ / "datos"
CARPETA_RESULTADOS = RAIZ / "resultados"

VOLUMENES = ("sf1", "sf5", "sf10")
TECNOLOGIAS = ("duckdb", "polars", "spark")
OPERACIONES = ("filtrado", "agregacion", "join")
ARCHIVOS_REQUERIDOS = {
    "filtrado": ("lineitem.parquet",),
    "agregacion": ("lineitem.parquet",),
    "join": ("orders.parquet", "customer.parquet"),
}

HILOS = 6
MEMORIA_LIMITE = "20GB"
MEMORIA_SPARK = "20g"
TIMEOUT_HIJO_S = 1800
MARCADOR_RESULTADO = "RESULTADO_VALIDACION_JSON="
MARCADOR_MEDICION = "RESULTADO_MEDICION_JSON="

CALENTAMIENTOS = 2
REPETICIONES_MEDIDAS = 10
INTERVALO_MEMORIA_MS = 2
DURACION_BASE_S = 0.5
SEMILLA_EXPERIMENTO = 20260924

RUTA_VALIDACION = CARPETA_RESULTADOS / "validacion_equivalencia.csv"
RUTA_PLAN = CARPETA_RESULTADOS / "plan_benchmark.json"
RUTA_DETALLE = CARPETA_RESULTADOS / "benchmark_detalle.csv"
RUTA_INCIDENCIAS = CARPETA_RESULTADOS / "benchmark_incidencias.csv"
RUTA_METADATOS_BENCHMARK = CARPETA_RESULTADOS / "benchmark_metadatos.json"
RUTA_PRUEBA_MEDICION = CARPETA_RESULTADOS / "prueba_medicion.csv"


def ahora_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def version_paquete(nombre: str) -> str:
    try:
        return importlib.metadata.version(nombre)
    except importlib.metadata.PackageNotFoundError:
        return "NO INSTALADO"


def version_java() -> str:
    try:
        proceso = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return f"NO DISPONIBLE ({exc})"

    texto = "\n".join(
        parte.strip()
        for parte in (proceso.stderr, proceso.stdout)
        if parte.strip()
    )
    return texto.splitlines()[0] if texto else "NO DISPONIBLE"


def recopilar_entorno() -> dict[str, Any]:
    return {
        "fecha_utc": ahora_iso(),
        "python": platform.python_version(),
        "ejecutable_python": sys.executable,
        "sistema": platform.platform(),
        "procesador": platform.processor(),
        "versiones": {
            "duckdb": version_paquete("duckdb"),
            "polars": version_paquete("polars"),
            "pyspark": version_paquete("pyspark"),
            "psutil": version_paquete("psutil"),
            "java": version_java(),
        },
        "configuracion": {
            "hilos": HILOS,
            "memoria_limite_duckdb": MEMORIA_LIMITE,
            "memoria_driver_spark": MEMORIA_SPARK,
            "spark_adaptive_query_execution": True,
        },
    }


def comprobar_dependencias(entorno: dict[str, Any]) -> None:
    faltantes = [
        nombre
        for nombre in ("duckdb", "polars", "pyspark", "psutil")
        if entorno["versiones"][nombre] == "NO INSTALADO"
    ]
    if faltantes:
        raise RuntimeError(
            "Faltan paquetes en el entorno virtual: " + ", ".join(faltantes)
        )
    if entorno["versiones"]["java"].startswith("NO DISPONIBLE"):
        raise RuntimeError(
            "Java no esta disponible en PATH. Spark no puede validarse."
        )


def rutas_operacion(volumen: str, operacion: str) -> dict[str, Path]:
    carpeta = CARPETA_DATOS / volumen
    return {
        nombre: carpeta / nombre
        for nombre in ARCHIVOS_REQUERIDOS[operacion]
    }


def comprobar_archivos() -> list[dict[str, Any]]:
    faltantes: list[Path] = []
    inventario: list[dict[str, Any]] = []
    ya_vistos: set[Path] = set()

    for volumen in VOLUMENES:
        for operacion in OPERACIONES:
            for ruta in rutas_operacion(volumen, operacion).values():
                if ruta in ya_vistos:
                    continue
                ya_vistos.add(ruta)
                if not ruta.is_file():
                    faltantes.append(ruta)
                    continue
                inventario.append(
                    {
                        "volumen": volumen,
                        "archivo": ruta.name,
                        "ruta": str(ruta),
                        "bytes": ruta.stat().st_size,
                    }
                )

    if faltantes:
        detalle = "\n".join(f"  - {ruta}" for ruta in faltantes)
        raise FileNotFoundError(
            "No se encontraron todos los archivos requeridos:\n" + detalle
        )
    return inventario


def sql_ruta(ruta: Path) -> str:
    return ruta.as_posix().replace("'", "''")


def decimal_dos_posiciones(valor: Any) -> str:
    try:
        numero = Decimal(str(valor)).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"No se pudo normalizar el valor decimal {valor!r}") from exc
    return format(numero, ".2f")


def normalizar_agregacion(filas: list[tuple[Any, Any, Any, Any]]) -> list[dict[str, Any]]:
    resultado = [
        {
            "l_returnflag": str(returnflag),
            "l_linestatus": str(linestatus),
            "cantidad_registros": int(cantidad),
            "suma_cantidad": decimal_dos_posiciones(suma),
        }
        for returnflag, linestatus, cantidad, suma in filas
    ]
    return sorted(
        resultado,
        key=lambda fila: (fila["l_returnflag"], fila["l_linestatus"]),
    )


def ejecutar_duckdb(volumen: str, operacion: str) -> Any:
    import duckdb

    rutas = rutas_operacion(volumen, operacion)
    conexion = duckdb.connect(":memory:")
    try:
        conexion.execute(f"SET threads = {HILOS}")
        conexion.execute(f"SET memory_limit = '{MEMORIA_LIMITE}'")

        if operacion == "filtrado":
            lineitem = sql_ruta(rutas["lineitem.parquet"])
            return int(
                conexion.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM read_parquet('{lineitem}')
                    WHERE l_shipdate >= DATE '1994-01-01'
                      AND l_shipdate < DATE '1995-01-01'
                      AND l_discount BETWEEN 0.05 AND 0.07
                      AND l_quantity < 24
                    """
                ).fetchone()[0]
            )

        if operacion == "agregacion":
            lineitem = sql_ruta(rutas["lineitem.parquet"])
            filas = conexion.execute(
                f"""
                SELECT
                    l_returnflag,
                    l_linestatus,
                    COUNT(*) AS cantidad_registros,
                    SUM(l_quantity) AS suma_cantidad
                FROM read_parquet('{lineitem}')
                GROUP BY l_returnflag, l_linestatus
                ORDER BY l_returnflag, l_linestatus
                """
            ).fetchall()
            return normalizar_agregacion(filas)

        orders = sql_ruta(rutas["orders.parquet"])
        customer = sql_ruta(rutas["customer.parquet"])
        return int(
            conexion.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet('{orders}') AS o
                INNER JOIN read_parquet('{customer}') AS c
                    ON o.o_custkey = c.c_custkey
                WHERE o.o_orderdate < DATE '1995-03-15'
                  AND c.c_mktsegment = 'BUILDING'
                """
            ).fetchone()[0]
        )
    finally:
        conexion.close()


def ejecutar_polars(volumen: str, operacion: str) -> Any:
    os.environ["POLARS_MAX_THREADS"] = str(HILOS)
    import polars as pl

    rutas = rutas_operacion(volumen, operacion)

    if operacion == "filtrado":
        return int(
            pl.scan_parquet(rutas["lineitem.parquet"])
            .filter(
                (pl.col("l_shipdate") >= date(1994, 1, 1))
                & (pl.col("l_shipdate") < date(1995, 1, 1))
                & pl.col("l_discount").is_between(
                    Decimal("0.05"), Decimal("0.07"), closed="both"
                )
                & (pl.col("l_quantity") < Decimal("24.00"))
            )
            .select(pl.len().alias("cantidad"))
            .collect(engine="in-memory")
            .item()
        )

    if operacion == "agregacion":
        datos = (
            pl.scan_parquet(rutas["lineitem.parquet"])
            .group_by("l_returnflag", "l_linestatus")
            .agg(
                pl.len().alias("cantidad_registros"),
                pl.col("l_quantity").sum().alias("suma_cantidad"),
            )
            .sort("l_returnflag", "l_linestatus")
            .collect(engine="in-memory")
        )
        filas = [
            (
                fila["l_returnflag"],
                fila["l_linestatus"],
                fila["cantidad_registros"],
                fila["suma_cantidad"],
            )
            for fila in datos.iter_rows(named=True)
        ]
        return normalizar_agregacion(filas)

    orders = pl.scan_parquet(rutas["orders.parquet"])
    customer = pl.scan_parquet(rutas["customer.parquet"])
    return int(
        orders.filter(pl.col("o_orderdate") < date(1995, 3, 15))
        .join(
            customer.filter(pl.col("c_mktsegment") == "BUILDING"),
            left_on="o_custkey",
            right_on="c_custkey",
            how="inner",
        )
        .select(pl.len().alias("cantidad"))
        .collect(engine="in-memory")
        .item()
    )


def crear_spark():
    os.environ.setdefault(
        "PYSPARK_SUBMIT_ARGS",
        f"--driver-memory {MEMORIA_SPARK} pyspark-shell",
    )
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master(f"local[{HILOS}]")
        .appName("validacion_benchmark_tfg")
        .config("spark.driver.memory", MEMORIA_SPARK)
        .config("spark.default.parallelism", str(HILOS))
        .config("spark.sql.shuffle.partitions", str(HILOS))
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def ejecutar_spark(volumen: str, operacion: str) -> Any:
    from pyspark.sql import functions as F

    rutas = rutas_operacion(volumen, operacion)
    spark = crear_spark()
    try:
        if operacion == "filtrado":
            datos = spark.read.parquet(str(rutas["lineitem.parquet"]))
            return int(
                datos.filter(
                    (F.col("l_shipdate") >= F.lit("1994-01-01").cast("date"))
                    & (F.col("l_shipdate") < F.lit("1995-01-01").cast("date"))
                    & F.col("l_discount").between(
                        F.lit("0.05").cast("decimal(15,2)"),
                        F.lit("0.07").cast("decimal(15,2)"),
                    )
                    & (F.col("l_quantity") < F.lit("24.00").cast("decimal(15,2)"))
                ).count()
            )

        if operacion == "agregacion":
            datos = spark.read.parquet(str(rutas["lineitem.parquet"]))
            filas_spark = (
                datos.groupBy("l_returnflag", "l_linestatus")
                .agg(
                    F.count(F.lit(1)).alias("cantidad_registros"),
                    F.sum("l_quantity").alias("suma_cantidad"),
                )
                .orderBy("l_returnflag", "l_linestatus")
                .collect()
            )
            filas = [
                (
                    fila["l_returnflag"],
                    fila["l_linestatus"],
                    fila["cantidad_registros"],
                    fila["suma_cantidad"],
                )
                for fila in filas_spark
            ]
            return normalizar_agregacion(filas)

        orders = spark.read.parquet(str(rutas["orders.parquet"]))
        customer = spark.read.parquet(str(rutas["customer.parquet"]))
        return int(
            orders.filter(
                F.col("o_orderdate") < F.lit("1995-03-15").cast("date")
            )
            .join(
                customer.filter(F.col("c_mktsegment") == "BUILDING"),
                orders["o_custkey"] == customer["c_custkey"],
                "inner",
            )
            .count()
        )
    finally:
        spark.stop()


def ejecutar_hijo(tecnologia: str, volumen: str, operacion: str) -> None:
    funciones = {
        "duckdb": ejecutar_duckdb,
        "polars": ejecutar_polars,
        "spark": ejecutar_spark,
    }
    resultado = funciones[tecnologia](volumen, operacion)
    carga = {
        "tecnologia": tecnologia,
        "volumen": volumen,
        "operacion": operacion,
        "resultado": resultado,
    }
    print(
        MARCADOR_RESULTADO
        + json.dumps(carga, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        flush=True,
    )


def extraer_resultado_hijo(salida: str) -> dict[str, Any]:
    for linea in reversed(salida.splitlines()):
        if linea.startswith(MARCADOR_RESULTADO):
            return json.loads(linea[len(MARCADOR_RESULTADO) :])
    raise RuntimeError("El proceso finalizo sin devolver un resultado reconocible.")


def invocar_hijo(tecnologia: str, volumen: str, operacion: str) -> dict[str, Any]:
    comando = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_ejecutar_validacion",
        "--tecnologia",
        tecnologia,
        "--volumen",
        volumen,
        "--operacion",
        operacion,
    ]
    entorno = os.environ.copy()
    entorno["POLARS_MAX_THREADS"] = str(HILOS)
    entorno["PYSPARK_PYTHON"] = sys.executable
    entorno["PYSPARK_DRIVER_PYTHON"] = sys.executable

    try:
        proceso = subprocess.run(
            comando,
            cwd=RAIZ,
            env=entorno,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_HIJO_S,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"La ejecucion supero {TIMEOUT_HIJO_S} segundos."
        ) from exc

    if proceso.returncode != 0:
        detalle = (proceso.stderr or proceso.stdout or "Sin detalle").strip()
        raise RuntimeError(
            f"El proceso termino con codigo {proceso.returncode}:\n{detalle}"
        )
    return extraer_resultado_hijo(proceso.stdout)


def texto_resultado(resultado: Any) -> str:
    return json.dumps(resultado, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def guardar_csv_validacion(filas: list[dict[str, Any]]) -> Path:
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    ruta = CARPETA_RESULTADOS / "validacion_equivalencia.csv"
    campos = [
        "fecha_utc",
        "volumen",
        "operacion",
        "tecnologia",
        "resultado_json",
        "resultado_referencia_json",
        "equivalente",
        "estado",
        "error",
    ]
    with ruta.open("w", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(filas)
    return ruta


def guardar_metadatos(entorno: dict[str, Any], inventario: list[dict[str, Any]]) -> Path:
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    ruta = CARPETA_RESULTADOS / "metadatos_validacion.json"
    contenido = {**entorno, "archivos": inventario}
    with ruta.open("w", encoding="utf-8") as archivo:
        json.dump(contenido, archivo, ensure_ascii=False, indent=2)
        archivo.write("\n")
    return ruta


def validar() -> int:
    print("VALIDACION PREVIA DEL BENCHMARK")
    print("Estas ejecuciones no forman parte de la muestra experimental.\n")

    entorno = recopilar_entorno()
    comprobar_dependencias(entorno)
    inventario = comprobar_archivos()
    ruta_metadatos = guardar_metadatos(entorno, inventario)

    print("Entorno detectado:")
    print(f"  Python: {entorno['python']}")
    for nombre, version in entorno["versiones"].items():
        print(f"  {nombre}: {version}")
    print(f"\nArchivos requeridos: {len(inventario)}/{len(VOLUMENES) * 3}")
    print("Equivalencia funcional: 27 ejecuciones independientes\n")

    resultados: dict[tuple[str, str], dict[str, Any]] = {}
    errores: dict[tuple[str, str, str], str] = {}
    total = len(VOLUMENES) * len(OPERACIONES) * len(TECNOLOGIAS)
    numero = 0

    for volumen in VOLUMENES:
        for operacion in OPERACIONES:
            clave = (volumen, operacion)
            resultados[clave] = {}
            for tecnologia in TECNOLOGIAS:
                numero += 1
                print(
                    f"[{numero:02d}/{total}] {volumen.upper()} - "
                    f"{operacion} - {tecnologia.upper()}",
                    flush=True,
                )
                try:
                    carga = invocar_hijo(tecnologia, volumen, operacion)
                    resultados[clave][tecnologia] = carga["resultado"]
                    print("    correcto", flush=True)
                except Exception as exc:  # conserva el CSV aunque falle una tecnologia
                    errores[(volumen, operacion, tecnologia)] = str(exc)
                    print(f"    ERROR: {exc}", flush=True)

    filas_csv: list[dict[str, Any]] = []
    escenarios_equivalentes = 0
    for volumen in VOLUMENES:
        for operacion in OPERACIONES:
            por_tecnologia = resultados[(volumen, operacion)]
            referencia = por_tecnologia.get("duckdb")
            escenario_completo = len(por_tecnologia) == len(TECNOLOGIAS)
            equivalente = escenario_completo and all(
                por_tecnologia[tecnologia] == referencia
                for tecnologia in TECNOLOGIAS
            )
            if equivalente:
                escenarios_equivalentes += 1

            for tecnologia in TECNOLOGIAS:
                resultado = por_tecnologia.get(tecnologia)
                error = errores.get((volumen, operacion, tecnologia), "")
                filas_csv.append(
                    {
                        "fecha_utc": ahora_iso(),
                        "volumen": volumen,
                        "operacion": operacion,
                        "tecnologia": tecnologia,
                        "resultado_json": "" if error else texto_resultado(resultado),
                        "resultado_referencia_json": (
                            "" if referencia is None else texto_resultado(referencia)
                        ),
                        "equivalente": equivalente,
                        "estado": "error" if error else "correcto",
                        "error": error,
                    }
                )

    ruta_csv = guardar_csv_validacion(filas_csv)
    print("\nRESUMEN")
    print(f"Escenarios equivalentes: {escenarios_equivalentes}/9")
    print(f"Ejecuciones con error: {len(errores)}/27")
    print(f"Detalle: {ruta_csv}")
    print(f"Metadatos: {ruta_metadatos}")

    if errores or escenarios_equivalentes != 9:
        print("\nVALIDACION NO SUPERADA. No iniciar las mediciones.")
        return 1

    print("\nVALIDACION SUPERADA. Las tres tecnologias dieron resultados equivalentes.")
    return 0


def es_verdadero(valor: Any) -> bool:
    return str(valor).strip().lower() in {"true", "1", "si", "sí"}


def cargar_resultados_esperados() -> dict[tuple[str, str], Any]:
    if not RUTA_VALIDACION.is_file():
        raise FileNotFoundError(
            "No se encontro validacion_equivalencia.csv. "
            "Ejecute primero: python benchmark_definitivo.py validar"
        )

    referencias: dict[tuple[str, str], Any] = {}
    with RUTA_VALIDACION.open("r", newline="", encoding="utf-8-sig") as archivo:
        for fila in csv.DictReader(archivo):
            if (
                fila.get("tecnologia") == "duckdb"
                and fila.get("estado") == "correcto"
                and es_verdadero(fila.get("equivalente"))
            ):
                referencias[(fila["volumen"], fila["operacion"])] = json.loads(
                    fila["resultado_json"]
                )

    esperadas = {
        (volumen, operacion)
        for volumen in VOLUMENES
        for operacion in OPERACIONES
    }
    faltantes = sorted(esperadas - set(referencias))
    if faltantes:
        detalle = ", ".join(f"{v}/{o}" for v, o in faltantes)
        raise RuntimeError(
            "La validacion previa no contiene nueve escenarios equivalentes. "
            f"Faltan: {detalle}"
        )
    return referencias


def preparar_duckdb_medicion(volumen: str, operacion: str):
    import duckdb

    rutas = rutas_operacion(volumen, operacion)
    conexion = duckdb.connect(":memory:")
    conexion.execute(f"SET threads = {HILOS}")
    conexion.execute(f"SET memory_limit = '{MEMORIA_LIMITE}'")

    if operacion == "filtrado":
        lineitem = sql_ruta(rutas["lineitem.parquet"])

        def ejecutar():
            return int(
                conexion.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM read_parquet('{lineitem}')
                    WHERE l_shipdate >= DATE '1994-01-01'
                      AND l_shipdate < DATE '1995-01-01'
                      AND l_discount BETWEEN 0.05 AND 0.07
                      AND l_quantity < 24
                    """
                ).fetchone()[0]
            )

    elif operacion == "agregacion":
        lineitem = sql_ruta(rutas["lineitem.parquet"])

        def ejecutar():
            filas = conexion.execute(
                f"""
                SELECT
                    l_returnflag,
                    l_linestatus,
                    COUNT(*) AS cantidad_registros,
                    SUM(l_quantity) AS suma_cantidad
                FROM read_parquet('{lineitem}')
                GROUP BY l_returnflag, l_linestatus
                ORDER BY l_returnflag, l_linestatus
                """
            ).fetchall()
            return normalizar_agregacion(filas)

    else:
        orders = sql_ruta(rutas["orders.parquet"])
        customer = sql_ruta(rutas["customer.parquet"])

        def ejecutar():
            return int(
                conexion.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM read_parquet('{orders}') AS o
                    INNER JOIN read_parquet('{customer}') AS c
                        ON o.o_custkey = c.c_custkey
                    WHERE o.o_orderdate < DATE '1995-03-15'
                      AND c.c_mktsegment = 'BUILDING'
                    """
                ).fetchone()[0]
            )

    return ejecutar, conexion.close


def preparar_polars_medicion(volumen: str, operacion: str):
    os.environ["POLARS_MAX_THREADS"] = str(HILOS)
    import polars as pl

    rutas = rutas_operacion(volumen, operacion)

    if operacion == "filtrado":

        def ejecutar():
            return int(
                pl.scan_parquet(rutas["lineitem.parquet"])
                .filter(
                    (pl.col("l_shipdate") >= date(1994, 1, 1))
                    & (pl.col("l_shipdate") < date(1995, 1, 1))
                    & pl.col("l_discount").is_between(
                        Decimal("0.05"), Decimal("0.07"), closed="both"
                    )
                    & (pl.col("l_quantity") < Decimal("24.00"))
                )
                .select(pl.len().alias("cantidad"))
                .collect(engine="in-memory")
                .item()
            )

    elif operacion == "agregacion":

        def ejecutar():
            datos = (
                pl.scan_parquet(rutas["lineitem.parquet"])
                .group_by("l_returnflag", "l_linestatus")
                .agg(
                    pl.len().alias("cantidad_registros"),
                    pl.col("l_quantity").sum().alias("suma_cantidad"),
                )
                .sort("l_returnflag", "l_linestatus")
                .collect(engine="in-memory")
            )
            filas = [
                (
                    fila["l_returnflag"],
                    fila["l_linestatus"],
                    fila["cantidad_registros"],
                    fila["suma_cantidad"],
                )
                for fila in datos.iter_rows(named=True)
            ]
            return normalizar_agregacion(filas)

    else:

        def ejecutar():
            orders = pl.scan_parquet(rutas["orders.parquet"])
            customer = pl.scan_parquet(rutas["customer.parquet"])
            return int(
                orders.filter(pl.col("o_orderdate") < date(1995, 3, 15))
                .join(
                    customer.filter(pl.col("c_mktsegment") == "BUILDING"),
                    left_on="o_custkey",
                    right_on="c_custkey",
                    how="inner",
                )
                .select(pl.len().alias("cantidad"))
                .collect(engine="in-memory")
                .item()
            )

    return ejecutar, lambda: None


def preparar_spark_medicion(volumen: str, operacion: str):
    from pyspark.sql import functions as F

    rutas = rutas_operacion(volumen, operacion)
    spark = crear_spark()

    if operacion == "filtrado":

        def ejecutar():
            datos = spark.read.parquet(str(rutas["lineitem.parquet"]))
            return int(
                datos.filter(
                    (F.col("l_shipdate") >= F.lit("1994-01-01").cast("date"))
                    & (F.col("l_shipdate") < F.lit("1995-01-01").cast("date"))
                    & F.col("l_discount").between(
                        F.lit("0.05").cast("decimal(15,2)"),
                        F.lit("0.07").cast("decimal(15,2)"),
                    )
                    & (F.col("l_quantity") < F.lit("24.00").cast("decimal(15,2)"))
                ).count()
            )

    elif operacion == "agregacion":

        def ejecutar():
            datos = spark.read.parquet(str(rutas["lineitem.parquet"]))
            filas_spark = (
                datos.groupBy("l_returnflag", "l_linestatus")
                .agg(
                    F.count(F.lit(1)).alias("cantidad_registros"),
                    F.sum("l_quantity").alias("suma_cantidad"),
                )
                .orderBy("l_returnflag", "l_linestatus")
                .collect()
            )
            filas = [
                (
                    fila["l_returnflag"],
                    fila["l_linestatus"],
                    fila["cantidad_registros"],
                    fila["suma_cantidad"],
                )
                for fila in filas_spark
            ]
            return normalizar_agregacion(filas)

    else:

        def ejecutar():
            orders = spark.read.parquet(str(rutas["orders.parquet"]))
            customer = spark.read.parquet(str(rutas["customer.parquet"]))
            orders_filtradas = orders.filter(
                F.col("o_orderdate") < F.lit("1995-03-15").cast("date")
            )
            clientes_filtrados = customer.filter(
                F.col("c_mktsegment") == "BUILDING"
            )
            return int(
                orders_filtradas.join(
                    clientes_filtrados,
                    orders_filtradas["o_custkey"]
                    == clientes_filtrados["c_custkey"],
                    "inner",
                ).count()
            )

    return ejecutar, spark.stop


def muestra_arbol(proceso_raiz, catalogo: dict[tuple[int, float], dict[str, Any]]):
    procesos = [proceso_raiz]
    try:
        procesos.extend(proceso_raiz.children(recursive=True))
    except Exception as exc:
        import psutil

        if not isinstance(exc, (psutil.NoSuchProcess, psutil.AccessDenied)):
            raise

    unicos: dict[tuple[int, float], int] = {}
    errores = 0
    import psutil

    for proceso in procesos:
        try:
            creado = proceso.create_time()
            identidad = (proceso.pid, creado)
            if identidad in unicos:
                continue
            memoria = proceso.memory_full_info()
            if not hasattr(memoria, "uss"):
                raise RuntimeError(
                    "psutil no ofrece memoria USS en este sistema."
                )
            unicos[identidad] = int(memoria.uss)
            if identidad not in catalogo:
                catalogo[identidad] = {
                    "pid": proceso.pid,
                    "ppid": proceso.ppid(),
                    "nombre": proceso.name(),
                    "creado": creado,
                }
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            errores += 1
    return sum(unicos.values()), len(unicos), errores


def registrar_durante(duracion_s, intervalo_s, proceso_raiz, catalogo):
    muestras: list[tuple[float, int]] = []
    cantidad_procesos: list[int] = []
    errores = 0
    inicio = time.perf_counter()
    siguiente = inicio
    while time.perf_counter() - inicio < duracion_s:
        instante = time.perf_counter()
        total, cantidad, fallos = muestra_arbol(proceso_raiz, catalogo)
        muestras.append((instante, total))
        cantidad_procesos.append(cantidad)
        errores += fallos
        siguiente += intervalo_s
        espera = siguiente - time.perf_counter()
        if espera > 0:
            time.sleep(espera)
        else:
            siguiente = time.perf_counter()
    return muestras, cantidad_procesos, errores


class MonitorOperacion:
    def __init__(self, proceso_raiz, intervalo_s, catalogo):
        self.proceso_raiz = proceso_raiz
        self.intervalo_s = intervalo_s
        self.catalogo = catalogo
        self.muestras: list[tuple[float, int]] = []
        self.cantidad_procesos: list[int] = []
        self.errores = 0
        self.detener = threading.Event()
        self.primera_muestra = threading.Event()
        self.hilo = threading.Thread(target=self._ejecutar, daemon=True)

    def _tomar_muestra(self):
        instante = time.perf_counter()
        total, cantidad, fallos = muestra_arbol(
            self.proceso_raiz, self.catalogo
        )
        self.muestras.append((instante, total))
        self.cantidad_procesos.append(cantidad)
        self.errores += fallos

    def _ejecutar(self):
        siguiente = time.perf_counter()
        while not self.detener.is_set():
            self._tomar_muestra()
            self.primera_muestra.set()
            siguiente += self.intervalo_s
            espera = siguiente - time.perf_counter()
            if espera > 0:
                self.detener.wait(espera)
            else:
                siguiente = time.perf_counter()
        self._tomar_muestra()

    def iniciar(self):
        self.hilo.start()
        if not self.primera_muestra.wait(timeout=10):
            raise RuntimeError("No se obtuvo la primera muestra de memoria.")

    def finalizar(self):
        self.detener.set()
        self.hilo.join(timeout=10)
        if self.hilo.is_alive():
            raise RuntimeError("El monitor de memoria no pudo finalizar.")


def intervalos_reales_ms(muestras: list[tuple[float, int]]) -> list[float]:
    tiempos = [instante for instante, _ in muestras]
    return [
        (actual - anterior) * 1000
        for anterior, actual in zip(tiempos, tiempos[1:])
    ]


def ejecutar_hijo_medicion(tecnologia: str, volumen: str, operacion: str) -> None:
    import psutil

    esperados = cargar_resultados_esperados()
    preparadores = {
        "duckdb": preparar_duckdb_medicion,
        "polars": preparar_polars_medicion,
        "spark": preparar_spark_medicion,
    }
    ejecutar, cerrar = preparadores[tecnologia](volumen, operacion)
    proceso_raiz = psutil.Process(os.getpid())
    catalogo: dict[tuple[int, float], dict[str, Any]] = {}
    intervalo_s = INTERVALO_MEMORIA_MS / 1000

    try:
        muestras_base, _, errores_base = registrar_durante(
            DURACION_BASE_S, intervalo_s, proceso_raiz, catalogo
        )
        memoria_base = statistics.median(
            memoria for _, memoria in muestras_base
        )

        monitor = MonitorOperacion(proceso_raiz, intervalo_s, catalogo)
        monitor.iniciar()
        fecha_inicio = ahora_iso()
        inicio = time.perf_counter()
        try:
            resultado = ejecutar()
            fin = time.perf_counter()
            fecha_fin = ahora_iso()
        finally:
            monitor.finalizar()

        pico = max(memoria for _, memoria in monitor.muestras)
        intervalos = intervalos_reales_ms(monitor.muestras)
        procesos = sorted(
            catalogo.values(), key=lambda item: (item["ppid"], item["pid"])
        )
        nombres = [str(item["nombre"]).lower() for item in procesos]
        registro = {
            "fecha_inicio_utc": fecha_inicio,
            "fecha_fin_utc": fecha_fin,
            "tecnologia": tecnologia,
            "volumen": volumen,
            "operacion": operacion,
            "resultado": resultado,
            "resultado_correcto": resultado == esperados[(volumen, operacion)],
            "tiempo_s": fin - inicio,
            "memoria_base_mb": memoria_base / (1024 ** 2),
            "memoria_pico_mb": pico / (1024 ** 2),
            "incremento_memoria_mb": (pico - memoria_base) / (1024 ** 2),
            "muestras_base": len(muestras_base),
            "muestras_operacion": len(monitor.muestras),
            "intervalo_nominal_ms": INTERVALO_MEMORIA_MS,
            "intervalo_real_mediana_ms": (
                statistics.median(intervalos) if intervalos else 0.0
            ),
            "intervalo_real_maximo_ms": max(intervalos) if intervalos else 0.0,
            "procesos_minimos": min(monitor.cantidad_procesos),
            "procesos_maximos": max(monitor.cantidad_procesos),
            "errores_muestreo": errores_base + monitor.errores,
            "tiene_python": any(nombre.startswith("python") for nombre in nombres),
            "tiene_java": any(nombre.startswith("java") for nombre in nombres),
            "procesos": procesos,
        }
    finally:
        cerrar()

    print(
        MARCADOR_MEDICION
        + json.dumps(registro, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        flush=True,
    )


def crear_tareas_experimento() -> list[dict[str, Any]]:
    rng = random.Random(SEMILLA_EXPERIMENTO)
    tareas: list[dict[str, Any]] = []
    numero_global = 0
    fases = (
        ("calentamiento", CALENTAMIENTOS, False),
        ("medicion", REPETICIONES_MEDIDAS, True),
    )
    for tipo, cantidad_bloques, incluir_analisis in fases:
        for bloque in range(1, cantidad_bloques + 1):
            combinaciones = [
                (tecnologia, volumen, operacion)
                for tecnologia in TECNOLOGIAS
                for volumen in VOLUMENES
                for operacion in OPERACIONES
            ]
            rng.shuffle(combinaciones)
            for orden, (tecnologia, volumen, operacion) in enumerate(
                combinaciones, start=1
            ):
                numero_global += 1
                tarea_id = (
                    f"{tipo[:1].upper()}{bloque:02d}-"
                    f"{volumen}-{operacion}-{tecnologia}"
                )
                tareas.append(
                    {
                        "tarea_id": tarea_id,
                        "numero_global": numero_global,
                        "tipo": tipo,
                        "bloque": bloque,
                        "orden_en_bloque": orden,
                        "incluir_analisis": incluir_analisis,
                        "tecnologia": tecnologia,
                        "volumen": volumen,
                        "operacion": operacion,
                    }
                )
    return tareas


def cargar_o_crear_plan() -> dict[str, Any]:
    tareas_esperadas = crear_tareas_experimento()
    if RUTA_PLAN.is_file():
        with RUTA_PLAN.open("r", encoding="utf-8") as archivo:
            plan = json.load(archivo)
        if (
            plan.get("semilla") != SEMILLA_EXPERIMENTO
            or plan.get("tareas") != tareas_esperadas
        ):
            raise RuntimeError(
                "El plan existente no coincide con la configuracion actual. "
                "No se modifico ningun resultado."
            )
        return plan

    if RUTA_DETALLE.exists():
        raise RuntimeError(
            "Existe benchmark_detalle.csv pero falta plan_benchmark.json. "
            "No es seguro continuar automaticamente."
        )

    plan = {
        "sesion_id": str(uuid.uuid4()),
        "creado_utc": ahora_iso(),
        "semilla": SEMILLA_EXPERIMENTO,
        "calentamientos_por_escenario": CALENTAMIENTOS,
        "mediciones_por_escenario": REPETICIONES_MEDIDAS,
        "total_ejecuciones": len(tareas_esperadas),
        "tareas": tareas_esperadas,
    }
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    with RUTA_PLAN.open("w", encoding="utf-8") as archivo:
        json.dump(plan, archivo, ensure_ascii=False, indent=2)
        archivo.write("\n")
    return plan


CAMPOS_DETALLE = [
    "sesion_id",
    "tarea_id",
    "numero_global",
    "tipo",
    "bloque",
    "orden_en_bloque",
    "incluir_analisis",
    "tecnologia",
    "volumen",
    "operacion",
    "fecha_inicio_utc",
    "fecha_fin_utc",
    "resultado_json",
    "resultado_correcto",
    "tiempo_s",
    "memoria_base_mb",
    "memoria_pico_mb",
    "incremento_memoria_mb",
    "muestras_base",
    "muestras_operacion",
    "intervalo_nominal_ms",
    "intervalo_real_mediana_ms",
    "intervalo_real_maximo_ms",
    "procesos_minimos",
    "procesos_maximos",
    "errores_muestreo",
    "tiene_python",
    "tiene_java",
    "procesos_detectados",
    "estado",
]


def tareas_completadas(sesion_id: str) -> set[str]:
    if not RUTA_DETALLE.is_file():
        return set()
    completadas: set[str] = set()
    with RUTA_DETALLE.open("r", newline="", encoding="utf-8-sig") as archivo:
        for fila in csv.DictReader(archivo):
            if fila.get("sesion_id") != sesion_id:
                raise RuntimeError(
                    "benchmark_detalle.csv pertenece a otra sesion experimental."
                )
            if fila.get("estado") == "correcto":
                completadas.add(fila["tarea_id"])
    return completadas


def anexar_fila_csv(ruta: Path, campos: list[str], fila: dict[str, Any]) -> None:
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    escribir_cabecera = not ruta.exists() or ruta.stat().st_size == 0
    with ruta.open("a", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        if escribir_cabecera:
            escritor.writeheader()
        escritor.writerow({campo: fila.get(campo, "") for campo in campos})
        archivo.flush()
        os.fsync(archivo.fileno())


def invocar_hijo_medicion(tarea: dict[str, Any]) -> dict[str, Any]:
    comando = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_ejecutar_medicion",
        "--tecnologia",
        tarea["tecnologia"],
        "--volumen",
        tarea["volumen"],
        "--operacion",
        tarea["operacion"],
    ]
    entorno = os.environ.copy()
    entorno["POLARS_MAX_THREADS"] = str(HILOS)
    entorno["PYSPARK_PYTHON"] = sys.executable
    entorno["PYSPARK_DRIVER_PYTHON"] = sys.executable
    try:
        proceso = subprocess.run(
            comando,
            cwd=RAIZ,
            env=entorno,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_HIJO_S,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"La ejecucion supero {TIMEOUT_HIJO_S} segundos."
        ) from exc

    if proceso.returncode != 0:
        detalle = (proceso.stderr or proceso.stdout or "Sin detalle").strip()
        raise RuntimeError(
            f"El proceso termino con codigo {proceso.returncode}:\n{detalle}"
        )
    for linea in reversed(proceso.stdout.splitlines()):
        if linea.startswith(MARCADOR_MEDICION):
            return json.loads(linea[len(MARCADOR_MEDICION) :])
    raise RuntimeError("El proceso no devolvio RESULTADO_MEDICION_JSON.")


def motivo_invalidez(tarea: dict[str, Any], registro: dict[str, Any]) -> str:
    if not registro["resultado_correcto"]:
        return "El resultado no coincide con la validacion previa."
    if int(registro["errores_muestreo"]) != 0:
        return f"Se registraron {registro['errores_muestreo']} errores de muestreo."
    if tarea["tecnologia"] == "spark":
        if not registro["tiene_python"] or not registro["tiene_java"]:
            return "El arbol de Spark no incluyo simultaneamente Python y Java."
    return ""


def registrar_incidencia(
    sesion_id: str,
    tarea: dict[str, Any],
    error: str,
    registro: dict[str, Any] | None = None,
) -> None:
    campos = [
        "fecha_utc",
        "sesion_id",
        "tarea_id",
        "numero_global",
        "tipo",
        "bloque",
        "tecnologia",
        "volumen",
        "operacion",
        "error",
        "registro_json",
    ]
    anexar_fila_csv(
        RUTA_INCIDENCIAS,
        campos,
        {
            "fecha_utc": ahora_iso(),
            "sesion_id": sesion_id,
            **tarea,
            "error": error,
            "registro_json": "" if registro is None else texto_resultado(registro),
        },
    )


def guardar_metadatos_benchmark(
    plan: dict[str, Any], entorno: dict[str, Any], inventario: list[dict[str, Any]]
) -> None:
    if RUTA_METADATOS_BENCHMARK.exists():
        return
    contenido = {
        **entorno,
        "sesion_id": plan["sesion_id"],
        "semilla": SEMILLA_EXPERIMENTO,
        "diseno": {
            "escenarios": 27,
            "calentamientos_por_escenario": CALENTAMIENTOS,
            "mediciones_por_escenario": REPETICIONES_MEDIDAS,
            "ejecuciones_totales": 324,
            "ejecuciones_analizadas": 270,
            "orden": "aleatorio dentro de cada bloque",
            "aislamiento": "un proceso independiente por ejecucion",
        },
        "memoria": {
            "metrica": "USS del proceso principal y sus descendientes",
            "intervalo_nominal_ms": INTERVALO_MEMORIA_MS,
            "duracion_linea_base_s": DURACION_BASE_S,
        },
        "archivos": inventario,
    }
    with RUTA_METADATOS_BENCHMARK.open("w", encoding="utf-8") as archivo:
        json.dump(contenido, archivo, ensure_ascii=False, indent=2)
        archivo.write("\n")


def probar_medicion() -> int:
    print("PRUEBA CORTA DEL SISTEMA DE MEDICION")
    print("Estas ejecuciones no integran la muestra experimental.\n", flush=True)

    entorno = recopilar_entorno()
    comprobar_dependencias(entorno)
    comprobar_archivos()
    cargar_resultados_esperados()

    tareas = [
        {
            "tarea_id": f"PRUEBA-sf1-filtrado-{tecnologia}",
            "numero_global": numero,
            "tipo": "prueba",
            "bloque": 0,
            "orden_en_bloque": numero,
            "incluir_analisis": False,
            "tecnologia": tecnologia,
            "volumen": "sf1",
            "operacion": "filtrado",
        }
        for numero, tecnologia in enumerate(TECNOLOGIAS, start=1)
    ]
    filas: list[dict[str, Any]] = []
    errores: list[str] = []

    for tarea in tareas:
        print(
            f"[{tarea['numero_global']}/3] SF1 - filtrado - "
            f"{tarea['tecnologia'].upper()}",
            flush=True,
        )
        try:
            registro = invocar_hijo_medicion(tarea)
            invalidez = motivo_invalidez(tarea, registro)
            estado = "correcto" if not invalidez else "error"
            error = invalidez
            print(
                f"    tiempo={registro['tiempo_s']:.6f} s, "
                f"pico={registro['memoria_pico_mb']:.2f} MB, "
                f"muestras={registro['muestras_operacion']}, "
                f"errores={registro['errores_muestreo']}",
                flush=True,
            )
            filas.append(
                {
                    "fecha_utc": ahora_iso(),
                    "tecnologia": tarea["tecnologia"],
                    "volumen": tarea["volumen"],
                    "operacion": tarea["operacion"],
                    "resultado_correcto": registro["resultado_correcto"],
                    "tiempo_s": registro["tiempo_s"],
                    "memoria_base_mb": registro["memoria_base_mb"],
                    "memoria_pico_mb": registro["memoria_pico_mb"],
                    "incremento_memoria_mb": registro["incremento_memoria_mb"],
                    "muestras_operacion": registro["muestras_operacion"],
                    "intervalo_real_mediana_ms": registro[
                        "intervalo_real_mediana_ms"
                    ],
                    "errores_muestreo": registro["errores_muestreo"],
                    "tiene_python": registro["tiene_python"],
                    "tiene_java": registro["tiene_java"],
                    "estado": estado,
                    "error": error,
                }
            )
            if invalidez:
                errores.append(f"{tarea['tecnologia']}: {invalidez}")
        except Exception as exc:
            errores.append(f"{tarea['tecnologia']}: {exc}")
            filas.append(
                {
                    "fecha_utc": ahora_iso(),
                    "tecnologia": tarea["tecnologia"],
                    "volumen": tarea["volumen"],
                    "operacion": tarea["operacion"],
                    "estado": "error",
                    "error": str(exc),
                }
            )
            print(f"    ERROR: {exc}", flush=True)

    campos = [
        "fecha_utc",
        "tecnologia",
        "volumen",
        "operacion",
        "resultado_correcto",
        "tiempo_s",
        "memoria_base_mb",
        "memoria_pico_mb",
        "incremento_memoria_mb",
        "muestras_operacion",
        "intervalo_real_mediana_ms",
        "errores_muestreo",
        "tiene_python",
        "tiene_java",
        "estado",
        "error",
    ]
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    with RUTA_PRUEBA_MEDICION.open(
        "w", newline="", encoding="utf-8-sig"
    ) as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(
            {campo: fila.get(campo, "") for campo in campos} for fila in filas
        )

    correctas = sum(fila.get("estado") == "correcto" for fila in filas)
    print("\nRESUMEN DE LA PRUEBA")
    print(f"Ejecuciones correctas: {correctas}/3")
    print(f"Ejecuciones con error: {len(errores)}/3")
    print(f"Detalle: {RUTA_PRUEBA_MEDICION}")
    if errores:
        print("\nPRUEBA NO SUPERADA. No iniciar el benchmark definitivo.")
        return 1
    print("\nPRUEBA SUPERADA. El sistema de medicion esta listo.")
    return 0


def ejecutar_benchmark() -> int:
    print("BENCHMARK DEFINITIVO")
    print("Se conservara cada ejecucion valida inmediatamente.\n", flush=True)

    entorno = recopilar_entorno()
    comprobar_dependencias(entorno)
    inventario = comprobar_archivos()
    cargar_resultados_esperados()
    plan = cargar_o_crear_plan()
    guardar_metadatos_benchmark(plan, entorno, inventario)
    sesion_id = plan["sesion_id"]
    completadas = tareas_completadas(sesion_id)
    tareas = plan["tareas"]
    pendientes = [tarea for tarea in tareas if tarea["tarea_id"] not in completadas]

    print(f"Sesion: {sesion_id}")
    print(f"Ejecuciones totales: {len(tareas)}")
    print(f"Ya completadas: {len(completadas)}")
    print(f"Pendientes: {len(pendientes)}\n", flush=True)

    for tarea in pendientes:
        print(
            f"[{tarea['numero_global']:03d}/{len(tareas)}] "
            f"{tarea['tipo']} {tarea['bloque']} - "
            f"{tarea['volumen'].upper()} - {tarea['operacion']} - "
            f"{tarea['tecnologia'].upper()}",
            flush=True,
        )
        try:
            registro = invocar_hijo_medicion(tarea)
        except Exception as exc:
            registrar_incidencia(sesion_id, tarea, str(exc))
            print(f"\nERROR: {exc}", file=sys.stderr)
            print(
                "La ejecucion se detuvo. Los resultados anteriores quedaron "
                "guardados y el mismo comando permitira continuar.",
                file=sys.stderr,
            )
            return 1

        invalidez = motivo_invalidez(tarea, registro)
        if invalidez:
            registrar_incidencia(sesion_id, tarea, invalidez, registro)
            print(f"\nERROR: {invalidez}", file=sys.stderr)
            print(
                "La ejecucion se detuvo sin incorporar esta medicion. "
                "Los resultados anteriores quedaron guardados.",
                file=sys.stderr,
            )
            return 1

        fila = {
            "sesion_id": sesion_id,
            **tarea,
            "fecha_inicio_utc": registro["fecha_inicio_utc"],
            "fecha_fin_utc": registro["fecha_fin_utc"],
            "resultado_json": texto_resultado(registro["resultado"]),
            "resultado_correcto": registro["resultado_correcto"],
            "tiempo_s": registro["tiempo_s"],
            "memoria_base_mb": registro["memoria_base_mb"],
            "memoria_pico_mb": registro["memoria_pico_mb"],
            "incremento_memoria_mb": registro["incremento_memoria_mb"],
            "muestras_base": registro["muestras_base"],
            "muestras_operacion": registro["muestras_operacion"],
            "intervalo_nominal_ms": registro["intervalo_nominal_ms"],
            "intervalo_real_mediana_ms": registro["intervalo_real_mediana_ms"],
            "intervalo_real_maximo_ms": registro["intervalo_real_maximo_ms"],
            "procesos_minimos": registro["procesos_minimos"],
            "procesos_maximos": registro["procesos_maximos"],
            "errores_muestreo": registro["errores_muestreo"],
            "tiene_python": registro["tiene_python"],
            "tiene_java": registro["tiene_java"],
            "procesos_detectados": " | ".join(
                f"PID={p['pid']};PPID={p['ppid']};{p['nombre']}"
                for p in registro["procesos"]
            ),
            "estado": "correcto",
        }
        anexar_fila_csv(RUTA_DETALLE, CAMPOS_DETALLE, fila)
        completadas.add(tarea["tarea_id"])
        print(
            f"    tiempo={registro['tiempo_s']:.6f} s, "
            f"pico={registro['memoria_pico_mb']:.2f} MB, "
            f"muestras={registro['muestras_operacion']}",
            flush=True,
        )

    with RUTA_DETALLE.open("r", newline="", encoding="utf-8-sig") as archivo:
        filas = list(csv.DictReader(archivo))
    calentamientos = sum(fila["tipo"] == "calentamiento" for fila in filas)
    mediciones = sum(es_verdadero(fila["incluir_analisis"]) for fila in filas)
    errores_muestreo = sum(int(fila["errores_muestreo"]) for fila in filas)

    print("\nRESUMEN FINAL")
    print(f"Ejecuciones completadas: {len(filas)}/324")
    print(f"Calentamientos registrados: {calentamientos}/54")
    print(f"Mediciones para el analisis: {mediciones}/270")
    print(f"Errores de muestreo: {errores_muestreo}")
    print(f"Detalle: {RUTA_DETALLE}")
    print(f"Metadatos: {RUTA_METADATOS_BENCHMARK}")
    print("\nBENCHMARK COMPLETADO.")
    return 0


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark reproducible de DuckDB, Polars y Spark: "
            "validacion funcional y mediciones definitivas."
        )
    )
    subparsers = parser.add_subparsers(dest="comando", required=True)
    subparsers.add_parser(
        "validar",
        help="Ejecuta las 27 pruebas funcionales sin medir rendimiento.",
    )
    subparsers.add_parser(
        "probar-medicion",
        help="Comprueba tiempo, memoria y procesos en tres ejecuciones excluidas.",
    )
    subparsers.add_parser(
        "ejecutar",
        help=(
            "Realiza 54 calentamientos y 270 mediciones; permite continuar "
            "una sesion interrumpida."
        ),
    )

    hijo = subparsers.add_parser("_ejecutar_validacion", help=argparse.SUPPRESS)
    hijo.add_argument("--tecnologia", choices=TECNOLOGIAS, required=True)
    hijo.add_argument("--volumen", choices=VOLUMENES, required=True)
    hijo.add_argument("--operacion", choices=OPERACIONES, required=True)

    hijo_medicion = subparsers.add_parser(
        "_ejecutar_medicion", help=argparse.SUPPRESS
    )
    hijo_medicion.add_argument("--tecnologia", choices=TECNOLOGIAS, required=True)
    hijo_medicion.add_argument("--volumen", choices=VOLUMENES, required=True)
    hijo_medicion.add_argument("--operacion", choices=OPERACIONES, required=True)
    return parser


def main() -> int:
    argumentos = crear_parser().parse_args()
    if argumentos.comando == "validar":
        try:
            return validar()
        except Exception as exc:
            print(f"ERROR DE VALIDACION: {exc}", file=sys.stderr)
            return 1

    if argumentos.comando == "ejecutar":
        try:
            return ejecutar_benchmark()
        except Exception as exc:
            print(f"ERROR DEL BENCHMARK: {exc}", file=sys.stderr)
            return 1

    if argumentos.comando == "probar-medicion":
        try:
            return probar_medicion()
        except Exception as exc:
            print(f"ERROR DE LA PRUEBA: {exc}", file=sys.stderr)
            return 1

    if argumentos.comando == "_ejecutar_medicion":
        ejecutar_hijo_medicion(
            argumentos.tecnologia,
            argumentos.volumen,
            argumentos.operacion,
        )
        return 0

    ejecutar_hijo(
        argumentos.tecnologia,
        argumentos.volumen,
        argumentos.operacion,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
