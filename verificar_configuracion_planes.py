"""Verifica configuraciones efectivas y guarda planes de ejecución.

Esta comprobación es complementaria: no forma parte de la muestra experimental,
no mide rendimiento y no modifica los archivos Parquet.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path


RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ / "datos" / "sf1"
RESULTADOS = RAIZ / "resultados"

HILOS = 6
MEMORIA_DUCKDB = "20GB"
MEMORIA_SPARK = "20g"

ARCHIVO_CONFIGURACION = RESULTADOS / "verificacion_configuracion.json"
ARCHIVO_PLANES = RESULTADOS / "planes_ejecucion.txt"


def ruta_sql(ruta: Path) -> str:
    return ruta.resolve().as_posix().replace("'", "''")


def comprobar_archivos() -> None:
    faltantes = [
        DATOS / nombre
        for nombre in ("lineitem.parquet", "orders.parquet", "customer.parquet")
        if not (DATOS / nombre).is_file()
    ]
    if faltantes:
        detalle = "\n".join(f"  - {ruta}" for ruta in faltantes)
        raise SystemExit(f"Faltan archivos requeridos:\n{detalle}")


def consultas_duckdb() -> dict[str, str]:
    lineitem = ruta_sql(DATOS / "lineitem.parquet")
    orders = ruta_sql(DATOS / "orders.parquet")
    customer = ruta_sql(DATOS / "customer.parquet")
    return {
        "filtrado": f"""
            SELECT COUNT(*)
            FROM read_parquet('{lineitem}')
            WHERE l_shipdate >= DATE '1994-01-01'
              AND l_shipdate < DATE '1995-01-01'
              AND l_discount BETWEEN 0.05 AND 0.07
              AND l_quantity < 24
        """,
        "agregacion": f"""
            SELECT
                l_returnflag,
                l_linestatus,
                COUNT(*) AS cantidad_registros,
                SUM(l_quantity) AS suma_cantidad
            FROM read_parquet('{lineitem}')
            GROUP BY l_returnflag, l_linestatus
            ORDER BY l_returnflag, l_linestatus
        """,
        "join": f"""
            SELECT COUNT(*)
            FROM read_parquet('{orders}') AS o
            INNER JOIN read_parquet('{customer}') AS c
                ON o.o_custkey = c.c_custkey
            WHERE o.o_orderdate < DATE '1995-03-15'
              AND c.c_mktsegment = 'BUILDING'
        """,
    }


def verificar_duckdb() -> tuple[dict[str, object], dict[str, str]]:
    import duckdb

    conexion = duckdb.connect(":memory:")
    try:
        conexion.execute(f"SET threads = {HILOS}")
        conexion.execute(f"SET memory_limit = '{MEMORIA_DUCKDB}'")
        hilos, memoria = conexion.execute(
            "SELECT current_setting('threads'), current_setting('memory_limit')"
        ).fetchone()

        planes: dict[str, str] = {}
        for nombre, consulta in consultas_duckdb().items():
            filas = conexion.execute("EXPLAIN " + consulta).fetchall()
            planes[nombre] = "\n".join(
                str(fila[-1]) if len(fila) > 1 else str(fila[0]) for fila in filas
            )

        configuracion = {
            "version": importlib.metadata.version("duckdb"),
            "threads_solicitado": HILOS,
            "threads_efectivo": int(hilos),
            "memory_limit_solicitado": MEMORIA_DUCKDB,
            "memory_limit_efectivo": str(memoria),
            "alcance": (
                "Configuración interna de DuckDB; no constituye un límite "
                "del USS total del proceso."
            ),
        }
        return configuracion, planes
    finally:
        conexion.close()


def verificar_polars() -> tuple[dict[str, object], dict[str, str]]:
    # Debe definirse antes de importar Polars para configurar su pool principal.
    os.environ["POLARS_MAX_THREADS"] = str(HILOS)
    import polars as pl

    lineitem = DATOS / "lineitem.parquet"
    orders = DATOS / "orders.parquet"
    customer = DATOS / "customer.parquet"

    planes_lazy = {
        "filtrado": (
            pl.scan_parquet(lineitem)
            .filter(
                (pl.col("l_shipdate") >= date(1994, 1, 1))
                & (pl.col("l_shipdate") < date(1995, 1, 1))
                & pl.col("l_discount").is_between(
                    Decimal("0.05"), Decimal("0.07"), closed="both"
                )
                & (pl.col("l_quantity") < Decimal("24.00"))
            )
            .select(pl.len().alias("cantidad"))
        ),
        "agregacion": (
            pl.scan_parquet(lineitem)
            .group_by("l_returnflag", "l_linestatus")
            .agg(
                pl.len().alias("cantidad_registros"),
                pl.col("l_quantity").sum().alias("suma_cantidad"),
            )
            .sort("l_returnflag", "l_linestatus")
        ),
        "join": (
            pl.scan_parquet(orders)
            .filter(pl.col("o_orderdate") < date(1995, 3, 15))
            .join(
                pl.scan_parquet(customer).filter(
                    pl.col("c_mktsegment") == "BUILDING"
                ),
                left_on="o_custkey",
                right_on="c_custkey",
                how="inner",
            )
            .select(pl.len().alias("cantidad"))
        ),
    }
    planes = {
        nombre: plan.explain(optimized=True)
        for nombre, plan in planes_lazy.items()
    }
    configuracion = {
        "version": importlib.metadata.version("polars"),
        "variable": "POLARS_MAX_THREADS",
        "valor_solicitado": HILOS,
        "valor_entorno": os.environ.get("POLARS_MAX_THREADS"),
        "thread_pool_size_efectivo": int(pl.thread_pool_size()),
        "limite_global_memoria": None,
        "alcance": (
            "La variable configura el pool principal de Polars; no limita "
            "todos los hilos auxiliares del proceso ni establece un tope global de memoria."
        ),
    }
    return configuracion, planes


def verificar_spark() -> tuple[dict[str, object], dict[str, str]]:
    # Se define antes de importar PySpark y, por lo tanto, antes de iniciar la JVM.
    os.environ["PYSPARK_SUBMIT_ARGS"] = (
        f"--driver-memory {MEMORIA_SPARK} pyspark-shell"
    )

    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    spark = (
        SparkSession.builder.master(f"local[{HILOS}]")
        .appName("verificacion_configuracion_tfg")
        .config("spark.driver.memory", MEMORIA_SPARK)
        .config("spark.default.parallelism", str(HILOS))
        .config("spark.sql.shuffle.partitions", str(HILOS))
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        lineitem = spark.read.parquet(str(DATOS / "lineitem.parquet"))
        orders = spark.read.parquet(str(DATOS / "orders.parquet"))
        customer = spark.read.parquet(str(DATOS / "customer.parquet"))

        planes_df = {
            "filtrado": (
                lineitem.filter(
                    (F.col("l_shipdate") >= F.lit("1994-01-01").cast("date"))
                    & (F.col("l_shipdate") < F.lit("1995-01-01").cast("date"))
                    & F.col("l_discount").between(
                        F.lit("0.05").cast("decimal(15,2)"),
                        F.lit("0.07").cast("decimal(15,2)"),
                    )
                    & (
                        F.col("l_quantity")
                        < F.lit("24.00").cast("decimal(15,2)")
                    )
                ).select(F.count(F.lit(1)).alias("cantidad"))
            ),
            "agregacion": (
                lineitem.groupBy("l_returnflag", "l_linestatus")
                .agg(
                    F.count(F.lit(1)).alias("cantidad_registros"),
                    F.sum("l_quantity").alias("suma_cantidad"),
                )
                .orderBy("l_returnflag", "l_linestatus")
            ),
            "join": (
                orders.filter(
                    F.col("o_orderdate") < F.lit("1995-03-15").cast("date")
                )
                .join(
                    customer.filter(F.col("c_mktsegment") == "BUILDING"),
                    orders["o_custkey"] == customer["c_custkey"],
                    "inner",
                )
                .select(F.count(F.lit(1)).alias("cantidad"))
            ),
        }
        planes = {
            nombre: df._jdf.queryExecution().toString()
            for nombre, df in planes_df.items()
        }

        runtime = spark.sparkContext._jvm.java.lang.Runtime.getRuntime()
        max_heap_bytes = int(runtime.maxMemory())
        argumentos_jvm = list(
            spark.sparkContext._jvm.java.lang.management.ManagementFactory
            .getRuntimeMXBean()
            .getInputArguments()
        )

        configuracion = {
            "version": spark.version,
            "master": spark.sparkContext.master,
            "local_tasks_concurrentes_solicitadas": HILOS,
            "default_parallelism_efectivo": int(
                spark.sparkContext.defaultParallelism
            ),
            "shuffle_partitions_efectivo": int(
                spark.conf.get("spark.sql.shuffle.partitions")
            ),
            "adaptive_query_execution": spark.conf.get(
                "spark.sql.adaptive.enabled"
            ),
            "driver_memory_configurada": spark.conf.get("spark.driver.memory"),
            "jvm_max_heap_bytes": max_heap_bytes,
            "jvm_max_heap_gib": round(max_heap_bytes / (1024 ** 3), 3),
            "argumentos_jvm_memoria": [
                argumento
                for argumento in argumentos_jvm
                if argumento.startswith("-Xmx") or argumento.startswith("-Xms")
            ],
            "pyspark_submit_args": os.environ["PYSPARK_SUBMIT_ARGS"],
            "alcance": (
                "local[6] limita las tareas concurrentes, no todos los hilos de la JVM. "
                "spark.driver.memory se aplica al heap del driver y no al USS total "
                "de Python, la JVM y sus demás procesos descendientes."
            ),
        }
        return configuracion, planes
    finally:
        spark.stop()


def texto_planes(planes: dict[str, dict[str, str]]) -> str:
    partes = [
        "PLANES DE EJECUCIÓN — COMPROBACIÓN COMPLEMENTARIA",
        "Estos planes no corresponden a nuevas mediciones de rendimiento.",
    ]
    for tecnologia in ("duckdb", "polars", "spark"):
        for operacion in ("filtrado", "agregacion", "join"):
            partes.extend(
                [
                    "",
                    "=" * 80,
                    f"{tecnologia.upper()} — {operacion.upper()} — SF1",
                    "=" * 80,
                    planes[tecnologia][operacion],
                ]
            )
    return "\n".join(partes) + "\n"


def main() -> None:
    comprobar_archivos()
    RESULTADOS.mkdir(exist_ok=True)

    print("[1/3] Verificando DuckDB...")
    duckdb_config, duckdb_planes = verificar_duckdb()

    print("[2/3] Verificando Polars...")
    polars_config, polars_planes = verificar_polars()

    print("[3/3] Verificando Spark y su JVM...")
    spark_config, spark_planes = verificar_spark()

    configuracion = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "proposito": (
            "Comprobación complementaria de configuraciones efectivas y planes; "
            "no integra la muestra experimental."
        ),
        "datos": "SF1",
        "duckdb": duckdb_config,
        "polars": polars_config,
        "spark": spark_config,
    }
    planes = {
        "duckdb": duckdb_planes,
        "polars": polars_planes,
        "spark": spark_planes,
    }

    ARCHIVO_CONFIGURACION.write_text(
        json.dumps(configuracion, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    ARCHIVO_PLANES.write_text(texto_planes(planes), encoding="utf-8")

    print("\nVERIFICACIÓN COMPLETADA")
    print(
        "DuckDB: "
        f"threads={duckdb_config['threads_efectivo']}, "
        f"memory_limit={duckdb_config['memory_limit_efectivo']}"
    )
    print(
        "Polars: "
        f"thread_pool_size={polars_config['thread_pool_size_efectivo']}"
    )
    print(
        "Spark: "
        f"master={spark_config['master']}, "
        f"defaultParallelism={spark_config['default_parallelism_efectivo']}, "
        f"driver.memory={spark_config['driver_memory_configurada']}, "
        f"JVM max heap={spark_config['jvm_max_heap_gib']} GiB"
    )
    print(f"Configuración: {ARCHIVO_CONFIGURACION.resolve()}")
    print(f"Planes: {ARCHIVO_PLANES.resolve()}")


if __name__ == "__main__":
    main()
