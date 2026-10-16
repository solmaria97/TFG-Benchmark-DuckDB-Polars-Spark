import argparse
import csv
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import date
from decimal import Decimal
from pathlib import Path

import psutil


RAIZ = Path(__file__).resolve().parent
LINEITEM = RAIZ / "datos" / "sf1" / "lineitem.parquet"
RESULTADOS = RAIZ / "resultados"
INTERVALOS_MS = (1, 2, 5)
TECNOLOGIAS = ("duckdb", "polars", "spark")
REPETICIONES = 5
DURACION_BASE_S = 0.5
SEMILLA = 20260922
RESULTADO_ESPERADO = 114_160


def percentil_mediana(valores):
    return statistics.median(valores) if valores else 0.0


def muestra_arbol(proceso_raiz, catalogo):
    procesos = [proceso_raiz]
    try:
        procesos.extend(proceso_raiz.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    unicos = {}
    errores = 0
    for proceso in procesos:
        try:
            creado = proceso.create_time()
            identidad = (proceso.pid, creado)
            if identidad in unicos:
                continue
            memoria = proceso.memory_full_info()
            if not hasattr(memoria, "uss"):
                raise RuntimeError(
                    "psutil no ofrece USS en este sistema. "
                    "No se puede continuar con este piloto."
                )
            uss = int(memoria.uss)
            unicos[identidad] = uss
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
    muestras = []
    cantidad_procesos = []
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
        self.muestras = []
        self.cantidad_procesos = []
        self.errores = 0
        self.detener = threading.Event()
        self.primera_muestra = threading.Event()
        self.hilo = threading.Thread(target=self._ejecutar, daemon=True)

    def _ejecutar(self):
        siguiente = time.perf_counter()
        while not self.detener.is_set():
            instante = time.perf_counter()
            total, cantidad, fallos = muestra_arbol(
                self.proceso_raiz,
                self.catalogo,
            )
            self.muestras.append((instante, total))
            self.cantidad_procesos.append(cantidad)
            self.errores += fallos
            self.primera_muestra.set()

            siguiente += self.intervalo_s
            espera = siguiente - time.perf_counter()
            if espera > 0:
                self.detener.wait(espera)
            else:
                siguiente = time.perf_counter()

        instante = time.perf_counter()
        total, cantidad, fallos = muestra_arbol(
            self.proceso_raiz,
            self.catalogo,
        )
        self.muestras.append((instante, total))
        self.cantidad_procesos.append(cantidad)
        self.errores += fallos

    def iniciar(self):
        self.hilo.start()
        if not self.primera_muestra.wait(timeout=10):
            raise RuntimeError("No se obtuvo la primera muestra de memoria.")

    def finalizar(self):
        self.detener.set()
        self.hilo.join(timeout=10)
        if self.hilo.is_alive():
            raise RuntimeError("El monitor de memoria no pudo finalizar.")


def intervalos_reales_ms(muestras):
    tiempos = [instante for instante, _ in muestras]
    return [
        (actual - anterior) * 1000
        for anterior, actual in zip(tiempos, tiempos[1:])
    ]


def preparar_duckdb():
    import duckdb

    conexion = duckdb.connect(":memory:")
    conexion.execute("SET threads = 6")
    conexion.execute("SET memory_limit = '20GB'")

    def operacion():
        ruta = LINEITEM.as_posix().replace("'", "''")
        return conexion.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{ruta}')
            WHERE l_shipdate >= DATE '1994-01-01'
              AND l_shipdate < DATE '1995-01-01'
              AND l_discount BETWEEN 0.05 AND 0.07
              AND l_quantity < 24
            """
        ).fetchone()[0]

    return operacion, conexion.close


def preparar_polars():
    os.environ["POLARS_MAX_THREADS"] = "6"
    import polars as pl

    def operacion():
        return (
            pl.scan_parquet(LINEITEM)
            .filter(
                (pl.col("l_shipdate") >= date(1994, 1, 1))
                & (pl.col("l_shipdate") < date(1995, 1, 1))
                & pl.col("l_discount").is_between(
                    Decimal("0.05"),
                    Decimal("0.07"),
                    closed="both",
                )
                & (pl.col("l_quantity") < Decimal("24.00"))
            )
            .select(pl.len())
            .collect(engine="in-memory")
            .item()
        )

    return operacion, lambda: None


def preparar_spark():
    os.environ.setdefault(
        "PYSPARK_SUBMIT_ARGS",
        "--driver-memory 20g pyspark-shell",
    )
    from pyspark.sql import SparkSession, functions as F

    spark = (
        SparkSession.builder
        .master("local[6]")
        .appName("piloto_memoria_uss")
        .config("spark.default.parallelism", "6")
        .config("spark.sql.shuffle.partitions", "6")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    def operacion():
        datos = spark.read.parquet(str(LINEITEM))
        return (
            datos
            .filter(
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
            )
            .count()
        )

    return operacion, spark.stop


def ejecutar_hijo(tecnologia, intervalo_ms, repeticion):
    if not LINEITEM.exists():
        raise SystemExit(f"No se encontro: {LINEITEM}")

    preparadores = {
        "duckdb": preparar_duckdb,
        "polars": preparar_polars,
        "spark": preparar_spark,
    }
    operacion, cerrar = preparadores[tecnologia]()

    proceso_raiz = psutil.Process(os.getpid())
    catalogo = {}
    intervalo_s = intervalo_ms / 1000

    muestras_base, procesos_base, errores_base = registrar_durante(
        DURACION_BASE_S,
        intervalo_s,
        proceso_raiz,
        catalogo,
    )
    linea_base = percentil_mediana(
        [memoria for _, memoria in muestras_base]
    )

    monitor = MonitorOperacion(proceso_raiz, intervalo_s, catalogo)
    monitor.iniciar()
    inicio = time.perf_counter()
    resultado = operacion()
    fin = time.perf_counter()
    monitor.finalizar()

    pico = max(memoria for _, memoria in monitor.muestras)
    incremento = pico - linea_base
    reales = intervalos_reales_ms(monitor.muestras)

    procesos = sorted(
        catalogo.values(),
        key=lambda item: (item["ppid"], item["pid"]),
    )
    registro = {
        "tecnologia": tecnologia,
        "intervalo_nominal_ms": intervalo_ms,
        "repeticion": repeticion,
        "resultado": int(resultado),
        "resultado_correcto": int(resultado) == RESULTADO_ESPERADO,
        "tiempo_s": fin - inicio,
        "memoria_base_mb": linea_base / (1024 ** 2),
        "memoria_pico_mb": pico / (1024 ** 2),
        "incremento_memoria_mb": incremento / (1024 ** 2),
        "muestras_base": len(muestras_base),
        "muestras_operacion": len(monitor.muestras),
        "intervalo_real_mediana_ms": percentil_mediana(reales),
        "intervalo_real_maximo_ms": max(reales) if reales else 0.0,
        "procesos_minimos": min(monitor.cantidad_procesos),
        "procesos_maximos": max(monitor.cantidad_procesos),
        "errores_muestreo": errores_base + monitor.errores,
        "procesos": procesos,
    }

    cerrar()
    print("RESULTADO_JSON=" + json.dumps(registro, ensure_ascii=False))


def extraer_resultado(salida):
    for linea in reversed(salida.splitlines()):
        if linea.startswith("RESULTADO_JSON="):
            return json.loads(linea.split("=", 1)[1])
    raise RuntimeError("El proceso hijo no devolvio RESULTADO_JSON.")


def escribir_detalle(registros):
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    ruta = RESULTADOS / "piloto_memoria_detalle.csv"
    campos = [
        "tecnologia",
        "intervalo_nominal_ms",
        "repeticion",
        "resultado",
        "resultado_correcto",
        "tiempo_s",
        "memoria_base_mb",
        "memoria_pico_mb",
        "incremento_memoria_mb",
        "muestras_base",
        "muestras_operacion",
        "intervalo_real_mediana_ms",
        "intervalo_real_maximo_ms",
        "procesos_minimos",
        "procesos_maximos",
        "errores_muestreo",
        "procesos_detectados",
    ]
    with ruta.open("w", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()
        for registro in registros:
            fila = {campo: registro.get(campo, "") for campo in campos}
            fila["procesos_detectados"] = " | ".join(
                f"PID={p['pid']};PPID={p['ppid']};{p['nombre']}"
                for p in registro["procesos"]
            )
            escritor.writerow(fila)
    return ruta


def resumir(registros):
    grupos = defaultdict(list)
    for registro in registros:
        grupos[
            (
                registro["tecnologia"],
                registro["intervalo_nominal_ms"],
            )
        ].append(registro)

    resumen = []
    referencias = {}
    for tecnologia in TECNOLOGIAS:
        grupo = grupos[(tecnologia, 1)]
        referencias[tecnologia] = percentil_mediana(
            [fila["memoria_pico_mb"] for fila in grupo]
        )

    for tecnologia in TECNOLOGIAS:
        for intervalo in INTERVALOS_MS:
            grupo = grupos[(tecnologia, intervalo)]
            pico = percentil_mediana(
                [fila["memoria_pico_mb"] for fila in grupo]
            )
            referencia = referencias[tecnologia]
            diferencia = (
                abs(pico - referencia) / referencia * 100
                if referencia
                else 0.0
            )
            resumen.append(
                {
                    "tecnologia": tecnologia,
                    "intervalo_nominal_ms": intervalo,
                    "repeticiones": len(grupo),
                    "mediana_tiempo_s": percentil_mediana(
                        [fila["tiempo_s"] for fila in grupo]
                    ),
                    "mediana_base_mb": percentil_mediana(
                        [fila["memoria_base_mb"] for fila in grupo]
                    ),
                    "mediana_pico_mb": pico,
                    "mediana_incremento_mb": percentil_mediana(
                        [fila["incremento_memoria_mb"] for fila in grupo]
                    ),
                    "mediana_muestras": percentil_mediana(
                        [fila["muestras_operacion"] for fila in grupo]
                    ),
                    "mediana_intervalo_real_ms": percentil_mediana(
                        [fila["intervalo_real_mediana_ms"] for fila in grupo]
                    ),
                    "diferencia_pico_vs_1ms_pct": diferencia,
                    "resultado_correcto": all(
                        fila["resultado_correcto"] for fila in grupo
                    ),
                    "errores_muestreo": sum(
                        fila["errores_muestreo"] for fila in grupo
                    ),
                }
            )

    seleccion = 1
    for candidato in (5, 2, 1):
        filas = [
            fila
            for fila in resumen
            if fila["intervalo_nominal_ms"] == candidato
        ]
        if all(
            fila["diferencia_pico_vs_1ms_pct"] <= 5
            and fila["resultado_correcto"]
            and fila["errores_muestreo"] == 0
            for fila in filas
        ):
            seleccion = candidato
            break

    ruta = RESULTADOS / "piloto_memoria_resumen.csv"
    campos = list(resumen[0].keys())
    with ruta.open("w", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(resumen)

    return resumen, seleccion, ruta


def ejecutar_controlador():
    if not LINEITEM.exists():
        raise SystemExit(
            f"No se encontro {LINEITEM}. "
            "Ejecute este archivo dentro de TFG_Benchmark."
        )

    tareas = [
        (tecnologia, intervalo, repeticion)
        for tecnologia in TECNOLOGIAS
        for intervalo in INTERVALOS_MS
        for repeticion in range(1, REPETICIONES + 1)
    ]
    random.Random(SEMILLA).shuffle(tareas)

    registros = []
    total = len(tareas)
    for numero, (tecnologia, intervalo, repeticion) in enumerate(
        tareas,
        start=1,
    ):
        print(
            f"[{numero:02d}/{total}] "
            f"{tecnologia.upper()} - {intervalo} ms - "
            f"repeticion {repeticion}"
        )
        proceso = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--hijo",
                "--tecnologia",
                tecnologia,
                "--intervalo-ms",
                str(intervalo),
                "--repeticion",
                str(repeticion),
            ],
            cwd=RAIZ,
            text=True,
            capture_output=True,
            timeout=300,
        )
        if proceso.returncode != 0:
            print(proceso.stdout)
            print(proceso.stderr, file=sys.stderr)
            raise SystemExit(
                f"Fallo {tecnologia}, {intervalo} ms, "
                f"repeticion {repeticion}."
            )
        registro = extraer_resultado(proceso.stdout)
        registros.append(registro)
        print(
            f"    pico={registro['memoria_pico_mb']:.2f} MB, "
            f"base={registro['memoria_base_mb']:.2f} MB, "
            f"muestras={registro['muestras_operacion']}, "
            f"procesos={registro['procesos_minimos']}-"
            f"{registro['procesos_maximos']}"
        )

    detalle = escribir_detalle(registros)
    resumen, seleccion, ruta_resumen = resumir(registros)

    print("\nRESUMEN")
    print(
        "Tecnologia | Intervalo | Pico MB | Dif. vs 1 ms | "
        "Muestras | Intervalo real"
    )
    for fila in resumen:
        print(
            f"{fila['tecnologia']:10s} | "
            f"{fila['intervalo_nominal_ms']:>2} ms | "
            f"{fila['mediana_pico_mb']:>8.2f} | "
            f"{fila['diferencia_pico_vs_1ms_pct']:>7.2f}% | "
            f"{fila['mediana_muestras']:>8.1f} | "
            f"{fila['mediana_intervalo_real_ms']:>7.3f} ms"
        )

    print(f"\nINTERVALO SELECCIONADO: {seleccion} ms")
    print(f"Detalle: {detalle}")
    print(f"Resumen: {ruta_resumen}")
    print(
        "\nAntes de aceptar el resultado, revise en el CSV de detalle "
        "que Spark incluya python.exe y java.exe, y que no existan "
        "errores de muestreo."
    )


def argumentos():
    analizador = argparse.ArgumentParser()
    analizador.add_argument("--hijo", action="store_true")
    analizador.add_argument("--tecnologia", choices=TECNOLOGIAS)
    analizador.add_argument("--intervalo-ms", type=int)
    analizador.add_argument("--repeticion", type=int)
    return analizador.parse_args()


if __name__ == "__main__":
    args = argumentos()
    if args.hijo:
        ejecutar_hijo(
            args.tecnologia,
            args.intervalo_ms,
            args.repeticion,
        )
    else:
        ejecutar_controlador()
