from pathlib import Path
import sys
import duckdb

VOLUMENES_PERMITIDOS = {1, 5, 10}

if len(sys.argv) != 2:
    raise SystemExit("Uso: python generar_tpch.py 1")

try:
    sf = int(sys.argv[1])
except ValueError:
    raise SystemExit("El factor de escala debe ser 1, 5 o 10.")

if sf not in VOLUMENES_PERMITIDOS:
    raise SystemExit("El factor de escala debe ser 1, 5 o 10.")

carpeta_datos = Path("datos")
base_datos = carpeta_datos / f"tpch_sf{sf}.duckdb"
carpeta_parquet = carpeta_datos / f"sf{sf}"

if base_datos.exists() or carpeta_parquet.exists():
    raise SystemExit(
        f"Ya existe información para SF{sf}. "
        "No se sobrescribió ningún archivo."
    )

carpeta_datos.mkdir(exist_ok=True)

conexion = duckdb.connect(str(base_datos))

try:
    conexion.execute("INSTALL tpch")
    conexion.execute("LOAD tpch")

    print(f"Generando TPC-H SF{sf}...")
    conexion.execute(f"CALL dbgen(sf = {sf})")

    ruta_exportacion = (
        carpeta_parquet.resolve().as_posix().replace("'", "''")
    )

    print("Exportando las tablas a formato Parquet...")
    conexion.execute(
        f"""
        EXPORT DATABASE '{ruta_exportacion}'
        (
            FORMAT parquet,
            COMPRESSION snappy,
            ROW_GROUP_SIZE 100_000
        )
        """
    )

    tablas = [
        "customer",
        "lineitem",
        "nation",
        "orders",
        "part",
        "partsupp",
        "region",
        "supplier",
    ]

    print("\nCantidad de filas:")
    for tabla in tablas:
        filas = conexion.execute(
            f'SELECT COUNT(*) FROM "{tabla}"'
        ).fetchone()[0]
        print(f"{tabla}: {filas:,}")

finally:
    conexion.close()

archivos = list(carpeta_parquet.glob("*.parquet"))
tamano_total = sum(archivo.stat().st_size for archivo in archivos)

print(f"\nArchivos Parquet creados: {len(archivos)}")
print(f"Tamaño total: {tamano_total / (1024 ** 3):.3f} GB")
print(f"Ubicación: {carpeta_parquet.resolve()}")
print(f"TPC-H SF{sf} generado correctamente.")