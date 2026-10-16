"""Verifica la estructura y la coherencia del analisis corregido.

Uso:
    python verificar_analisis_corregido.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


RAIZ = Path(__file__).resolve().parent
FUENTE = RAIZ / "resultados" / "benchmark_detalle.csv"
SALIDA = RAIZ / "resultados" / "analisis_corregido"
REPORTE = SALIDA / "11_verificacion_analisis.txt"


def sha256(ruta: Path) -> str:
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def exigir(condicion: bool, mensaje: str) -> None:
    if not condicion:
        raise RuntimeError(mensaje)


def holm(valores: np.ndarray) -> np.ndarray:
    orden = np.argsort(valores)
    ajustados_ordenados = np.maximum.accumulate(
        (len(valores) - np.arange(len(valores))) * valores[orden]
    )
    resultado = np.empty(len(valores), dtype=float)
    resultado[orden] = np.minimum(ajustados_ordenados, 1.0)
    return resultado


def main() -> int:
    requeridos = [
        FUENTE,
        SALIDA / "00_metadatos_analisis.json",
        SALIDA / "01_control_integridad.csv",
        SALIDA / "02_resumen_descriptivo.csv",
        SALIDA / "03_valores_atipicos_revision.csv",
        SALIDA / "04_intervalos_reales_muestreo.csv",
        SALIDA / "05a_modelo_mixto_diagnostico.csv",
        SALIDA / "05b_modelo_robusto_coeficientes.csv",
        SALIDA / "06_pruebas_supuestos.csv",
        SALIDA / "07_efectos_factoriales.csv",
        SALIDA / "08_comparaciones_tecnologias.csv",
        SALIDA / "09_familias_holm.csv",
        SALIDA / "10_especificacion_metodos.txt",
    ]
    faltantes = [str(ruta) for ruta in requeridos if not ruta.is_file()]
    exigir(not faltantes, "Faltan archivos: " + ", ".join(faltantes))

    datos = pd.read_csv(FUENTE, encoding="utf-8-sig")
    efectos = pd.read_csv(
        SALIDA / "07_efectos_factoriales.csv", encoding="utf-8-sig"
    )
    comparaciones = pd.read_csv(
        SALIDA / "08_comparaciones_tecnologias.csv", encoding="utf-8-sig"
    )
    familias = pd.read_csv(
        SALIDA / "09_familias_holm.csv", encoding="utf-8-sig"
    )
    metadatos = json.loads(
        (SALIDA / "00_metadatos_analisis.json").read_text(encoding="utf-8")
    )

    exigir(len(datos) == 324, "La fuente no contiene 324 registros.")
    incluir = datos["incluir_analisis"].astype(str).str.lower() == "true"
    exigir(int(incluir.sum()) == 270, "La fuente no contiene 270 mediciones.")
    exigir(
        metadatos["sha256_fuente"] == sha256(FUENTE),
        "El hash de la fuente no coincide con los metadatos.",
    )

    exigir(len(efectos) == 14, "Deben existir catorce pruebas factoriales.")
    exigir(
        set(efectos["metodo"])
        == {"MCO factorial; prueba conjunta de Wald con covarianza HC3"},
        "Se encontro un metodo factorial diferente de HC3.",
    )
    exigir(set(efectos["gl_error"]) == {233}, "Los gl del error deben ser 233.")
    gl_esperados = {
        "tecnologia": 2,
        "operacion": 2,
        "volumen": 2,
        "tecnologia:operacion": 4,
        "tecnologia:volumen": 4,
        "operacion:volumen": 4,
        "tecnologia:operacion:volumen": 8,
    }
    for termino, grupo in efectos.groupby("termino"):
        exigir(len(grupo) == 2, f"La familia {termino} no tiene dos respuestas.")
        exigir(
            set(grupo["gl_efecto"]) == {gl_esperados[termino]},
            f"Grados de libertad incorrectos para {termino}.",
        )
        esperado = holm(grupo["p_sin_ajustar"].to_numpy(dtype=float))
        observado = grupo["p_holm_familia_efecto"].to_numpy(dtype=float)
        exigir(
            np.allclose(esperado, observado, rtol=1e-12, atol=1e-15),
            f"Holm factorial incorrecto para {termino}.",
        )
    exigir(
        efectos[
            [
                "eta_cuadrado_parcial",
                "eta_ci95_inferior",
                "eta_ci95_superior",
            ]
        ].apply(lambda columna: columna.between(0, 1).all()).all(),
        "Eta parcial o sus limites quedaron fuera de [0, 1].",
    )

    exigir(len(comparaciones) == 54, "Deben existir 54 comparaciones pareadas.")
    exigir(
        set(comparaciones["n_pares"]) == {10},
        "Todas las comparaciones deben contener diez pares.",
    )
    exigir(
        set(comparaciones["permutaciones_signos"]) == {1024},
        "Cada contraste debe enumerar 1.024 cambios de signo.",
    )
    unidades = comparaciones["p_permutacion_sin_ajustar"] * 1024
    exigir(
        np.allclose(unidades, np.round(unidades), atol=1e-12),
        "Los valores p exactos no son multiplos de 1/1.024.",
    )
    for familia, grupo in comparaciones.groupby("familia_holm"):
        exigir(len(grupo) == 6, f"La familia {familia} no contiene seis pruebas.")
        esperado = holm(
            grupo["p_permutacion_sin_ajustar"].to_numpy(dtype=float)
        )
        observado = grupo["p_holm_familia_escenario"].to_numpy(dtype=float)
        exigir(
            np.allclose(esperado, observado, rtol=1e-12, atol=1e-15),
            f"Holm posterior incorrecto para {familia}.",
        )

    resumen_familias = familias.groupby(
        ["tipo", "cantidad_hipotesis"]
    ).size().to_dict()
    exigir(
        resumen_familias.get(("factorial", 2)) == 7,
        "No se documentaron las siete familias factoriales.",
    )
    exigir(
        resumen_familias.get(("comparaciones_pareadas", 6)) == 9,
        "No se documentaron las nueve familias posteriores.",
    )

    lineas = [
        "VERIFICACION DEL ANALISIS CORREGIDO",
        "",
        "Estado: SUPERADA",
        f"Registros fuente: {len(datos)}",
        f"Mediciones analizadas: {int(incluir.sum())}",
        f"Pruebas factoriales HC3: {len(efectos)}",
        "Familias factoriales de Holm: 7 familias de 2 hipotesis",
        f"Comparaciones pareadas exactas: {len(comparaciones)}",
        "Familias posteriores de Holm: 9 familias de 6 hipotesis",
        "Cambios de signo por contraste: 1.024, enumeracion exhaustiva",
        f"Comparaciones significativas: {int(comparaciones['significativo_0_05'].sum())}/54",
        f"SHA-256 de la fuente: {sha256(FUENTE)}",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("\n".join(lineas))
    print(f"Reporte: {REPORTE}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"VERIFICACION NO SUPERADA: {exc}")
        raise SystemExit(1)
