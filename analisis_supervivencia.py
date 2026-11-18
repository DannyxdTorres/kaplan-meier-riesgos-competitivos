# -*- coding: utf-8 -*-
"""
Este código permite analizar las trayectorias académicas
de estudiantes universitarios mediante técnicas de
análisis de supervivencia y riesgos competitivos.

Se utilizan los estimadores Kaplan–Meier y
Aalen–Johansen para estudiar el egreso, el abandono
y la progresión académica, incluyendo el análisis
de asignaturas y semestres.

El código se comparte con fines académicos y de
reproducibilidad científica.

Los datos administrativos individuales no se publican
debido a restricciones de confidencialidad.

Archivos de entrada/
------------------------------------
alumnos.xlsx
    Planilla de estudiantes de la carrera.
    id_alumno, sexo, localidad, anio_ingreso

egresados.xlsx
    La planilla externa de titulados.
    id_alumno, anio_egreso

inscripciones.xlsx
    Inscripciones de los alumnos a una asignatura.
    id_alumno, mat_codigo, anio

calificaciones.xlsx
    Calificaciones de los alumnos por asignatura.
    id_alumno, mat_codigo, calificacion, aprobado, anio

plan_materias.xlsx
    El plan de estudios oficial de la carrera.
    mat_codigo, mat_materia, mat_semestre, mat_curso, mat_credito

actividad.xlsx 
    Se registra si el estudiante a pagado por algunos de estos eventos.
    id_alumno, anio_tfg, anio_pasantia
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


# ---------------------------------------------------------------------------
# Parámetros del estudio publicado
# ---------------------------------------------------------------------------

# Cada cohorte necesita al menos cinco años posibles de seguimiento.
COHORTE_INICIO = 2015
COHORTE_FINAL = 2021

# Cierre administrativo. Quien sigue con actividad en este año está activo.
ANHO_CORTE = 2026

# Duración nominal: cinco años de malla y seis meses de trabajo final.
# El tiempo está en años enteros, así que 5,5 se lee en el escalón del año 5.
PLAN_ANIOS = 5.5

# Asignaturas extraordinarias. No integran la malla regular y, si se contaran,
# marcarían semestres incompletos que en realidad no corresponden al plan.
MATERIAS_EXCLUIDAS = ("QGE12", "QGE22")

# Años que se informan. El año 11 queda fuera: el conjunto de riesgo es de dos.
ANIOS_INFORME = (5, 6, 7, 8, 9, 10)
ANIOS_KM = (5, 8, 10)

# Una asignatura entra al ranking si al menos cinco estudiantes se inscribieron.
MINIMO_MATERIA = 5

BASE = Path(__file__).resolve().parent
ENTRADA = BASE / "entrada"
ELABORADO = BASE / "elaborado"
SALIDA = BASE / "salida"

# Nombres que el programa acepta además del nombre canónico.
ALIAS = {
    "id_alumno": ("id_alumno", "id", "alumno", "alu_cedula", "cedula", "cédula"),
    "sexo": ("sexo", "alu_sexo"),
    "localidad": ("localidad", "alu_localidad"),
    "anio_ingreso": ("anio_ingreso", "año_ingreso", "ins_car_anho_inicio"),
    "anio_egreso": ("anio_egreso", "año_egreso", "anio_egreso"),
    "excluir": ("excluir",),
    "mat_codigo": ("mat_codigo", "codigo", "asignatura"),
    "mat_materia": ("mat_materia", "materia", "nombre"),
    "mat_semestre": ("mat_semestre", "semestre"),
    "mat_curso": ("mat_curso", "curso"),
    "mat_credito": ("mat_credito", "creditos", "créditos"),
    "anio": ("anio", "año", "anho", "ins_mat_anho"),
    "calificacion": ("calificacion", "calificación", "nota", "cal_calificacion"),
    "aprobado": ("aprobado", "cal_aprobado"),
    "anio_tfg": ("anio_tfg", "ultimo_anho_tfg"),
    "anio_pasantia": ("anio_pasantia", "ultimo_anho_pasantia"),
}


# ---------------------------------------------------------------------------
# Lectura de las planillas recogidas
# ---------------------------------------------------------------------------

def _normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Deja los nombres en minúscula y sin espacios, y aplica los alias."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    renombres = {}
    presentes = set(df.columns)
    for canonico, opciones in ALIAS.items():
        if canonico in presentes:
            continue
        for opcion in opciones:
            if opcion in presentes:
                renombres[opcion] = canonico
                break
    return df.rename(columns=renombres)


def _como_id(serie: pd.Series) -> pd.Series:
    """Identificador de texto, estable aunque Excel lo haya guardado como número."""
    numero = pd.to_numeric(serie, errors="coerce")
    texto = serie.astype(str).str.strip()
    texto = texto.mask(numero.notna(), numero.dropna().astype("int64").astype(str).reindex(serie.index))
    texto = texto.mask(serie.isna() | texto.isin(["", "nan", "None", "<NA>"]))
    return texto


def _como_numero(serie: pd.Series) -> pd.Series:
    if serie.dtype == object:
        serie = serie.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(serie, errors="coerce")


def leer_planilla(nombre: str, obligatoria: bool) -> pd.DataFrame | None:
    """Lee un Excel de entrada. Si falta uno obligatorio, avisa y no estima."""
    ruta = ENTRADA / nombre
    if not ruta.exists():
        if obligatoria:
            return None
        return pd.DataFrame()
    df = pd.read_excel(ruta)
    df = _normalizar_columnas(df)
    df = df.dropna(how="all")
    return df


def escribir_plantillas() -> None:
    """Crea las planillas vacías, con una hoja que explica cada columna."""
    ENTRADA.mkdir(parents=True, exist_ok=True)
    plantillas = {
        "alumnos.xlsx": (
            ["id_alumno", "sexo", "localidad", "anio_ingreso", "excluir"],
            "Una fila por estudiante. sexo: Femenino o Masculino. "
            "excluir = 1 deja el registro fuera del estudio.",
        ),
        "egresados.xlsx": (
            ["id_alumno", "anio_egreso"],
            "Solo estudiantes que ya tienen año de titulación.",
        ),
        "inscripciones.xlsx": (
            ["id_alumno", "mat_codigo", "anio"],
            "Una fila por inscripción. Si se inscribió varias veces, van varias filas.",
        ),
        "calificaciones.xlsx": (
            ["id_alumno", "mat_codigo", "calificacion", "aprobado", "anio"],
            "aprobado puede ser SI o NO. calificacion usa la escala 1 a 5; "
            "se aprueba con 2 o más. Puede completarse solo una de las dos.",
        ),
        "plan_materias.xlsx": (
            ["mat_codigo", "mat_materia", "mat_semestre", "mat_curso", "mat_credito"],
            "Asignaturas oficiales del plan y el semestre al que pertenecen.",
        ),
        "actividad.xlsx": (
            ["id_alumno", "anio_tfg", "anio_pasantia"],
            "Opcional. Último año con tutoría, trabajo final o pasantía.",
        ),
    }
    for nombre, (columnas, texto) in plantillas.items():
        destino = ENTRADA / nombre
        if destino.exists():
            continue
        with pd.ExcelWriter(destino) as libro:
            pd.DataFrame(columns=columnas).to_excel(libro, sheet_name="datos", index=False)
            pd.DataFrame({"instruccion": [texto]}).to_excel(
                libro, sheet_name="instrucciones", index=False
            )
        print(f"  Plantilla nueva: {destino.name}")


def _exigir(df: pd.DataFrame | None, nombre: str, columnas: tuple[str, ...]) -> pd.DataFrame:
    if df is None or df.empty:
        raise SystemExit(f"Falta {nombre}, o la hoja está vacía.")
    ausentes = [c for c in columnas if c not in df.columns]
    if ausentes:
        raise SystemExit(f"{nombre} no tiene estas columnas: {', '.join(ausentes)}")
    return df


# ---------------------------------------------------------------------------
# Elaboración del Excel de trabajo
# ---------------------------------------------------------------------------

def _sexo(valor) -> str:
    texto = str(valor).strip().upper()
    if texto in {"F", "FEMENINO", "MUJER"}:
        return "Femenino"
    if texto in {"M", "MASCULINO", "HOMBRE"}:
        return "Masculino"
    return "No informado"


def _localidad(valor) -> str:
    if pd.isna(valor) or str(valor).strip() == "":
        return "Desconocida"
    if "CONCEPC" in str(valor).upper():
        return "Concepción"
    return "Fuera de Concepción"


def _fila_aprobada(calificacion, flag) -> bool:
    """Aprueba si el indicador es SI o si la nota es al menos 2."""
    if pd.notna(flag) and str(flag).strip().upper() in {"SI", "S", "1", "TRUE", "APROBADO"}:
        return True
    if pd.notna(calificacion):
        try:
            return float(calificacion) >= 2
        except (TypeError, ValueError):
            return False
    return False


def elaborar_seguimiento(
    alumnos: pd.DataFrame,
    egresados: pd.DataFrame,
    inscripciones: pd.DataFrame,
    actividad: pd.DataFrame,
) -> pd.DataFrame:
    """
    Una fila por estudiante, con el estado y el reloj del estudio.

    Estados, los mismos del artículo:
      1 egreso, si hay año de titulación;
      0 activo, si no egresó y en el año de cierre se inscribió, o tiene
        tutoría, trabajo final o pasantía en un año no posterior al cierre;
      2 abandono operacional, en cualquier otro caso.
    Un regreso dentro de la ventana queda activo. Una interrupción que ya
    terminó antes del cierre, sin actividad nueva, es abandono.
    """
    base = alumnos.copy()
    base["id_alumno"] = _como_id(base["id_alumno"])
    base["anio_ingreso"] = _como_numero(base["anio_ingreso"])
    if "excluir" not in base.columns:
        base["excluir"] = 0
    base["excluir"] = _como_numero(base["excluir"]).fillna(0)
    base = base[base["excluir"] != 1].copy()

    titulados = egresados.copy()
    titulados["id_alumno"] = _como_id(titulados["id_alumno"])
    titulados["anio_egreso"] = _como_numero(titulados["anio_egreso"])
    titulados = (
        titulados.dropna(subset=["id_alumno"])
        .groupby("id_alumno", as_index=False)["anio_egreso"]
        .min()
    )
    base = base.merge(titulados, on="id_alumno", how="left")

    ins = inscripciones.copy()
    ins["id_alumno"] = _como_id(ins["id_alumno"])
    ins["anio"] = _como_numero(ins["anio"])
    ultima = ins.groupby("id_alumno", as_index=False)["anio"].max()
    ultima = ultima.rename(columns={"anio": "ultimo_anho_inscripcion"})
    base = base.merge(ultima, on="id_alumno", how="left")

    if not actividad.empty and "id_alumno" in actividad.columns:
        act = actividad.copy()
        act["id_alumno"] = _como_id(act["id_alumno"])
        for columna in ("anio_tfg", "anio_pasantia"):
            if columna not in act.columns:
                act[columna] = np.nan
            act[columna] = _como_numero(act[columna])
        act = act.groupby("id_alumno", as_index=False)[["anio_tfg", "anio_pasantia"]].max()
        base = base.merge(act, on="id_alumno", how="left")
    else:
        base["anio_tfg"] = np.nan
        base["anio_pasantia"] = np.nan

    # La ventana publicada. Cohortes más nuevas no tuvieron cinco años posibles.
    base = base[
        base["anio_ingreso"].between(COHORTE_INICIO, COHORTE_FINAL)
    ].copy()

    egreso = base["anio_egreso"].notna()
    activo = (~egreso) & (
        (base["ultimo_anho_inscripcion"] == ANHO_CORTE)
        | (base["anio_tfg"].notna() & (base["anio_tfg"] <= ANHO_CORTE))
        | (base["anio_pasantia"].notna() & (base["anio_pasantia"] <= ANHO_CORTE))
    )
    base["estado"] = np.select([egreso, activo], [1, 0], default=2)

    ultima_actividad = base[
        ["ultimo_anho_inscripcion", "anio_tfg", "anio_pasantia"]
    ].max(axis=1)
    anho_final = np.select(
        [base["estado"] == 1, (base["estado"] == 2) & ultima_actividad.notna()],
        [base["anio_egreso"], ultima_actividad + 1],
        default=ANHO_CORTE,
    )
    anho_final = pd.Series(anho_final, index=base.index)
    sin_actividad = (base["estado"] == 2) & ultima_actividad.isna()
    anho_final.loc[sin_actividad] = base.loc[sin_actividad, "anio_ingreso"] + 1

    base["anho_final"] = anho_final
    base["tiempo"] = base["anho_final"] - base["anio_ingreso"]
    base["sexo"] = base["sexo"].map(_sexo)
    base["grupo_localidad"] = base["localidad"].map(_localidad)
    base["estado_descripcion"] = base["estado"].map(
        {0: "ACTIVO", 1: "EGRESADO", 2: "ABANDONO"}
    )
    base = base[base["tiempo"].notna() & (base["tiempo"] >= 0)].copy()
    return base.sort_values(["anio_ingreso", "id_alumno"]).reset_index(drop=True)


def elaborar_historial(
    inscripciones: pd.DataFrame,
    calificaciones: pd.DataFrame,
    plan: pd.DataFrame,
    seguimiento: pd.DataFrame,
) -> pd.DataFrame:
    """
    Una fila por estudiante y asignatura.

    El año de la primera inscripción abre el reloj de la asignatura. El de
    aprobación es el primer año en que la nota o el indicador dicen que aprobó.
    """
    ins = inscripciones.copy()
    ins["id_alumno"] = _como_id(ins["id_alumno"])
    ins["mat_codigo"] = ins["mat_codigo"].astype(str).str.strip()
    ins["anio"] = _como_numero(ins["anio"])
    ins = ins[~ins["mat_codigo"].isin(MATERIAS_EXCLUIDAS)]
    historia = ins.groupby(["id_alumno", "mat_codigo"], as_index=False).agg(
        anho_primera_inscripcion=("anio", "min"),
        anho_ultima_inscripcion=("anio", "max"),
    )

    notas = calificaciones.copy()
    notas["id_alumno"] = _como_id(notas["id_alumno"])
    notas["mat_codigo"] = notas["mat_codigo"].astype(str).str.strip()
    if "calificacion" not in notas.columns:
        notas["calificacion"] = np.nan
    if "aprobado" not in notas.columns:
        notas["aprobado"] = np.nan
    if "anio" not in notas.columns:
        notas["anio"] = np.nan
    notas["calificacion"] = _como_numero(notas["calificacion"])
    notas["anio"] = _como_numero(notas["anio"])
    notas["pasa"] = [
        _fila_aprobada(nota, flag)
        for nota, flag in zip(notas["calificacion"], notas["aprobado"])
    ]
    aprobadas = notas[notas["pasa"]].groupby(["id_alumno", "mat_codigo"], as_index=False).agg(
        anho_aprobacion=("anio", "min")
    )
    historia = historia.merge(aprobadas, on=["id_alumno", "mat_codigo"], how="left")
    historia["aprobado"] = historia["anho_aprobacion"].notna().astype(int)

    catalogo = plan.copy()
    catalogo["mat_codigo"] = catalogo["mat_codigo"].astype(str).str.strip()
    catalogo = catalogo[~catalogo["mat_codigo"].isin(MATERIAS_EXCLUIDAS)]
    historia = historia.merge(
        catalogo[["mat_codigo", "mat_materia", "mat_semestre"]],
        on="mat_codigo",
        how="left",
    )

    # Solo estudiantes que quedaron en el seguimiento elaborado.
    ids = set(seguimiento["id_alumno"])
    historia = historia[historia["id_alumno"].isin(ids)].copy()
    ingreso = seguimiento.set_index("id_alumno")["anio_ingreso"]
    historia["anio_ingreso"] = historia["id_alumno"].map(ingreso)
    return historia.sort_values(
        ["id_alumno", "mat_semestre", "mat_codigo"]
    ).reset_index(drop=True)


def elaborar_plan(plan: pd.DataFrame) -> pd.DataFrame:
    plan = plan.copy()
    plan["mat_codigo"] = plan["mat_codigo"].astype(str).str.strip()
    plan["mat_semestre"] = _como_numero(plan["mat_semestre"])
    plan = plan[~plan["mat_codigo"].isin(MATERIAS_EXCLUIDAS)]
    plan = plan.dropna(subset=["mat_codigo", "mat_semestre"])
    return plan.sort_values(["mat_semestre", "mat_codigo"]).reset_index(drop=True)


def elaborar() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Lee la recogida y escribe las tres planillas de trabajo."""
    print("Elaboración de las planillas de trabajo")
    alumnos = _exigir(
        leer_planilla("alumnos.xlsx", True),
        "alumnos.xlsx",
        ("id_alumno", "sexo", "localidad", "anio_ingreso"),
    )
    egresados = _exigir(
        leer_planilla("egresados.xlsx", True),
        "egresados.xlsx",
        ("id_alumno", "anio_egreso"),
    )
    inscripciones = _exigir(
        leer_planilla("inscripciones.xlsx", True),
        "inscripciones.xlsx",
        ("id_alumno", "mat_codigo", "anio"),
    )
    calificaciones = _exigir(
        leer_planilla("calificaciones.xlsx", True),
        "calificaciones.xlsx",
        ("id_alumno", "mat_codigo"),
    )
    plan = _exigir(
        leer_planilla("plan_materias.xlsx", True),
        "plan_materias.xlsx",
        ("mat_codigo", "mat_materia", "mat_semestre"),
    )
    actividad = leer_planilla("actividad.xlsx", False)
    if actividad is None:
        actividad = pd.DataFrame()

    seguimiento = elaborar_seguimiento(alumnos, egresados, inscripciones, actividad)
    plan_listo = elaborar_plan(plan)
    historial = elaborar_historial(inscripciones, calificaciones, plan_listo, seguimiento)

    ELABORADO.mkdir(parents=True, exist_ok=True)
    seguimiento.to_excel(ELABORADO / "seguimiento.xlsx", index=False)
    historial.to_excel(ELABORADO / "historial_materias.xlsx", index=False)
    plan_listo.to_excel(ELABORADO / "plan_materias.xlsx", index=False)
    print(f"  Estudiantes: {len(seguimiento)}")
    print(f"  Filas de historial: {len(historial)}")
    print(f"  Asignaturas del plan: {len(plan_listo)}")
    print("  Estas planillas quedan en elaborado/ y no se publican.")
    return seguimiento, historial, plan_listo


# ---------------------------------------------------------------------------
# Incidencia acumulada con empates de un mismo año
# ---------------------------------------------------------------------------

def aalen_johansen(tiempo: np.ndarray, evento: np.ndarray) -> pd.DataFrame:
    """
    Incidencia acumulada de egreso (1) y abandono (2). La censura es 0.

    En un año t, el conjunto de riesgo es quien tiene tiempo mayor o igual
    que t, incluidas las censuras de ese año. Las dos causas usan el mismo
    denominador:

        dF_k(t) = S(t-) * d_k(t) / Y(t)

    Los eventos del mismo año no se ordenan. Hacerlo con el identificador
    cambiaría el conjunto de riesgo sin corresponder a una diferencia de
    calendario.
    """
    orden = np.argsort(tiempo, kind="stable")
    t = np.asarray(tiempo, dtype=float)[orden]
    e = np.asarray(evento, dtype=int)[orden]
    n = t.size
    cortes = np.flatnonzero(np.diff(t)) + 1
    inicios = np.r_[0, cortes]
    fines = np.r_[cortes, n]

    supervivencia = 1.0
    f1 = 0.0
    f2 = 0.0
    filas = []
    for lo, hi in zip(inicios, fines):
        en_riesgo = float(n - lo)
        egresos = float(np.sum(e[lo:hi] == 1))
        abandonos = float(np.sum(e[lo:hi] == 2))
        f1 += supervivencia * egresos / en_riesgo
        f2 += supervivencia * abandonos / en_riesgo
        supervivencia *= 1.0 - (egresos + abandonos) / en_riesgo
        filas.append(
            {
                "anio": float(t[lo]),
                "en_riesgo": en_riesgo,
                "egresos": egresos,
                "abandonos": abandonos,
                "censuras": float(hi - lo) - egresos - abandonos,
                "cif_egreso": f1,
                "cif_abandono": f2,
            }
        )
    return pd.DataFrame(filas)


def valor_al_cierre(tabla: pd.DataFrame, anio: float, columna: str) -> float:
    """Valor de la función escalonada al terminar el año, no antes del salto."""
    previos = tabla.loc[tabla["anio"] <= anio + 1e-9, columna]
    if previos.empty:
        return 0.0
    return float(previos.iloc[-1])


def _redondear(valor: float) -> float:
    """Redondeo independiente a cuatro decimales. No se fuerza la suma a 1."""
    return float(f"{valor:.4f}")


def tabla_incidencia(tabla: pd.DataFrame) -> pd.DataFrame:
    """
    La incidencia de un año sin eventos conserva el valor ya alcanzado.
    El conjunto de riesgo es quien todavía tiene tiempo mayor o igual que
    ese año, aunque ese año no traiga una fila propia.
    """
    filas = []
    for anio in ANIOS_INFORME:
        egreso = valor_al_cierre(tabla, anio, "cif_egreso")
        abandono = valor_al_cierre(tabla, anio, "cif_abandono")
        retencion = 1.0 - egreso - abandono
        posteriores = tabla.loc[tabla["anio"] >= anio - 1e-9]
        en_riesgo = 0 if posteriores.empty else int(posteriores.iloc[0]["en_riesgo"])
        filas.append(
            {
                "anio": anio,
                "en_riesgo": en_riesgo,
                "cif_egreso": _redondear(egreso),
                "cif_abandono": _redondear(abandono),
                "retencion": _redondear(retencion),
            }
        )
    return pd.DataFrame(filas)


def intervalos_pycmprsk(tiempo, evento) -> pd.DataFrame | None:
    """
    Intervalos log-log con la varianza de Aalen que calcula pycmprsk.

    Si el paquete no está instalado, el informe sigue con las estimaciones
    puntuales. No se reemplazan por intervalos binomiales.
    """
    try:
        from pycmprsk import cuminc
    except ImportError:
        return None

    ajuste = cuminc(np.asarray(tiempo, float), np.asarray(evento, int), cencode=0)
    z = 1.959963984540054
    filas = []
    for anio in ANIOS_INFORME:
        for causa, nombre in (("1 1", "egreso"), ("1 2", "abandono")):
            curva = ajuste.curves[causa]
            mascara = curva.time <= anio + 1e-9
            if not np.any(mascara):
                bajo, alto = np.nan, np.nan
            else:
                i = int(np.flatnonzero(mascara)[-1])
                est, var = float(curva.est[i]), float(curva.var[i])
                if est <= 0.0 or est >= 1.0 or var <= 0.0:
                    bajo, alto = np.nan, np.nan
                else:
                    centro = np.log(-np.log(est))
                    se = np.sqrt(var) / (est * abs(np.log(est)))
                    bajo = float(np.exp(-np.exp(centro + z * se)))
                    alto = float(np.exp(-np.exp(centro - z * se)))
                    bajo, alto = min(bajo, alto), max(bajo, alto)
            filas.append(
                {
                    "anio": anio,
                    f"{nombre}_ic_inf": None if np.isnan(bajo) else _redondear(bajo),
                    f"{nombre}_ic_sup": None if np.isnan(alto) else _redondear(alto),
                }
            )
    egreso = pd.DataFrame([f for f in filas if "egreso_ic_inf" in f])
    abandono = pd.DataFrame([f for f in filas if "abandono_ic_inf" in f])
    return egreso.merge(abandono, on="anio")


def gray_por_sexo(tiempo, evento, sexo) -> dict | None:
    """Prueba de Gray para la incidencia de egreso. None si no hay pycmprsk."""
    try:
        from pycmprsk import cuminc
    except ImportError:
        return None
    ajuste = cuminc(
        np.asarray(tiempo, float),
        np.asarray(evento, int),
        group=np.asarray(sexo),
        cencode=0,
    )
    estadistico, p_valor, grados = (float(v) for v in ajuste.tests[0])
    return {
        "prueba": "Gray",
        "causa": "egreso",
        "estadistico": round(estadistico, 4),
        "grados_libertad": int(grados),
        "p": round(p_valor, 4),
    }


# ---------------------------------------------------------------------------
# Kaplan-Meier y log-rank
# ---------------------------------------------------------------------------

def ajustar_km(tiempo, evento, etiqueta: str) -> KaplanMeierFitter:
    return KaplanMeierFitter(label=etiqueta).fit(tiempo, evento)


def complemento_km(kmf: KaplanMeierFitter, anios=ANIOS_KM) -> dict[int, float]:
    """1 - S(t). Describe el estimador habitual; no es la probabilidad de egreso."""
    curva = kmf.survival_function_.iloc[:, 0]
    valores = {}
    for anio in anios:
        previos = curva.index[curva.index <= anio]
        supervivencia = 1.0 if len(previos) == 0 else float(curva.loc[previos[-1]])
        valores[anio] = _redondear(1.0 - supervivencia)
    return valores


def mediana_km(kmf: KaplanMeierFitter) -> float:
    mediana = float(kmf.median_survival_time_)
    if np.isinf(mediana):
        return np.nan
    return mediana


def logrank(df: pd.DataFrame, columna: str, tiempo: str, evento: str) -> float:
    """p del log-rank entre mujeres y hombres, o entre los dos grupos presentes."""
    if columna == "sexo":
        grupos = [df[df[columna] == sexo] for sexo in ("Femenino", "Masculino")]
        if any(grupo.empty for grupo in grupos):
            return np.nan
    else:
        grupos = [grupo for _, grupo in df.groupby(columna) if len(grupo) > 0]
        if len(grupos) != 2:
            return np.nan
    a, b = grupos
    resultado = logrank_test(a[tiempo], b[tiempo], a[evento], b[evento])
    return round(float(resultado.p_value), 4)


def benjamini_hochberg(valores: pd.Series) -> pd.Series:
    """p ajustados. El artículo los aplicó a las 58 pruebas de sexo por asignatura."""
    p = valores.to_numpy(dtype=float)
    ajustados = np.full(p.shape, np.nan)
    validos = np.isfinite(p)
    if validos.sum() == 0:
        return pd.Series(ajustados, index=valores.index)
    orden = np.argsort(p[validos])
    crudos = p[validos][orden]
    n = crudos.size
    rango = np.arange(1, n + 1)
    corregidos = np.minimum.accumulate((crudos * n / rango)[::-1])[::-1]
    corregidos = np.clip(corregidos, 0, 1)
    destino = np.empty(n)
    destino[orden] = corregidos
    ajustados[np.flatnonzero(validos)] = destino
    return pd.Series(ajustados, index=valores.index)


# ---------------------------------------------------------------------------
# Ruta 1. Trayectorias
# ---------------------------------------------------------------------------

def analizar_trayectorias(seguimiento: pd.DataFrame) -> dict[str, pd.DataFrame]:
    print("\nTrayectorias: egreso, abandono y retención")
    df = seguimiento.dropna(subset=["tiempo", "estado", "sexo"]).copy()
    df["estado"] = df["estado"].astype(int)
    tiempo = df["tiempo"].to_numpy(float)
    evento = df["estado"].to_numpy(int)

    incidencia = tabla_incidencia(aalen_johansen(tiempo, evento))
    intervalos = intervalos_pycmprsk(tiempo, evento)
    if intervalos is not None:
        incidencia = incidencia.merge(intervalos, on="anio", how="left")

    evento_egreso = (df["estado"] == 1).astype(int)
    km = ajustar_km(tiempo, evento_egreso, "Egreso")
    complementos = complemento_km(km)
    comparacion = []
    for anio in ANIOS_KM:
        aj = float(incidencia.loc[incidencia["anio"] == anio, "cif_egreso"].iloc[0])
        comparacion.append(
            {
                "anio": anio,
                "complemento_km": complementos[anio],
                "cif_egreso": aj,
                "diferencia_puntos": round((complementos[anio] - aj) * 100, 1),
            }
        )

    filas_sexo = []
    for sexo, grupo in df.groupby("sexo"):
        if sexo == "No informado" or len(grupo) < 5:
            continue
        cif = aalen_johansen(grupo["tiempo"].to_numpy(float), grupo["estado"].to_numpy(int))
        km_sexo = ajustar_km(
            grupo["tiempo"], (grupo["estado"] == 1).astype(int), str(sexo)
        )
        fila = {
            "sexo": sexo,
            "n": len(grupo),
            "egresos": int((grupo["estado"] == 1).sum()),
            "mediana_km": mediana_km(km_sexo),
        }
        for anio in ANIOS_KM:
            fila[f"cif_egreso_anio_{anio}"] = _redondear(
                valor_al_cierre(cif, anio, "cif_egreso")
            )
        filas_sexo.append(fila)
    por_sexo = pd.DataFrame(filas_sexo)
    # En el log-rank el evento es solo la titulación: el abandono se censura.
    df = df.copy()
    df["evento_egreso"] = (df["estado"] == 1).astype(int)
    p_logrank = logrank(df, "sexo", "tiempo", "evento_egreso")

    gray = gray_por_sexo(tiempo, evento, df["sexo"].astype(str).to_numpy())
    pruebas = pd.DataFrame(
        [
            {
                "prueba": "log-rank",
                "que_compara": "Kaplan-Meier del egreso por sexo; el abandono se censura",
                "estadistico": np.nan,
                "grados_libertad": 1,
                "p": p_logrank,
            }
        ]
    )
    if gray is not None:
        pruebas = pd.concat(
            [
                pruebas,
                pd.DataFrame(
                    [
                        {
                            "prueba": "Gray",
                            "que_compara": "Incidencia acumulada de egreso por sexo",
                            "estadistico": gray["estadistico"],
                            "grados_libertad": gray["grados_libertad"],
                            "p": gray["p"],
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    conteo = (
        df.groupby(["anio_ingreso", "estado_descripcion"]).size().unstack(fill_value=0).reset_index()
    )
    for columna in ("ACTIVO", "EGRESADO", "ABANDONO"):
        if columna not in conteo.columns:
            conteo[columna] = 0
    conteo["total"] = conteo[["ACTIVO", "EGRESADO", "ABANDONO"]].sum(axis=1)
    conteo["pct_egresado"] = (100 * conteo["EGRESADO"] / conteo["total"]).round(1)
    conteo["pct_abandono"] = (100 * conteo["ABANDONO"] / conteo["total"]).round(1)
    conteo["pct_activo"] = (100 * conteo["ACTIVO"] / conteo["total"]).round(1)
    cohortes = conteo.rename(columns={"anio_ingreso": "cohorte"}).sort_values("cohorte")

    _figura_incidencia(aalen_johansen(tiempo, evento), km)
    return {
        "incidencia": incidencia,
        "km_vs_aj": pd.DataFrame(comparacion),
        "sexo": por_sexo,
        "pruebas": pruebas,
        "cohortes": cohortes,
    }


def _figura_incidencia(tabla: pd.DataFrame, km: KaplanMeierFitter) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(tabla["anio"], tabla["cif_egreso"], where="post", label="Egreso (Aalen-Johansen)")
    ax.step(
        tabla["anio"], tabla["cif_abandono"], where="post", linestyle="--", label="Abandono (Aalen-Johansen)"
    )
    curva = 1 - km.survival_function_.iloc[:, 0]
    ax.step(curva.index, curva.values, where="post", linestyle=":", label="Complemento de Kaplan-Meier")
    ax.axvspan(5, 5.5, color="0.85", label="Duración nominal (5,5 años)")
    ax.set_xlabel("Años desde el ingreso")
    ax.set_ylabel("Probabilidad acumulada")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(SALIDA / "incidencia.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Ruta 2. Asignaturas
# ---------------------------------------------------------------------------

def analizar_asignaturas(seguimiento: pd.DataFrame, historial: pd.DataFrame) -> dict[str, pd.DataFrame]:
    print("\nAsignaturas: tiempo hasta aprobar")
    reloj = seguimiento.set_index("id_alumno")[["anho_final", "sexo", "grupo_localidad"]]
    df = historial.join(reloj, on="id_alumno", how="inner")
    df["evento"] = df["aprobado"].fillna(0).astype(int)
    df["tiempo_inscripcion"] = np.where(
        df["evento"] == 1,
        df["anho_aprobacion"] - df["anho_primera_inscripcion"],
        df["anho_final"] - df["anho_primera_inscripcion"],
    )
    df["tiempo_carrera"] = np.where(
        df["evento"] == 1,
        df["anho_aprobacion"] - df["anio_ingreso"],
        df["anho_final"] - df["anio_ingreso"],
    )
    df = df[
        df["tiempo_inscripcion"].notna()
        & (df["tiempo_inscripcion"] >= 0)
        & df["tiempo_carrera"].notna()
        & (df["tiempo_carrera"] >= 0)
    ].copy()

    filas = []
    pruebas = []
    for (codigo, nombre, semestre), grupo in df.groupby(
        ["mat_codigo", "mat_materia", "mat_semestre"], dropna=False
    ):
        if len(grupo) < MINIMO_MATERIA or pd.isna(nombre):
            continue
        km_ins = ajustar_km(grupo["tiempo_inscripcion"], grupo["evento"], str(codigo))
        km_car = ajustar_km(grupo["tiempo_carrera"], grupo["evento"], str(codigo))
        filas.append(
            {
                "mat_codigo": codigo,
                "mat_materia": nombre,
                "mat_semestre": semestre,
                "inscritos": len(grupo),
                "aprobados": int(grupo["evento"].sum()),
                "tasa_aprobacion": round(float(grupo["evento"].mean()), 3),
                "mediana_desde_inscripcion": mediana_km(km_ins),
                "mediana_desde_carrera": mediana_km(km_car),
            }
        )
        p = logrank(grupo, "sexo", "tiempo_inscripcion", "evento")
        pruebas.append({"mat_codigo": codigo, "mat_materia": nombre, "logrank_sexo_p": p})

    ranking = pd.DataFrame(filas)
    if ranking.empty:
        return {"ranking": ranking, "sexo": pd.DataFrame()}

    def _minmax(serie: pd.Series) -> pd.Series:
        if serie.max() == serie.min():
            return pd.Series(0.0, index=serie.index)
        return (serie - serie.min()) / (serie.max() - serie.min())

    # El peso del tiempo es 0,5. Cálculo II es la única mediana de 2 años y
    # la tasa de aprobación más baja, así que encabeza el índice con este peso.
    # Este archivo no recorre otros pesos.
    ranking["indice_criticidad"] = (
        100
        * (
            0.5 * _minmax(ranking["mediana_desde_inscripcion"].fillna(0))
            + 0.5 * _minmax(1 - ranking["tasa_aprobacion"])
        )
    ).round(1)
    ranking = ranking.sort_values("indice_criticidad", ascending=False)

    sexo = pd.DataFrame(pruebas)
    sexo["p_bh"] = benjamini_hochberg(sexo["logrank_sexo_p"])
    sexo["significativo_bh_0_05"] = sexo["p_bh"] < 0.05

    top = ranking.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["mat_materia"].astype(str).str.slice(0, 32), top["indice_criticidad"], color="#8c3a3a")
    ax.set_xlabel("Índice de criticidad (0 a 100)")
    ax.set_title("Asignaturas de mayor criticidad relativa")
    fig.tight_layout()
    fig.savefig(SALIDA / "criticidad_asignaturas.png", dpi=200)
    plt.close(fig)
    return {"ranking": ranking, "sexo": sexo}


# ---------------------------------------------------------------------------
# Ruta 3. Semestres
# ---------------------------------------------------------------------------

def analizar_semestres(
    seguimiento: pd.DataFrame, historial: pd.DataFrame, plan: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    print("\nSemestres: tiempo hasta completar el bloque")
    materias = (
        plan.dropna(subset=["mat_semestre"])
        .groupby("mat_semestre")["mat_codigo"]
        .apply(lambda s: set(s.astype(str)))
        .to_dict()
    )
    aprobadas = historial[historial["aprobado"] == 1]
    filas = []
    for _, alumno in seguimiento.iterrows():
        propias = historial[historial["id_alumno"] == alumno["id_alumno"]]
        ok = aprobadas[aprobadas["id_alumno"] == alumno["id_alumno"]]
        for semestre, codigos in sorted(materias.items()):
            empezadas = propias[propias["mat_codigo"].isin(codigos)]
            if empezadas.empty:
                continue
            inicio = empezadas["anho_primera_inscripcion"].min()
            hechas = ok[ok["mat_codigo"].isin(codigos)]
            completo = set(hechas["mat_codigo"]) >= codigos
            if completo and not hechas.empty:
                fin = hechas["anho_aprobacion"].max()
                evento = 1
            else:
                fin = alumno["anho_final"]
                evento = 0
            tiempo_sem = fin - inicio
            tiempo_car = fin - alumno["anio_ingreso"]
            if pd.isna(tiempo_sem) or tiempo_sem < 0 or pd.isna(tiempo_car) or tiempo_car < 0:
                continue
            filas.append(
                {
                    "sexo": alumno["sexo"],
                    "mat_semestre": int(semestre),
                    "tiempo_desde_inicio": tiempo_sem,
                    "tiempo_desde_carrera": tiempo_car,
                    "evento": evento,
                }
            )
    detalle = pd.DataFrame(filas)

    def resumir(columna: str, nombre: str) -> pd.DataFrame:
        salida = []
        for semestre, grupo in detalle.groupby("mat_semestre"):
            if len(grupo) < MINIMO_MATERIA:
                continue
            km = ajustar_km(grupo[columna], grupo["evento"], f"Semestre {semestre}")
            completaron = grupo.loc[grupo["evento"] == 1, columna]
            salida.append(
                {
                    "semestre": int(semestre),
                    "reloj": nombre,
                    "n": len(grupo),
                    "completaron": int(grupo["evento"].sum()),
                    "proporcion": round(float(grupo["evento"].mean()), 3),
                    "mediana_km": mediana_km(km),
                    "media_solo_completaron": round(float(completaron.mean()), 2)
                    if len(completaron)
                    else np.nan,
                }
            )
        return pd.DataFrame(salida).sort_values("semestre")

    desde_inicio = resumir("tiempo_desde_inicio", "desde el inicio del semestre")
    desde_carrera = resumir("tiempo_desde_carrera", "desde el ingreso a la carrera")
    sem1 = detalle[detalle["mat_semestre"] == 1]
    p_sem1 = logrank(sem1, "sexo", "tiempo_desde_inicio", "evento") if not sem1.empty else np.nan
    pruebas = pd.DataFrame(
        [
            {
                "prueba": "log-rank",
                "que_compara": "Tiempo para completar el semestre 1, por sexo",
                "p": p_sem1,
            }
        ]
    )

    if not desde_inicio.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(desde_inicio["semestre"].astype(str), desde_inicio["mediana_km"], color="#3d6b8c")
        ax.plot(
            desde_inicio["semestre"].astype(str),
            desde_inicio["media_solo_completaron"],
            "o-",
            color="#c47b2b",
            label="Media de quienes completaron",
        )
        ax.set_xlabel("Semestre curricular")
        ax.set_ylabel("Años")
        ax.set_title("Tiempo hasta completar el semestre")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(SALIDA / "semestres.png", dpi=200)
        plt.close(fig)
    return {"desde_inicio": desde_inicio, "desde_carrera": desde_carrera, "pruebas": pruebas}


# ---------------------------------------------------------------------------
# Informe
# ---------------------------------------------------------------------------

def main() -> None:
    print("Supervivencia académica desde las planillas Excel")
    ENTRADA.mkdir(parents=True, exist_ok=True)
    obligatorios = [
        "alumnos.xlsx",
        "egresados.xlsx",
        "inscripciones.xlsx",
        "calificaciones.xlsx",
        "plan_materias.xlsx",
    ]
    if any(not (ENTRADA / nombre).exists() for nombre in obligatorios):
        print("Faltan planillas en entrada/. Se escriben las plantillas y se detiene.")
        escribir_plantillas()
        print("Complete las hojas llamadas datos y vuelva a ejecutar el archivo.")
        return

    SALIDA.mkdir(parents=True, exist_ok=True)
    seguimiento, historial, plan = elaborar()
    if seguimiento.empty or historial.empty or plan.empty:
        raise SystemExit("La elaboración no produjo estudiantes, historial o plan.")

    analizar_trayectorias(seguimiento)
    
    analizar_asignaturas(seguimiento, historial)
    
    analizar_semestres(seguimiento, historial, plan)

    print("PROCESO CULMINADO!")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        if str(exc):
            print(exc)
        sys.exit(1 if str(exc) else 0)
