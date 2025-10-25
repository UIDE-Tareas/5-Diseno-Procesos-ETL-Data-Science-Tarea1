import sys
import subprocess
import os
from pathlib import Path
from enum import Enum
import zipfile
from typing import Optional, Iterable
from dataclasses import dataclass
from typing import cast
from typing import Tuple
from types import SimpleNamespace

# ████████████████████████████████████████████████████████████ Funciones base ████████████████████████████████████████████████████████████

# Libs a instalar
LIBS = [
    "plotly",
    "dash",
    "dash-bootstrap-components",
    "ipython",
    "customtkinter",
    "requests",
    "numpy",
    "pandas",
    "seaborn",
    "matplotlib",
    "ipython",
    "scikit-learn",
    "requests",
    "wcwidth",
    "tabulate",
    "tabulate",
]


class ConsoleColor(Enum):
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    RESET = "\033[0m"


def PrintColor(message: str, color: ConsoleColor) -> str:
    RESET = ConsoleColor.RESET.value
    return f"{color.value}{message}{RESET}"


def ShowMessage(
    message: str, title: str, icon: str, color: ConsoleColor, end: str = "\n"
):
    colored_title = PrintColor(icon + f"  " + title.upper() + ":", color)
    print(f"{colored_title} {message}", end=end)


def ShowInfoMessage(
    message: str, title: str = "Info", icon: str = "ℹ️", end: str = "\n"
):
    ShowMessage(message, title, icon, ConsoleColor.CYAN, end)


def ShowSuccessMessage(
    message: str, title: str = "Success", icon: str = "✅", end: str = "\n"
):
    ShowMessage(message, title, icon, ConsoleColor.GREEN, end)


def ShowErrorMessage(
    message: str, title: str = "Error", icon: str = "❌", end: str = "\n"
):
    ShowMessage(message, title, icon, ConsoleColor.RED, end)


def ShowWarningMessage(
    message: str, title: str = "Warning", icon: str = "⚠️", end: str = "\n"
):
    ShowMessage(message, title, icon, ConsoleColor.YELLOW, end)


# Funcion para ejecutar comandos
def RunCommand(
    commandList: list[str], printCommand: bool = True, printError: bool = True
) -> subprocess.CompletedProcess[str]:
    print("⏳", " ".join(commandList))

    if printCommand:
        proc = subprocess.Popen(
            commandList,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        out_lines: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            out_lines.append(line)

        proc.wait()
        err_text = ""
        if proc.stderr is not None:
            err_text = proc.stderr.read() or ""

        if proc.returncode != 0 and printError and err_text:
            ShowErrorMessage(err_text, "", end="")
            # print(err_text, end="")

        return subprocess.CompletedProcess(
            args=commandList,
            returncode=proc.returncode,
            stdout="".join(out_lines),
            stderr=err_text,
        )

    else:
        result = subprocess.run(
            commandList, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode != 0 and printError and result.stderr:
            ShowErrorMessage(result.stderr, "", end="")
            # print(result.stderr, end="")
        return result


# Función para instalar las dependencias
def InstallDeps(libs: Optional[list[str]] = None):
    print("ℹ️ Installing deps.")
    printCommand = False
    printError = True
    RunCommand(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        printCommand=printCommand,
        printError=printError,
    )
    if libs is None or libs.count == 0:
        print("No hay elementos a instalar.")
    else:
        RunCommand(
            [sys.executable, "-m", "pip", "install", *libs],
            printCommand=printCommand,
            printError=printError,
        )
        print("Deps installed.")
    print()


# Función para mostrar info el ambiente de ejecución
def ShowEnvironmentInfo():
    print("ℹ️  Environment Info:")
    print("Python Version:", sys.version)
    print("Platform:", sys.platform)
    print("Executable Path:", sys.executable)
    print("Current Working Directory:", os.getcwd())
    print("VIRTUAL_ENV:", os.environ.get("VIRTUAL_ENV"))
    print("sys.prefix:", sys.prefix)
    print("sys.base_prefix:", sys.base_prefix)
    print()


InstallDeps(LIBS)
ShowEnvironmentInfo()

import requests


@dataclass(frozen=True)
class BoxStyle:
    TL: str
    TR: str
    BL: str
    BR: str
    H: str
    V: str


class TitleBoxLineStyle(Enum):
    SIMPLE = BoxStyle("┌", "┐", "└", "┘", "─", "│")
    DOUBLE = BoxStyle("╔", "╗", "╚", "╝", "═", "║")
    ROUNDED = BoxStyle("╭", "╮", "╰", "╯", "─", "│")
    HEAVY = BoxStyle("┏", "┓", "┗", "┛", "━", "┃")
    ASCII = BoxStyle("+", "+", "+", "+", "-", "|")
    DOUBLE_BOLD = BoxStyle("╔", "╗", "╚", "╝", "╬", "║")
    BLOCK = BoxStyle("█", "█", "█", "█", "█", "█")
    HEAVY_CROSS = BoxStyle("╒", "╕", "╘", "╛", "╪", "┃")
    METAL = BoxStyle("╞", "╡", "╘", "╛", "═", "║")


# Función para mostrar un título con recuadro
def ShowTitleBox(
    text: str,
    max_len: int = 100,
    boxLineStyle: TitleBoxLineStyle = TitleBoxLineStyle.SIMPLE,
    color: ConsoleColor = ConsoleColor.CYAN,
):
    try:

        def vislen(s: str) -> int:
            from wcwidth import wcswidth as _w

            n = _w(s)
            return n if n >= 0 else len(s)

    except Exception:

        def vislen(s: str) -> int:
            return len(s)

    pad = 1
    tlen = vislen(text)
    inner = max(max_len, tlen)
    left = (inner - tlen) // 2
    right = inner - tlen - left

    top = f"{boxLineStyle.value.TL}{boxLineStyle.value.H * (inner + 2 * pad)}{boxLineStyle.value.TR}"
    mid = f"{boxLineStyle.value.V}{' ' * pad}{' ' * left}{text}{' ' * right}{' ' * pad}{boxLineStyle.value.V}"
    bot = f"{boxLineStyle.value.BL}{boxLineStyle.value.H * (inner + 2 * pad)}{boxLineStyle.value.BR}"
    print(PrintColor("\n".join([top, mid, bot]), color))


# Función para descargar un archivo
def DownloadFile(uri: str, filename: str, overwrite: bool = False, timeout: int = 20):
    dest = Path(filename).resolve()
    if dest.exists() and dest.is_file() and dest.stat().st_size > 0 and not overwrite:
        print(
            f'✅ Ya existe: "{dest}". No se descarga (use overwrite=True para forzar).'
        )
        return
    if dest.parent and not dest.parent.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
    print(f'ℹ️ Descargando "{uri}" → "{dest}"')
    try:
        with requests.get(uri, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 64):
                    if chunk:  # filtra keep-alive chunks
                        f.write(chunk)
            tmp.replace(dest)
        print(f'✅ Archivo "{dest}" descargado exitosamente.')
    except requests.exceptions.RequestException as e:
        print(f"❌ Error al descargar: {e}")


# Función para descomprimir un archivo zip
def UnzipFile(filename: str, outputDir: str):
    print(f'ℹ️ Descomprimiendo "{filename}" en "{outputDir}"')
    try:
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(outputDir)
        print(f"Descomprimido en: {os.path.abspath(outputDir)}")
    except Exception as e:
        print(f"Error: {e}")


import pandas as pd
import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from IPython.display import display
from tabulate import tabulate

warnings.filterwarnings("ignore")

# Configurar opciones de Pandas
pd.set_option("display.float_format", "{:.2f}".format)
pandas.set_option("display.max_rows", None)
pandas.set_option("display.max_columns", None)


# Función para mostrar la información del DataFrame.
def ShowDfInfo(df: pandas.DataFrame, title):
    display(f"ℹ️ INFO {title} ℹ️")
    df.info()
    display()


# Función para mostrar las n primeras filas del DataFrame.
def ShowDfHead(df: pandas.DataFrame, title: str, headQty=10):
    display(f"ℹ️ {title}: Primeros {headQty} elementos.")
    print(
        tabulate(
            df.head(headQty).to_dict(orient="records"), headers="keys", tablefmt="psql"
        )
    )
    display()


# Función para mostrar las n últimas filas del DataFrame.
def ShowDfTail(df: pandas.DataFrame, title: str, tailQty=10):
    display(f"ℹ️ {title}: Últimos {tailQty} elementos.")
    print(
        tabulate(
            df.tail(tailQty).to_dict(orient="records"), headers="keys", tablefmt="psql"
        )
    )
    display()


# Mostrar el tamaño del DataFrame
def ShowDfShape(df: pandas.DataFrame, title: str):
    display(f"ℹ️ {title} - Tamaño de los datos")
    display(f"{df.shape[0]} filas x {df.shape[1]} columnas")
    display()


# Función para mostrar la estadística descriptiva de todas las columnas del DataFrame, por tipo de dato.
def ShowDfStats(df: pandas.DataFrame, title: str = ""):
    display(f"ℹ️ Estadística descriptiva - {title}")
    numeric_cols = df.select_dtypes(include="number")
    if not numeric_cols.empty:
        display("    🔢 Columnas numéricas".upper())
        numeric_desc = (
            numeric_cols.describe().round(2).T
        )  # Transpuesta para añadir columna
        numeric_desc["var"] = numeric_cols.var(numeric_only=True).round(2)
        display(numeric_desc.T)
    non_numeric_cols = df.select_dtypes(
        include=["boolean", "string", "category", "object"]
    )
    if not non_numeric_cols.empty:
        display("    🔡 Columnas no numéricas".upper())
        non_numeric_desc = non_numeric_cols.describe()
        display(non_numeric_desc)
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"])
    if not datetime_cols.empty:
        display("    📅 Columnas fechas".upper())
        datetime_desc = datetime_cols.describe()
        display(datetime_desc)


# Función para mostrar los valores nulos o NaN de cada columna en un DataFrame
def ShowDfNanValues(df: pandas.DataFrame, title: str):
    display(f"ℹ️ Contador de valores Nulos - {title}")
    nulls_count = df.isnull().sum()
    nulls_df = nulls_count.reset_index()
    nulls_df.columns = ["Columna", "Cantidad_Nulos"]
    display(nulls_df)
    display()


# Tipos de correlación
class CorrelationType(Enum):
    ALL = "all"
    STRONG = "strong"
    WEAK = "weak"


# Muestra las correlaciones completas, débiles y fuertes.
def ShowDfCorrelation(
    df: pandas.DataFrame,
    title: str,
    fig: Figure,
    ax: Axes,
    level: CorrelationType = CorrelationType.ALL,
    umbral: float = 0.6,  # |r| >= umbral => fuerte; |r| <= umbral => débil
    showTable: bool = False,
    annotate: bool = True,
):
    display(f"ℹ️ {(title).upper()} - Matriz de Correlación, Type: {level.name}")
    corr = df.select_dtypes(include=["number"]).corr().copy()
    if level == CorrelationType.STRONG:
        corr = corr.where(np.abs(corr) >= umbral)
    elif level == CorrelationType.WEAK:
        corr = corr.where(np.abs(corr) <= umbral)
        np.fill_diagonal(corr.values, 1)
    elif level != CorrelationType.ALL:
        raise ValueError(f"Invalid level: {level}")
    cax = ax.matshow(corr, vmin=-1, vmax=1)

    cols = corr.columns
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, ha="left")
    ax.set_yticklabels(cols)

    fig.colorbar(cax)

    if annotate:
        for (i, j), value in np.ndenumerate(corr.values):
            if not np.isnan(value):
                ax.text(j, i, f"{value:+.2f}", ha="center", va="center")

    if level == CorrelationType.ALL:
        titulo = "Matriz de correlación completa"
    else:
        titulo = f"Matriz de correlación ({level.name}, umbral={umbral})"

    total_elementos = corr.size
    total_nodiagonal = corr.size - corr.shape[0]
    total_nan = corr.isna().sum().sum()
    total_validos = total_elementos - total_nan - corr.shape[0]
    titulo = f"{titulo}, Total Matriz: {total_nodiagonal}, Total válidos: {total_validos}({((total_validos*100)/total_nodiagonal):.2f}%)"

    ax.set_title(titulo, pad=20)
    ax.grid(False)
    plt.tight_layout()
    plt.show()
    if showTable:
        display(corr)
    return corr


# ████████████████████████████████████████████████████████████ Inicio del script ████████████████████████████████████████████████████████████

DOWNLOAD_DIR = "Temp"

DATA_FILE_URI = "https://github.com/UIDE-Tareas/5-Diseno-Procesos-ETL-Data-Science-Tarea1/raw/refs/heads/main/Data/NotasMasterDataScience.csv"
DATA_FILENAME = f"{DOWNLOAD_DIR}/NotasMasterDataScience.csv"

ShowTitleBox(
    "DESCARGANDO BASE DE DATOS",
    boxLineStyle=TitleBoxLineStyle.BLOCK,
    color=ConsoleColor.CYAN,
)
DownloadFile(DATA_FILE_URI, DATA_FILENAME, False)

ShowTitleBox(
    "ANÁLISIS INICIAL DE DATOS",
    boxLineStyle=TitleBoxLineStyle.BLOCK,
    color=ConsoleColor.CYAN,
)

data = pd.read_csv(DATA_FILENAME)
ShowDfInfo(data, "Notas Master Data Science")
ShowDfStats(data, "Notas Master Data Science")
ShowDfHead(data, "Notas Master Data Science", 10)
ShowDfShape(data, "Notas Master Data Science")

materiasList = data.columns[1:].tolist()
data[materiasList] = data[materiasList].astype(pd.Float64Dtype())

ShowDfInfo(data, "Notas Master Data Science - Columnas convertidas a Float64")

ShowTitleBox(
    "CALCULO DE PROMEDIOS",
    boxLineStyle=TitleBoxLineStyle.BLOCK,
    color=ConsoleColor.CYAN,
)
data["Promedio"] = data.iloc[:, 1:].mean(axis=1).round(2)
ShowDfHead(data, "Notas Master Data Science", 10)

UMBRAL_APROBADO = 60.0
data["Estado"] = data["Promedio"].apply(
    lambda x: "Aprobado" if x >= UMBRAL_APROBADO else "Reprobado"
)
dataAprobados = data[data.Promedio >= UMBRAL_APROBADO]
dataReprobados = data[data.Promedio < UMBRAL_APROBADO]
ShowDfHead(dataAprobados, " Master Data Science - Estudiantes Aprobados ✅", 10)
ShowDfHead(dataReprobados, " Master Data Science - Estudiantes Reprobados ❌", 10)


promediosMateria = data.iloc[:, 1:-2].mean(axis=0).round(2)
dataPromediosMateria = pandas.DataFrame(
    {"Materia": promediosMateria.index, "Promedio": promediosMateria.values}
)
ShowDfHead(dataPromediosMateria, " Master Data Science - Promedios por materia", 10)

mejorEstudiante = data.loc[data["Promedio"].idxmax()]
peorEstudiante = data.loc[data["Promedio"].idxmin()]


data["Promedio"] = data["Promedio"].astype(pd.Float64Dtype())

# ████████████████████████████████████████████████████████████ Dashboard ████████████████████████████████████████████████████████████

from dash import Dash, html
from dash import Dash, dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import webbrowser

HOST = "localhost"
PORT = 7374
app = Dash(__name__, external_stylesheets=[dbc.themes.QUARTZ])
print(f"Iniciando Dashboard Host:{HOST}, Port:{PORT}...")
app.title = "Master Data Science - Análisis de Notas"

figPastel = px.pie(
    data,
    names="Estado",
    title="Distribución de Aprobados vs Reprobados",
    color="Estado",
    color_discrete_map={"Aprobado": "#00cc96", "Reprobado": "#ef553b"},
    hole=0.3,
)
figPastel.update_layout(template="plotly_dark")


figBarras = px.bar(
    dataPromediosMateria,
    x="Materia",
    y="Promedio",
    title="Promedio de Calificaciones por Materia",
    text="Promedio",
    color="Promedio",
    color_continuous_scale="Blues",
)
figBarras.update_traces(texttemplate="%{text:.2f}", textposition="outside")
figBarras.update_layout(
    template="plotly_dark", xaxis_title="Materia", yaxis_title="Promedio"
)


def KpiCard(title, value, subtitle):
    return dbc.Card(
        dbc.CardBody(
            [
                html.H6(title, className="text-muted"),
                html.H2(value, className="mb-1"),
                html.P(subtitle, className="text-muted mb-0"),
            ]
        ),
        className="h-100 text-center",
    )


kpiMejorEstudiante = KpiCard(
    "🏆 Mejor Estudiante",
    f"{mejorEstudiante['Promedio']:.2f}",
    mejorEstudiante["Nombre"],
)
kpiPeorEstudiante = KpiCard(
    "📉 Peor Estudiante", f"{peorEstudiante['Promedio']:.2f}", peorEstudiante["Nombre"]
)

app.layout = dbc.Container(
    [
        dbc.NavbarSimple(
            brand="Dashboard de Calificaciones — Análisis de Notas",
            color="primary",
            dark=True,
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(kpiMejorEstudiante, md=6),
                dbc.Col(kpiPeorEstudiante, md=6),
            ],
            className="g-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Distribución de Aprobados vs Reprobados"),
                                dcc.Graph(figure=figPastel),
                            ]
                        )
                    ),
                    md=6,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Promedio de Calificaciones por Materia"),
                                dcc.Graph(figure=figBarras),
                            ]
                        )
                    ),
                    md=6,
                ),
            ],
            className="g-4",
        ),
        html.Hr(),
        dbc.Card(
            dbc.CardBody(
                [
                    html.H4("📘 Análisis por materia", className="mb-3"),
                    dbc.Select(
    id="materiaDropdown",
    options=[{"label": m, "value": m} for m in materiasList],
    value=None,
    className="bg-light text-dark border-secondary",
    style={"width": "50%", "fontSize": "1rem", "borderRadius": "8px"},
),
                    html.Br(),
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.H6(
                                                "Promedio de la materia",
                                                className="text-muted",
                                            ),
                                            html.H2(id="materiaMean", className="mb-1"),
                                        ]
                                    )
                                ),
                                md=4,
                            ),
                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.H6(
                                                "🏆 Mejor estudiante",
                                                className="text-muted",
                                            ),
                                            html.H2(id="materiaBest", className="mb-1"),
                                        ]
                                    )
                                ),
                                md=4,
                            ),
                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.H6(
                                                "📉 Peor estudiante",
                                                className="text-muted",
                                            ),
                                            html.H2(
                                                id="materiaWorst", className="mb-1"
                                            ),
                                        ]
                                    )
                                ),
                                md=4,
                            ),
                        ],
                        className="g-4",
                    ),
                ]
            )
        ),
        html.Br(),
        dbc.Card(
            dbc.CardBody(
                [
                    html.H4(
                        "📊 Distribución de Aprobados vs Reprobados — Materia seleccionada",
                        className="mb-3",
                    ),
                    dcc.Graph(id="materiaPastel"),
                ]
            )
        ),
    ],
    fluid=True,
)


@app.callback(
    Output("materiaMean", "children"),
    Output("materiaBest", "children"),
    Output("materiaWorst", "children"),
    Output("materiaPastel", "figure"),
    Input("materiaDropdown", "value"),
)
def UpdateMateria(materia):
    if materia is None:
        # Gráfico vacío cuando no hay selección
        figEmpty = px.pie(
            names=["Seleccione una materia"],
            values=[1],
            title="Seleccione una materia para ver la distribución",
        )
        figEmpty.update_layout(template="plotly_dark")
        return "—", "—", "—", figEmpty

    mean = data[materia].mean().round(2)
    best = data.loc[data[materia].idxmax()]
    worst = data.loc[data[materia].idxmin()]

    bestName = f"{best['Nombre']} ({best[materia]:.2f})"
    worstName = f"{worst['Nombre']} ({worst[materia]:.2f})"

    aprobados = (data[materia] >= UMBRAL_APROBADO).sum()
    reprobados = (data[materia] < UMBRAL_APROBADO).sum()

    figMateriaPastel = px.pie(
        names=["Aprobado", "Reprobado"],
        values=[aprobados, reprobados],
        title=f"Distribución de Aprobados vs Reprobados en {materia}",
        color=["Aprobado", "Reprobado"],
        color_discrete_map={"Aprobado": "#00cc96", "Reprobado": "#ef553b"},
        hole=0.3,
    )
    figMateriaPastel.update_traces(textinfo="label+percent")
    figMateriaPastel.update_layout(template="plotly_dark")

    return f"{mean:.2f}", bestName, worstName, figMateriaPastel


if __name__ == "__main__":
    webbrowser.open(f"http://{HOST}:{PORT}")
    app.run(debug=False, port=PORT, host=HOST, use_reloader=True)

# TODO
# Agregar la tabla de todos los estudiantes con sus notas, promedios y estado (aprobado/reprobado).
# Agregar gráfico de barras de todas las materias con total aprobados y reprobados por materia.
# Generar pdf
