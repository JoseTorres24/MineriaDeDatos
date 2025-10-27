import matplotlib.pyplot as plt
import statsmodels.api as sm
import numbers
import pandas as pd
from tabulate import tabulate
from typing import Tuple, Dict
import numpy as np
import os
from io import StringIO

def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt="orgtbl"))

def transform_variable(df: pd.DataFrame, x:str) -> pd.Series:
    if isinstance(df[x].iloc[0], numbers.Number):
        return df[x]
    else:
        return pd.Series([i for i in range(0, len(df[x]))])

def linear_regression(df: pd.DataFrame, x:str, y: str) -> Dict[str, float]:
    fixed_x = transform_variable(df, x)
    model = sm.OLS(df[y], sm.add_constant(fixed_x)).fit()
    
    # usar stringio para evitar warnings de deprecación
    table1_html = model.summary().tables[1].as_html()
    bands = pd.read_html(StringIO(table1_html), header=0, index_col=0)[0]
    
    table0_html = model.summary().tables[0].as_html()
    r_2_t = pd.read_html(StringIO(table0_html), header=None, index_col=None)[0]
    
    # corregir índices de columnas
    coef_values = bands['coef'].values
    # las columnas pueden variar, usar nombres más robustos
    low_band_value = bands.iloc[0]['[0.025'] if '[0.025' in bands.columns else bands.iloc[0, 4]
    hi_band_value = bands.iloc[0]['0.975]'] if '0.975]' in bands.columns else bands.iloc[0, 5]
    
    return {
        'm': coef_values[1], 
        'b': coef_values[0], 
        'r2': r_2_t.iloc[0, 3], 
        'r2_adj': r_2_t.iloc[1, 3], 
        'low_band': low_band_value, 
        'hi_band': hi_band_value
    }

def plt_lr(df: pd.DataFrame, x:str, y: str, m: float, b: float, r2: float, r2_adj: float, 
           low_band: float, hi_band: float, colors: Tuple[str,str], title: str = ""):
    fixed_x = transform_variable(df, x)
    df.plot(x=x, y=y, kind='scatter', figsize=(12, 8))
    plt.plot(df[x], [m * x + b for _, x in fixed_x.items()], color=colors[0], linewidth=2)
    plt.fill_between(df[x],
                     [m * x + low_band for _, x in fixed_x.items()],
                     [m * x + hi_band for _, x in fixed_x.items()], 
                     alpha=0.2, color=colors[1])
    
    plt.title(f'regresión lineal: {title}\nr² = {float(r2):.3f} | r² ajustado = {float(r2_adj):.3f}')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True, alpha=0.3)

# crear directorio si no existe
os.makedirs('practica8', exist_ok=True)

# cargar el dataset de delitos
df = pd.read_csv("practica1/INM_2025_limpia.csv")

print("\n" + "="*50)
print("analisis 1: incidencia delictiva por año")
print("="*50)

df_by_year = df.groupby("anio")["incidencia_delictiva"].sum().reset_index()
df_by_year.columns = ["anio", "total_incidencias"]
print_tabulate(df_by_year)

if len(df_by_year) > 1:
    a1 = linear_regression(df_by_year, "anio", "total_incidencias")
    plt_lr(df=df_by_year, x="anio", y="total_incidencias", 
           colors=('blue', 'lightblue'), title="incidencia delictiva por año", **a1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('practica8/lr_incidencia_anio.png', dpi=300, bbox_inches='tight')
    plt.close()

print("\n" + "="*50)
print("analisis 2: incidencia delictiva por mes")
print("="*50)

df_by_month = df.groupby("mes_num")["incidencia_delictiva"].sum().reset_index()
df_by_month.columns = ["mes_num", "total_incidencias"]
print_tabulate(df_by_month)

if len(df_by_month) > 1:
    a2 = linear_regression(df_by_month, "mes_num", "total_incidencias")
    plt_lr(df=df_by_month, x="mes_num", y="total_incidencias", 
           colors=('red', 'pink'), title="incidencia delictiva por mes", **a2)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('practica8/lr_incidencia_mes.png', dpi=300, bbox_inches='tight')
    plt.close()

print("\n" + "="*50)
print("analisis 3: incidencia por tipo de delito (top 10)")
print("="*50)

df_by_crime = df.groupby("tipo_delito")["incidencia_delictiva"].sum().reset_index()
df_by_crime.columns = ["tipo_delito", "total_incidencias"]
df_by_crime = df_by_crime.sort_values("total_incidencias", ascending=False)
print_tabulate(df_by_crime.head(10))

print("\n" + "="*50)
print("analisis 4: incidencia por entidad federativa (top 10)")
print("="*50)

df_by_state = df.groupby("entidad_federativa")["incidencia_delictiva"].sum().reset_index()
df_by_state.columns = ["entidad_federativa", "total_incidencias"]
df_by_state = df_by_state.sort_values("total_incidencias", ascending=False)
print_tabulate(df_by_state.head(10))

print("\n" + "="*50)
print("analisis 5: tendencia de fraude por mes")
print("="*50)

df_fraude = df[df["tipo_delito"] == "fraude"]
df_fraude_by_month = df_fraude.groupby("mes_num")["incidencia_delictiva"].sum().reset_index()
df_fraude_by_month.columns = ["mes_num", "total_incidencias"]
print_tabulate(df_fraude_by_month)

if len(df_fraude_by_month) > 1:
    a5 = linear_regression(df_fraude_by_month, "mes_num", "total_incidencias")
    plt_lr(df=df_fraude_by_month, x="mes_num", y="total_incidencias", 
           colors=('green', 'lightgreen'), title="tendencia de fraudes por mes", **a5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('practica8/lr_fraude_mes.png', dpi=300, bbox_inches='tight')
    plt.close()

print("\n" + "="*50)
print("analisis 6: incidencia por bien juridico afectado")
print("="*50)

df_by_legal = df.groupby("bien_juridico_afectado")["incidencia_delictiva"].sum().reset_index()
df_by_legal.columns = ["bien_juridico_afectado", "total_incidencias"]
df_by_legal = df_by_legal.sort_values("total_incidencias", ascending=False)
print_tabulate(df_by_legal)

print("\n" + "="*50)
print("analisis 7: tendencia de robo por año")
print("="*50)

df_robo = df[df["tipo_delito"] == "robo"]
df_robo_by_year = df_robo.groupby("anio")["incidencia_delictiva"].sum().reset_index()
df_robo_by_year.columns = ["anio", "total_incidencias"]
print_tabulate(df_robo_by_year)

if len(df_robo_by_year) > 1:
    a7 = linear_regression(df_robo_by_year, "anio", "total_incidencias")
    plt_lr(df=df_robo_by_year, x="anio", y="total_incidencias", 
           colors=('purple', 'lavender'), title="tendencia de robos por año", **a7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('practica8/lr_robo_anio.png', dpi=300, bbox_inches='tight')
    plt.close()

print("\n" + "="*50)
print("analisis 8: tendencia de violencia familiar por mes")
print("="*50)

df_violencia = df[df["tipo_delito"] == "violencia familiar"]
df_violencia_by_month = df_violencia.groupby("mes_num")["incidencia_delictiva"].sum().reset_index()
df_violencia_by_month.columns = ["mes_num", "total_incidencias"]
print_tabulate(df_violencia_by_month)

if len(df_violencia_by_month) > 1:
    a8 = linear_regression(df_violencia_by_month, "mes_num", "total_incidencias")
    plt_lr(df=df_violencia_by_month, x="mes_num", y="total_incidencias", 
           colors=('orange', 'yellow'), title="tendencia de violencia familiar por mes", **a8)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('practica8/lr_violencia_mes.png', dpi=300, bbox_inches='tight')
    plt.close()

print("\n" + "="*60)
print("resumen ejecutivo del analisis")
print("="*60)
print(f"total de registros analizados: {len(df):,}")
print(f"periodo cubierto: {df['anio'].min()} - {df['anio'].max()}")
print(f"tipos de delito únicos: {df['tipo_delito'].nunique()}")
print(f"entidades federativas: {df['entidad_federativa'].nunique()}")

print("\nresultados de regresiones lineales:")
print("-" * 40)
if len(df_by_year) > 1:
    print(f"• incidencia por año: r² = {float(a1['r2']):.3f} (tendencia: {'positiva' if a1['m'] > 0 else 'negativa'})")
if len(df_by_month) > 1:
    print(f"• incidencia por mes: r² = {float(a2['r2']):.3f} (tendencia: {'positiva' if a2['m'] > 0 else 'negativa'})")
if len(df_fraude_by_month) > 1:
    print(f"• fraudes por mes: r² = {float(a5['r2']):.3f} (tendencia: {'positiva' if a5['m'] > 0 else 'negativa'})")
if len(df_robo_by_year) > 1:
    print(f"• robos por año: r² = {float(a7['r2']):.3f} (tendencia: {'positiva' if a7['m'] > 0 else 'negativa'})")
if len(df_violencia_by_month) > 1:
    print(f"• violencia familiar por mes: r² = {float(a8['r2']):.3f} (tendencia: {'positiva' if a8['m'] > 0 else 'negativa'})")

print("\nhallazgos principales:")
print("-" * 25)
print(f"• delito mas común: {df_by_crime.iloc[0]['tipo_delito']} ({df_by_crime.iloc[0]['total_incidencias']:,} casos)")
print(f"• entidad con mas incidencia: {df_by_state.iloc[0]['entidad_federativa']} ({df_by_state.iloc[0]['total_incidencias']:,} casos)")
print(f"• bien jurídico mas afectado: {df_by_legal.iloc[0]['bien_juridico_afectado']}")

print("\ninterpretación de r²:")
print("-" * 25)
print("• r² = 1: modelo perfecto")
print("• r² > 0.7: buen ajuste")
print("• r² > 0.5: ajuste moderado") 
print("• r² < 0.3: bajo poder predictivo")
