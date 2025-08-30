import pandas as pd
#limpieza de datos de INM_estatal_jun25.csv
df = pd.read_csv('practica1/INM_estatal_jun25.csv')
# solo reviso cuales de las columnas "entidad" y "entidad_federativa" son diferentes y como son
# pero demuestro aqui que realmente son iguales solo que utilizan los nombres completos en entidad y las abreviaciones en entidad_federativa,
#me carcomia porque los datos terminan dando estos
"""
                            entidad          entidad_federativa
4704               Coahuila de Zaragoza           Coahuila
4705               Coahuila de Zaragoza           Coahuila
4706               Coahuila de Zaragoza           Coahuila
4707               Coahuila de Zaragoza           Coahuila
4708               Coahuila de Zaragoza           Coahuila
...                                 ...                ...
393955  Veracruz de Ignacio de la Llave           Veracruz
393956  Veracruz de Ignacio de la Llave           Veracruz
393957  Veracruz de Ignacio de la Llave           Veracruz
393958  Veracruz de Ignacio de la Llave           Veracruz
393959  Veracruz de Ignacio de la Llave           Veracruz

"""
print(df[df["entidad"] != df["entidad_federativa"]][["entidad", "entidad_federativa"]])

df = df.drop(columns=["entidad"])
df = df.drop(columns=["mes"])
df = df.drop(columns=["anio"])

# Asegurar que fecha sea datetime
df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

# Derivar componentes de la fecha
df["anio"] = df["fecha"].dt.year
df["mes_num"] = df["fecha"].dt.month
df["dia"] = df["fecha"].dt.day

df = df.drop(columns=["fecha"])


cols = ["bien_juridico_afectado", "tipo_delito", "subtipo_delito", "modalidad"]
#esto como es puro text pues ya lo pongo en minusculas y sin espacios al inicio o final
for c in cols:
    df[c] = df[c].str.lower().str.strip()

# esto es para ver si hay valores negativos en la columna incidencia_delictiva
negativos = df[df["incidencia_delictiva"] < 0]
print(f"Filas con valores negativos: {len(negativos)}")

#elimino los  que sean 0 y que no tengan que aportar nada a la incidencia delictiva
df = df[df["incidencia_delictiva"] > 0]
# el tema es que habia puros valores int, entonces pues por eso lo cambie de float a int
# que feo se ve que ssean float los datos
df["incidencia_delictiva"] = df["incidencia_delictiva"].astype(int)


df.to_csv('practica1/INM_2025.csv', index=False)

#en resumen converti la fecha en 3 columnas debido a que pues para mi es necesario pues si quiero buscar
#por mes si quiero saber de alguna entidad el mes que tuvo mas incidencia delictiva por eso lo hice para que sea mas facil

"""deje la columnna de clave_entidad porque pues para metodos de busqueda o por nombre de entidad
puede que haya errores de tipeo y asi con la clave_entidad no habria ese mismo problema"""

