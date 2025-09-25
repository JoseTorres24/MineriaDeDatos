import pandas as pd
import os

os.makedirs("csv_apoyo", exist_ok=True)

# Leer con latin-1
df = pd.read_csv("IDEFC_NM_ago25.csv", encoding="latin-1")

# Estandarizar nombres (manteniendo tildes y ñ)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Crear clave única para delitos
df['delito_key'] = (
    df['bien_jurídico_afectado'] + '|' +
    df['tipo_de_delito'] + '|' +
    df['subtipo_de_delito'] + '|' +
    df['modalidad']
)

# Tabla de entidades
entidades = df[['clave_ent', 'entidad']].drop_duplicates().reset_index(drop=True)
entidades['entidad_id'] = entidades.index + 1

# Tabla de delitos
delitos = df[['delito_key']].drop_duplicates().reset_index(drop=True)
delitos['delito_id'] = delitos.index + 1
delitos[['bien_jurídico', 'tipo_delito', 'subtipo_delito', 'modalidad']] = delitos['delito_key'].str.split('|', expand=True)

# Tabla de hechos
hechos = (
    df.merge(entidades, on=['clave_ent', 'entidad'])
      .merge(delitos[['delito_id', 'delito_key']], on='delito_key')
)

hechos = hechos[['año', 'entidad_id', 'delito_id',
                 'enero','febrero','marzo','abril','mayo','junio',
                 'julio','agosto','septiembre','octubre','noviembre','diciembre']]

# Guardar
entidades.to_csv("entidades.csv", index=False, encoding="utf-8-sig")
delitos.to_csv("delitos.csv", index=False, encoding="utf-8-sig")
hechos.to_csv("hechos.csv", index=False, encoding="utf-8-sig")

print("listoooo, Tablas normalizadas creadas en csv_apoyo")
