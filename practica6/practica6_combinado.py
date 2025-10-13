import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

#Sinceramente NO SE PORQUE NO SE GENERA ya cheque el proceso con muchos prints
# pero bueno creo lo intente mas o menos
def cargar_y_combinar_datasets():
    
    df_principal = pd.read_csv("practica1/INM_2025_limpia.csv", encoding="utf-8-sig")

    df_combinado = df_principal.copy()
    
    carpeta_apoyo = "csv_apoyo"
    
    if os.path.exists(carpeta_apoyo):
        print("Cargando datasets de apoyo...")
        for archivo in os.listdir(carpeta_apoyo):
            if archivo == 'delitos.csv':
                ruta_archivo = os.path.join(carpeta_apoyo, archivo)
                try:
                    df_delitos = pd.read_csv(ruta_archivo, encoding="utf-8-sig")
                    print(f"{archivo}: {df_delitos.shape}")
                    print(f"Columnas: {df_delitos.columns.tolist()}")
                    
                    print("Primeras filas del dataset de delitos:")
                    print(df_delitos.head(3))
                    df_combinado = integrar_dataset_delitos(df_principal, df_delitos)
                    
                except Exception as e:
                    print(f"Error cargando {archivo}: {e}")
    
    return df_combinado

def integrar_dataset_delitos(df_principal, df_delitos):
   
    df_principal['clave_delito'] = (
        df_principal['bien_juridico_afectado'].str.strip() + "|" +
        df_principal['tipo_delito'].str.strip() + "|" +
        df_principal['subtipo_delito'].str.strip() + "|" +
        df_principal['modalidad'].str.strip()
    )
    
    mapeo_columnas = {
        'bien_jurídico': 'bien_juridico_afectado',
        'delito_key': 'clave_delito_original'
    }
    
    df_delitos = df_delitos.rename(columns=mapeo_columnas)
    
    df_delitos['clave_delito'] = (
        df_delitos['bien_juridico_afectado'].str.strip() + "|" +
        df_delitos['tipo_delito'].str.strip() + "|" +
        df_delitos['subtipo_delito'].str.strip() + "|" +
        df_delitos['modalidad'].str.strip()
    )
    
    print(f"Claves únicas en principal: {df_principal['clave_delito'].nunique()}")
    print(f"Claves únicas en delitos: {df_delitos['clave_delito'].nunique()}")
    
    claves_coincidentes = set(df_principal['clave_delito']).intersection(set(df_delitos['clave_delito']))
    print(f"Coincidencias encontradas: {len(claves_coincidentes)}")
    
    df_combinado = pd.merge(
        df_principal,
        df_delitos[['clave_delito', 'delito_id', 'delito_key']],
        on='clave_delito',
        how='left',
        suffixes=('', '_catalogo')
    )
    
    print(f"Dataset después del merge: {df_combinado.shape}")
    print(f"Delitos con ID asignado: {df_combinado['delito_id'].notna().sum()}")
    
    return df_combinado

def enriquecer_features(df):
    
    df = df.copy()
    
    df['trimestre'] = (df['mes_num'] - 1) // 3 + 1
    df['semana_año'] = df['mes_num'] * 4
    df['fin_de_semana'] = ((df['dia'] % 7) >= 5).astype(int)
    
    if 'delito_id' in df.columns:
        df['grupo_gravedad'] = pd.cut(
            df['delito_id'].fillna(0), 
            bins=[-1, 10, 50, 100, float('inf')],
            labels=['Baja', 'Media', 'Alta', 'Muy Alta']
        )
        
        freq_por_delito = df.groupby('delito_id')['incidencia_delictiva'].transform('mean')
        df['frecuencia_relativa_delito'] = freq_por_delito
        
        print(f"Delitos categorizados por gravedad: {df['grupo_gravedad'].value_counts().to_dict()}")
    
    freq_por_tipo = df.groupby('tipo_delito')['incidencia_delictiva'].transform('mean')
    df['frecuencia_relativa_tipo'] = freq_por_tipo
    
    freq_por_entidad = df.groupby('entidad_federativa')['incidencia_delictiva'].transform('mean')
    df['frecuencia_relativa_entidad'] = freq_por_entidad
    
    df['interaccion_tipo_entidad'] = df['frecuencia_relativa_tipo'] * df['frecuencia_relativa_entidad']
    
    df['estacion'] = pd.cut(
        df['mes_num'],
        bins=[0, 3, 6, 9, 12],
        labels=['Invierno', 'Primavera', 'Verano', 'Otoño'],
        right=True
    )
    
    return df

def data_classification_avanzada(df: pd.DataFrame, features: list, label: str, test_size: float = 0.2, n_neighbors: int = 5):
    """Versión avanzada del análisis de clasificación"""
    
    df = df.copy()
    
    print("ANÁLISIS AVANZADO DE DELITOS CON CATÁLOGO")
    print(f"Dataset combinado: {df.shape}")
    
    df = enriquecer_features(df)
    print(f"Dataset enriquecido: {df.shape}")
    
    le = LabelEncoder()
    codificaciones = {}
    
    if df[label].dtype == 'object':
        df[f'{label}_cod'] = le.fit_transform(df[label])
        label_final = f'{label}_cod'
        codificaciones[label] = le
    else:
        label_final = label
    
    features_numericas = []
    features_categoricas = []
    
    for col in features:
        if df[col].dtype == 'object':
            features_categoricas.append(col)
            le_col = LabelEncoder()
            df[f'{col}_cod'] = le_col.fit_transform(df[col])
            codificaciones[col] = le_col
            features_numericas.append(f'{col}_cod')
        else:
            features_numericas.append(col)
    
    features_extendidas = features_numericas + [
        'trimestre', 'semana_año', 'fin_de_semana',
        'frecuencia_relativa_tipo', 'frecuencia_relativa_entidad',
        'interaccion_tipo_entidad'
    ]
    
    if 'delito_id' in df.columns:
        features_extendidas.extend(['delito_id', 'frecuencia_relativa_delito'])
    
    features_extendidas = [f for f in features_extendidas if f in df.columns]
    
    print(f"Features utilizadas: {len(features_extendidas)}")
    
    X = df[features_extendidas]
    y = df[label_final]
    # en su mayoria de este parte lo tome en base como entrenar las variables a un curso de kaggle, de una manera BASICA aplico el conocimiento
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nBuscando mejor valor de k...")
    best_score = 0
    best_k = 3
    
    for k in [3, 5, 7, 9, 11]:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        score = knn.score(X_test_scaled, y_test)
        print(f"   k={k}, precisión: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k
    
    print(f"Mejor k: {best_k} (precisión: {best_score:.4f})")
    
    knn_final = KNeighborsClassifier(n_neighbors=best_k)
    knn_final.fit(X_train_scaled, y_train)
    y_pred = knn_final.predict(X_test_scaled)
    
    print("\n" + "="*50)
    print("EVALUACIÓN DEL MODELO")
    print("="*50)
    
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))
    
    analizar_vecinos_entidades(df, features_extendidas, scaler, codificaciones)
    
    analizar_delitos_catalogo(df)
    
    return {
        'modelo': knn_final,
        'scaler': scaler,
        'features': features_extendidas,
        'codificaciones': codificaciones,
        'dataset': df
    }

def analizar_vecinos_entidades(df, features, scaler, codificaciones):

    
    print("\n" + "="*50)
    print("VECINOS MÁS CERCANOS POR ENTIDAD FEDERATIVA")
    print("="*50)
    
    # Crear copia del DataFrame
    df_temp = df.copy()
    
    # Aseguramos que todas las columnas codificadas existan
    for col in features:
        if col.endswith('_cod') and col not in df_temp.columns:
            col_original = col.replace('_cod', '')
            if col_original in df_temp.columns and col_original in codificaciones:
                df_temp[col] = codificaciones[col_original].transform(df_temp[col_original])
    
    # Solo las features que existen en df_temp
    features_agrupacion = [f for f in features if f in df_temp.columns]
    
    print(f"   Features disponibles para agrupación: {len(features_agrupacion)}")
    
    # Determinar columna de agrupación (entidad)
    group_col = 'entidad_federativa'
    group_col_cod = f'{group_col}_cod'
    
    if group_col_cod in df_temp.columns:
        group_col_used = group_col_cod
        if group_col in codificaciones:
            mapeo_entidades = df_temp[[group_col, group_col_cod]].drop_duplicates()
            mapeo_entidades = mapeo_entidades.set_index(group_col_cod)[group_col].to_dict()
    else:
        group_col_used = group_col
        mapeo_entidades = None
    
    if group_col_used in features_agrupacion:
        features_agrupacion = [f for f in features_agrupacion if f != group_col_used]

    try:
        # Agrupar entidades y calcular medias
        df_entity = df_temp.groupby(group_col_used)[features_agrupacion].mean().reset_index()
    except Exception as e:
        print(f"Error al agrupar entidades: {e}")
        print(f"Columna usada para agrupar: {group_col_used}")
        print(f"Features de agrupación: {features_agrupacion}")
        return
    
    print(f"Entidades a analizar: {len(df_entity)}")
    
    if len(df_entity) < 2:
        print("No hay suficientes entidades para el análisis de vecinos.")
        return
    
    X_entity = df_entity[features_agrupacion]
    
    # Verificar consistencia
    if not all(feature in X_entity.columns for feature in features_agrupacion):
        missing = set(features_agrupacion) - set(X_entity.columns)
        print(f"Features faltantes: {missing}")
        features_agrupacion = [f for f in features_agrupacion if f in X_entity.columns]
        X_entity = df_entity[features_agrupacion]
    
    try:
        # Escalar las características
        X_entity_scaled = scaler.transform(X_entity)
        
        # Calcular vecinos
        nn_neighbors = min(4, len(df_entity))
        nn = NearestNeighbors(n_neighbors=nn_neighbors, metric='euclidean')
        nn.fit(X_entity_scaled)
        
        distances, indices = nn.kneighbors(X_entity_scaled)
        
        resultados = []
        
        # Mostrar resultados por entidad
        for i in range(len(df_entity)):
            if mapeo_entidades:
                entidad_cod = df_entity.iloc[i][group_col_used]
                entidad_nombre = mapeo_entidades.get(entidad_cod, f"Entidad_{entidad_cod}")
            else:
                entidad_nombre = df_entity.iloc[i][group_col_used]
            
            vecinos_indices = indices[i][1:]
            distancias = distances[i][1:]
            
            vecinos_nombres = []
            for idx in vecinos_indices:
                if mapeo_entidades:
                    vecino_cod = df_entity.iloc[idx][group_col_used]
                    vecino_nombre = mapeo_entidades.get(vecino_cod, f"Entidad_{vecino_cod}")
                else:
                    vecino_nombre = df_entity.iloc[idx][group_col_used]
                vecinos_nombres.append(vecino_nombre)
            
            print(f"\n🔹 {entidad_nombre}")
            print(f"   Vecinos similares: {', '.join(map(str, vecinos_nombres))}")
            print(f"   Distancias: {[f'{d:.3f}' for d in distancias]}")
            
            resultados.append({
                "entidad": entidad_nombre,
                "vecino_1": vecinos_nombres[0] if len(vecinos_nombres) > 0 else "",
                "vecino_2": vecinos_nombres[1] if len(vecinos_nombres) > 1 else "",
                "vecino_3": vecinos_nombres[2] if len(vecinos_nombres) > 2 else "",
                "distancia_1": distancias[0] if len(distancias) > 0 else 0,
                "distancia_2": distancias[1] if len(distancias) > 1 else 0,
                "distancia_3": distancias[2] if len(distancias) > 2 else 0,
            })
        
        # Guardar resultados
        vecinos_df = pd.DataFrame(resultados)
        vecinos_df.to_csv("analisis_avanzado_entidades.csv", index=False, encoding="utf-8-sig")
        print("\nArchivo 'analisis_avanzado_entidades.csv' guardado correctamente.")
        
    except Exception as e:
        print(f"Error en análisis de vecinos: {e}")
        print(f"Features esperadas: {features_agrupacion}")
        print(f"Features disponibles: {X_entity.columns.tolist()}")
        print(f"¿Coinciden?: {set(features_agrupacion) == set(X_entity.columns)}")


def analizar_delitos_catalogo(df):
    
    print("\n" + "="*50)
    print("ANÁLISIS CON CATÁLOGO DE DELITOS")
    print("="*50)
    
    if 'delito_id' in df.columns:
        delitos_comunes = df.groupby(['tipo_delito', 'delito_id']).size().reset_index(name='count')
        delitos_comunes = delitos_comunes.sort_values('count', ascending=False).head(10)
        
        print("\nTop 10 delitos más comunes (con ID):")
        for _, row in delitos_comunes.iterrows():
            print(f"   • {row['tipo_delito']} (ID: {row['delito_id']}): {row['count']} registros")
        
        if 'grupo_gravedad' in df.columns:
            print(f"\nDistribución por gravedad:")
            gravedad_counts = df['grupo_gravedad'].value_counts()
            for gravedad, count in gravedad_counts.items():
                print(f"   • {gravedad}: {count} registros")
    
    if 'trimestre' in df.columns:
        print(f"\nDistribución por trimestres:")
        trimestre_counts = df['trimestre'].value_counts().sort_index()
        for trimestre, count in trimestre_counts.items():
            print(f"   • Trimestre {trimestre}: {count} registros")

if __name__ == "__main__":
    df_combinado = cargar_y_combinar_datasets()
    features_base = [
        'bien_juridico_afectado', 'tipo_delito', 'subtipo_delito', 
        'modalidad', 'incidencia_delictiva', 'entidad_federativa', 
        'anio', 'mes_num', 'dia'
    ]
    
    label = 'clave_ent'

    resultados = data_classification_avanzada(df_combinado, features_base, label)
    
    print("\nANÁLISIS COMPLETADO EXITOSAMENTE!")
    print("Archivos generados:")
    print("- analisis_avanzado_entidades.csv")
    print("- Dataset enriquecido con catálogo de delitos")