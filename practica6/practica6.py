import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder

def data_classification(df: pd.DataFrame, features: list, label: str, test_size: float = 0.2, n_neighbors: int = 5):
    df = df.copy()
    le = LabelEncoder()
    for col in features + [label]:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])

    X = df[features]
    y = df[label]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Crear modelo KNN (clasificación)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    print(" Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    print("\n Reporte de clasificación:")
    print(classification_report(y_test, y_pred))

    print("\n==============================")
    print("Vecinos más cercanos por entidad federativa")
    print("==============================\n")

    group_col = 'entidad_federativa'

    # Eliminamos 'entidad_federativa' de las features antes de agrupar
    features_no_group = [col for col in features if col != group_col]

    # Agrupar por entidad y calcular promedios de las demás características
    df_entity = df.groupby(group_col)[features_no_group].mean().reset_index()

    # Escalar los promedios
    X_entity_scaled = scaler.fit_transform(df_entity[features_no_group])

    # Modelo de vecinos más cercanos
    nn = NearestNeighbors(n_neighbors=4, metric='euclidean')  # 3 vecinos + el mismo osea el [0 y luego 1 , 2 ,3 ] qie son enteramente igual a el
    nn.fit(X_entity_scaled)

    distances, indices = nn.kneighbors(X_entity_scaled)

    resultados = []

    for i, entidad in enumerate(df_entity[group_col]):
        vecinos = df_entity.loc[indices[i][1:], group_col].values  
        distancias = distances[i][1:]
        print(f"🔹 Entidad: {entidad}")
        print(f"   Vecinos más cercanos: {', '.join(map(str, vecinos))}")
        print(f"   Distancias: {distancias}\n")

        resultados.append({
            "entidad": entidad,
            "vecinos": ", ".join(map(str, vecinos)),
            "distancias": ", ".join([f"{d:.4f}" for d in distancias])
        })

    vecinos_df = pd.DataFrame(resultados)
    vecinos_df.to_csv("vecinos_por_entidad.csv", index=False, encoding="utf-8-sig")
    print("Archivo 'vecinos_por_entidad.csv' guardado correctamente.")

df = pd.read_csv("practica1/INM_2025_limpia.csv", encoding="utf-8-sig")

features = ['bien_juridico_afectado', 'tipo_delito', 'subtipo_delito', 
            'modalidad', 'incidencia_delictiva', 'entidad_federativa', 
            'anio', 'mes_num', 'dia']
label = 'clave_ent'

data_classification(df, features, label)

# lo haremos mas comlejo todavia en esta practica, porque mezclare el dataset que tengo en la carpeta csv_apoyo con este dataset para   
#temas de generos, edades y modalidad(armas, etc)

