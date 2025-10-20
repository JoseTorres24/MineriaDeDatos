import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List
import numpy as np
import os

def create_normal_distribution(mean, sd, size) -> np.array:
    return np.random.normal(loc=mean, scale=sd, size=size)

def create_distribution(mean: float, size: int) -> pd.Series:
    return create_normal_distribution(mean, mean * 0.25, size)

def generate_df(means: List[Tuple[float, float, str]], n: int) -> pd.DataFrame:
    lists = [
        (create_distribution(_x, n), create_distribution(_y, n), np.repeat(_l, n))
        for _x, _y, _l in means
    ]
    x = np.array([])
    y = np.array([])
    labels = np.array([])
    for _x, _y, _l in lists:
        x = np.concatenate((x, _x), axis=None)
        y = np.concatenate((y, _y))
        labels = np.concatenate((labels, _l))
    return pd.DataFrame({"x": x, "y": y, "label": labels})

def get_cmap(n, name="hsv"):
    return plt.colormaps.get_cmap(name)

def scatter_group_by(file_path: str, df: pd.DataFrame, x_column: str, y_column: str, label_column: str):
    fig, ax = plt.subplots()
    labels = pd.unique(df[label_column])
    cmap = get_cmap(len(labels) + 1)
    for i, label in enumerate(labels):
        filter_df = df.query(f"{label_column} == '{label}'")
        ax.scatter(filter_df[x_column], filter_df[y_column], label=label)
    ax.legend()
    plt.set_cmap(cmap)
    plt.savefig(file_path)
    plt.close()

def euclidean_distance(p_1: np.array, p_2: np.array) -> float:
    return np.sqrt(np.sum((p_2 - p_1) ** 2))

def calculate_means(points: np.array, labels: np.array, clusters: int) -> np.array:
    mean = []
    for k in range(clusters):
        m = np.mean(points[labels == k], axis=0)
        mean.append(m)
    return mean

def calculate_nearest_k(point: np.array, actual_means: List[np.array]):
    distance = [euclidean_distance(mean, point) for mean in actual_means]
    nearest_k = np.argmin(distance)
    return (point, nearest_k)

def k_means(points: List[np.array], k: int, output_folder: str = "img"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    N = len(points)
    num_cluster = k
    max_iterations = 15

    x = np.array(points)
    y = np.random.randint(0, num_cluster, N)

    dimensions = len(points[0])
    mean = np.zeros((num_cluster, dimensions))

    for t in range(max_iterations):
        actual_mean = calculate_means(points=x, labels=y, clusters=num_cluster)
        y = np.array([calculate_nearest_k(point=point, actual_means=actual_mean)[1] for point in x])

        df_points = pd.DataFrame(x, columns=['x','y'])
        df_points['label'] = np.char.mod('%d', y)
        df_mean = pd.DataFrame(actual_mean, columns=['x','y'])
        df_mean['label'] = ['centroid' for _ in range(len(actual_mean))]
        df = pd.concat([df_points, df_mean])

        scatter_group_by(file_path=f"{output_folder}/kmeans_{t}.png", df=df, x_column="x", y_column="y", label_column='label')

        if np.array_equal(actual_mean, mean):
            break
        mean = actual_mean.copy()
    return y, mean

if __name__ == "__main__":
    df = pd.read_csv("practica1/INM_2025_limpia.csv")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist() 

# como tiene varias columnas creo que preguntar si tiene 2 columnas es incesario, pero bueno
# Tomamos las dos primeras columnas numéricas para el análisis
    x_col, y_col = num_cols[0], num_cols[1] #siempre tomara clave_ent y incidencia delectiva porque pues son las primeras dos columnas numericas, aunque si quiere en vez de 1 pongale 2 para que utilice en base al año.
    print(f"\nUsando columnas para clustering: {x_col} y {y_col}")

    points = df[[x_col, y_col]].dropna().values  # eliminamos filas vacía, aunque de esto ya me habia percatado en la practica 1


    clusters = 5
    labels, centroids = k_means(points, clusters, output_folder="practica7/img")


    df_result = df.copy()
    df_result["cluster"] = np.nan
    df_result.loc[df[[x_col, y_col]].dropna().index, "cluster"] = labels
    df_result.to_csv("practica7/resultado_kmeans.csv", index=False)

    print("\nAnálisis completado.")
    print(f"Se generaron las imágenes en la carpeta 'img/' y el archivo 'resultado_kmeans.csv' con los grupos.")
    print("Centroides finales:")
    print(centroids)

# conclusion
# a traves de los promedios de los k vecinos mas cercanos se puede observar que los grupos formados tienen diferencias significativas en los valores de las columnas seleccionadas, lo que indica que el algoritmo de k-means ha logrado identificar patrones distintos en los datos.
# Esto sugiere que los datos pueden estar segmentados en diferentes categorías o comportamientos basados en las características analizadas.
# pero eso se debe que independeientemente de los centroides, los puntos pueden estar muy dispersos dentro de cada grupo, lo que puede afectar la interpretación de los resultados.
# por eso se usaron 5 clusters, para tener una mejor visualizacion de los datos y sus patrones.
# la eleccion de las columnas tambien influye en los resultados, ya que diferentes combinaciones de variables pueden revelar distintos aspectos de los datos.
# por ejemplo esto fue por entidad y incidencia delictiva, pero si se usaran otras columnas numericas, los resultados podrian variar significativamente.
# tal cual por eso lo pudieramos hacer con el de dia, mes o año, para ver si hay patrones temporales en los datos delictivos.
# a final de cuenta, lo que se es que el hacer este tipo de codigo y adapartarlo a diferentes datasets, nos permite explorar y entender mejor la estructura subyacente de los datos, lo que es fundamental en el análisis de datos y la ciencia de datos en general.
# por eso creo que me tome mucha libertad al hacer mi practica anterior, respecto a esta, solo intente hacer pruebas en la carpetas de imagenes encontrara como se visualizan los datos.