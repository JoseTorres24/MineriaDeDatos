import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass # la verdad no se si usar dataclass aqui es lo mejor, pero bueno, me gusta como queda, porque pues es una clase de configuracion y no tiene mucha logica
#solo es tener los datos con los que se va a trabajar y ya xd
class ClusterConfig:
    """Configuración para el algoritmo K-means"""
    k: int = 5
    max_iterations: int = 15
    output_folder: str = "practica7/img"
    random_seed: Optional[int] = 42

class DataProcessor:
    """Procesador de datos para clustering"""
    
    def __init__(self, file_path: str):
        self.df = pd.read_csv(file_path)
        self.numeric_columns = self._get_numeric_columns()
    def _get_numeric_columns(self) -> List[str]:
       
        return self.df.select_dtypes(include=[np.number]).columns.tolist()
    def get_points_for_clustering(self, x_col: str, y_col: str) -> np.ndarray:
       
        valid_data = self.df[[x_col, y_col]].dropna()
        return valid_data.values
    def get_column_names(self) -> List[str]:
       
        return self.numeric_columns.copy()

class KMeansClustering:
    """Implementación del algoritmo K-means"""
    # decidi ponerlo todo esto en una clase para tener todo mas ordenado y no tantas funciones sueltas, ademas de que asi puedo reutilizar codigo mas facil en un futuro si quiero hacer alguna variacion del kmeans
    
    def __init__(self, config: ClusterConfig):
        self.config = config
        self._setup_environment()
    def _setup_environment(self):
        """Crea la carpeta de salida si no existe"""
        os.makedirs(self.config.output_folder, exist_ok=True)
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
    @staticmethod
    def compute_euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
        """Calcula la distancia euclidiana entre dos puntos"""
        return np.sqrt(np.sum((point2 - point1) ** 2))
    
    def _initialize_clusters(self, points: np.ndarray) -> np.ndarray:
        """Inicializa las asignaciones de clusters aleatoriamente"""
        return np.random.randint(0, self.config.k, len(points))
    def _calculate_cluster_centers(self, points: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Calcula los centroides de cada cluster"""
        centers = []
        for cluster_id in range(self.config.k):
            cluster_points = points[labels == cluster_id]
            if len(cluster_points) > 0:
                center = np.mean(cluster_points, axis=0)
            else:
                # Si un cluster está vacío, reinicializa con un punto aleatorio
                center = points[np.random.randint(0, len(points))]
            centers.append(center)
        return np.array(centers)
    def _assign_points_to_clusters(self, points: np.ndarray, centers: np.ndarray) -> np.ndarray:
        
        labels = []
        for point in points:
            distances = [self.compute_euclidean_distance(point, center) for center in centers]
            nearest_cluster = np.argmin(distances)
            labels.append(nearest_cluster)
        return np.array(labels)
    def _has_converged(self, old_centers: np.ndarray, new_centers: np.ndarray) -> bool:
        return np.allclose(old_centers, new_centers)
    def _visualize_iteration(self, points: np.ndarray, labels: np.ndarray, 
                           centers: np.ndarray, iteration: int):
        """Genera visualización de la iteración actual"""
        # Crear dataframe combinado de puntos y centroides, pero diferenciándolos
        points_df = pd.DataFrame(points, columns=['x', 'y'])
        points_df['label'] = labels.astype(str)
        points_df['type'] = 'point'
        
        centers_df = pd.DataFrame(centers, columns=['x', 'y'])
        centers_df['label'] = [f'centroid_{i}' for i in range(len(centers))]
        centers_df['type'] = 'centroid'
        
        combined_df = pd.concat([points_df, centers_df], ignore_index=True)
        
      
        self._create_scatter_plot(combined_df, iteration)
    """Crea un gráfico de dispersión para los clusters, Muy similar a la funcion suya maestro, pero adaptada a mis necesidades por la cantidad de datos"""
    def _create_scatter_plot(self, df: pd.DataFrame, iteration: int):
        fig, ax = plt.subplots(figsize=(10, 8))
        # Graficar puntos por cluster
        for cluster_id in range(self.config.k):
            cluster_points = df[(df['type'] == 'point') & (df['label'] == str(cluster_id))]
            ax.scatter(cluster_points['x'], cluster_points['y'], 
                      label=f'Cluster {cluster_id}', alpha=0.7, s=50)
        # Graficar centroides
        centroids = df[df['type'] == 'centroid']
        ax.scatter(centroids['x'], centroids['y'], 
                  marker='X', s=200, c='black', label='Centroides', edgecolors='red', linewidth=2)
        ax.set_title(f'K-means - Iteración {iteration}')
        ax.set_xlabel('Coordenada X')
        ax.set_ylabel('Coordenada Y')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(f"{self.config.output_folder}/kmeans_iteration_{iteration:02d}.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
    def fit(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Ejecuta el algoritmo K-means"""
        current_labels = self._initialize_clusters(points)
        current_centers = self._calculate_cluster_centers(points, current_labels)
        
        for iteration in range(self.config.max_iterations):
           # se pudiera hacer un try except aqui por si acaso, pero bueno, no creo que sea necesario
            new_labels = self._assign_points_to_clusters(points, current_centers)
            new_centers = self._calculate_cluster_centers(points, new_labels)
            self._visualize_iteration(points, new_labels, new_centers, iteration)
            print(f"Iteración {iteration}: {len(np.unique(new_labels))} clusters activos")
            if self._has_converged(current_centers, new_centers):
                print(f"convergencia alcanzada en la iteracion {iteration}")
                break
            
            current_labels = new_labels
            current_centers = new_centers
        
        return current_labels, current_centers

class ManejadorResultados:
    
    def __init__(self, original_df: pd.DataFrame):
        self.original_df = original_df
    
    def guardar_resultados(self, labels: np.ndarray, centers: np.ndarray, 
                    valid_indices: np.ndarray, output_path: str):
        result_df = self.original_df.copy()
        result_df["cluster"] = np.nan
        result_df.loc[valid_indices, "cluster"] = labels
        
        result_df.to_csv(output_path, index=False)
        
        print(f"\nResultados guardados en: {output_path}")
        print("Centroides finales:")
        for i, center in enumerate(centers):
            print(f"Cluster {i}: {center}")

def main():
 
    config = ClusterConfig(k=5, output_folder="practica7/img")
    
    processor = DataProcessor("practica1/INM_2025_limpia.csv")
    available_columns = processor.get_column_names()

    x_col, y_col = available_columns[0], available_columns[1]
    print(f"\nUsando columnas: '{x_col}' y '{y_col}'")
    
    points = processor.get_points_for_clustering(x_col, y_col)
    valid_indices = processor.df[[x_col, y_col]].dropna().index
    
    print(f"Puntos válidos para clustering: {len(points)}")
    
    # Ejecutar K-means
    kmeans = KMeansClustering(config)
    labels, centers = kmeans.fit(points)
    
    # Guardar resultados
    manejador_resultados =ManejadorResultados(processor.df)
    manejador_resultados.guardar_resultados(labels, centers, valid_indices,"practica7/resultado_kmeans.csv")
    

if __name__ == "__main__":
    main()