import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('practica1/INM_2025_limpia.csv')

#HAREMOS muchos graficos jejejej asi que pues aqui van de acuerdo a lo que podemos realizar
# aqui realizamos un scatter plot para ver la relasao con el anio y la incidencia delictiva para tambien ver como varia por tipo de delito
sns.set(style="whitegrid")
plt.figure(figsize=(50, 30))
#podriamos usar la
sns.scatterplot(x='anio', y='incidencia_delictiva',hue='tipo_delito', data=df)
plt.title('Incidencia Delictiva por Año y Tipo de Delito')
plt.xlabel('Año')
plt.ylabel('Incidencia Delictiva')
plt.legend(title='Tipo de Delito', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('scatter_plot_incidencia_tipo_delito.png')
plt.show()


#podemos hacer un histograma ya que tenemos los anios y la incidencia delictiva
plt.figure(figsize=(32, 24))
sns.histplot(data=df, x='anio', y='incidencia_delictiva')
plt.title('Histograma de Incidencia Delictiva por Año')
plt.xlabel('Año')
plt.ylabel('Incidencia Delictiva')
plt.tight_layout()
plt.savefig('histograma_incidencia_anio.png')
plt.show()

#
#podemos hacer un boxplot para ver la distribucion de la incidencia delictiva por entidad federativa
plt.figure(figsize=(32, 24))
sns.boxplot(x='entidad_federativa', y='incidencia_delictiva', data=df)
plt.title('Distribución de Incidencia Delictiva por Entidad Federativa')
plt.xlabel('Entidad Federativa')
plt.ylabel('Incidencia Delictiva')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('boxplot_incidencia_entidad_federativa.png')
plt.show()
#tambien podemos usar pie dramagm para realizar la proporcion de incidencia delictiva por entidad federativa
incidencia_por_entidad = df.groupby('entidad_federativa')['incidencia_delictiva'].sum().sort_values(ascending=False) # aqui hacemos uso de groupby para agrupar por entidad federativa y sumar la incidencia delictiva
plt.figure(figsize=(40, 30))
plt.pie(incidencia_por_entidad, labels=incidencia_por_entidad.index, autopct='%1.1f%%', startangle=140)
plt.title('Proporción de Incidencia Delictiva por Entidad Federativa')
plt.axis('equal')
plt.tight_layout()
plt.savefig('graficadepastel_incidencia_entidad_federativa.png')
plt.show()