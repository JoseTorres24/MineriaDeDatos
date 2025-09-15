import pandas as pd

df = pd.read_csv('practica1/INM_2025_limpia.csv')
#realmente es dificil describir y desmenuzar la tabla en varias entidades para asi agruparlas y realizar juegos con la informacion mas a detalle
""" cuando realizmoas especificamente este tipo de tarea hay formas de obtener la infomracion de acuerdo a cosas que queremos buscar en el csv
, como tenemos iloc y loc que nos permiten encontrar informacion como si querys en sql, pero obvio estos solo hecho en pandas """
# aqui hacemos un ejemplo de la funcion de head de la tabla que imprime las primeras filas
head_datos = df.iloc[0:5,:]
print(head_datos)

#pero con loc podemos hacer comparaciones y busquedas mas especificas para obtener segmentos de datos que queremos, en este caso pues los relacionados
#a nuevo leon
nuevoleon_datos = df.loc[df.entidad_federativa == 'Nuevo León']
print(nuevoleon_datos)

#aqui lo que buscamos es que me de en cuestion en es nuevo leon, en un formato se Series su numero de incidencia delectiva por anio y mes
# y ordenanarlos de mayor a menor
incidenciaEnNuevoLeon = nuevoleon_datos.groupby(['entidad_federativa','anio','mes_num']).incidencia_delictiva.sum().sort_values(ascending=False)
print(incidenciaEnNuevoLeon)
"""Ahora podemos buscar diferentes cosas con este podremos buscasr difefentes cosas respecto a la indencia delectiva en nuevo leon por su mes y dia, asi que 
pues es mas facil determinar cosas con los meses y el anio, como pueden ser promedios, menores y mayores, etc"""
print('incidencia en nuevo leon por mes y anio')
print(incidenciaEnNuevoLeon.loc[('Nuevo León',[2019])].mean())
print(incidenciaEnNuevoLeon.loc[('Nuevo León',[2019])].max())
print(incidenciaEnNuevoLeon.loc[('Nuevo León',[2019])].min())  #
# pero lo podemos hacer tanbie para los meses para obtener pues su promedio, maximo y minimo en incidencia delectiva de acuerdo a las anios transcurridos
print(incidenciaEnNuevoLeon.groupby('mes_num').agg(['mean','max','min']))

#esto lo podemos repetir con otro estado    
# Filtramos por otra entidad, ejemplo: Jalisco
jalisco_datos = df.loc[df.entidad_federativa == 'Jalisco']
print(jalisco_datos)

# Serie con incidencia por año y mes, ordenada
incidenciaEnJalisco = (
    jalisco_datos
    .groupby(['entidad_federativa','anio','mes_num'])
    .incidencia_delictiva
    .sum()
    .sort_values(ascending=False)
)
print(incidenciaEnJalisco)

print('Incidencia en Jalisco por mes y anio (2019)')
print("Mean:", incidenciaEnJalisco.loc[('Jalisco',[2019])].mean())
print("Max:", incidenciaEnJalisco.loc[('Jalisco',[2019])].max())
print("Min:", incidenciaEnJalisco.loc[('Jalisco',[2019])].min())

# Estadísticas agrupadas por mes (en todos los años disponibles)
print("Promedio, máximo y mínimo de incidencia por mes en Jalisco")
print(incidenciaEnJalisco.groupby('mes_num').agg(['mean','max','min']))

