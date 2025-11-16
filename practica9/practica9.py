import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


df = pd.read_csv("practica1/INM_2025_limpia.csv")
columnas = ["bien_juridico_afectado","tipo_delito","subtipo_delito","modalidad"]

texto = ""

for col in columnas:
    texto += " ".join(df[col].astype(str).tolist()) + " "
stopwords = set(STOPWORDS)
stopwords.update(["el", "la", "los", "las", "un", "una", "unos", "unas"]) #es que la verdad no queda bonito el word cloud con tantas palabras que no sean las exactas referidas

palabras_filtradas = []
for palabra in texto.split():
    if "y" not in palabra.lower():  # si NO contiene "y"
        palabras_filtradas.append(palabra)

texto_filtrado = " ".join(palabras_filtradas)


wc = WordCloud(
    width=1000,
    height=500,
    background_color="white",
    stopwords=stopwords
).generate(texto_filtrado)

## le di una limpia al word cloud para que no tenga palabras como articulos y eso, que sea mas enfocado a las palabras importantes
plt.figure(figsize=(12,6))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.savefig("practica9/nube_palabras.png", bbox_inches="tight", dpi=300)
plt.show()

wc.to_file("practica9/nube_palabras.png")
