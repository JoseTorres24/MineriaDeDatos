import pandas as pd
import scipy.stats as stats
import scikit_posthocs as sp

df = pd.read_csv("practica1/INM_2025_limpia.csv")
# ANOVA, Levene, Kruskal-Wallis y post-hoc Dunn test para "bien_juridico_afectado" y "incidencia_delictiva"
groups = [g["incidencia_delictiva"].values for _, g in df.groupby("bien_juridico_afectado")]
# aqui hacemos uso de groupby para agrupar por bien juridico afectado y obtener los valores de incidencia delictiva para cada grupo
anova_res = stats.f_oneway(*groups)
print("ANOVA:", anova_res)


levene_res = stats.levene(*groups)
print("Levene:", levene_res)

kruskal_res = stats.kruskal(*groups)
print("Kruskal-Wallis:", kruskal_res)

posthoc = sp.posthoc_dunn(df, val_col="incidencia_delictiva", group_col="bien_juridico_afectado", p_adjust="bonferroni")
print("Post-hoc Dunn test (p-values):\n", posthoc)
