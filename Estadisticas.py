"""
Código para generar gráficos a partir de las estadísticas exportadas del algoritmo genético.
Se asume que el archivo Excel 'algorithm_stats.xlsx' contiene las columnas:
'Generaciones', 'Mejor Fitness' y 'Mutacion'
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Leer el archivo Excel con las estadísticas
excel_file = "algorithm_stats.xlsx"
df = pd.read_excel(excel_file)

# Configurar la figura
fig, ax = plt.subplots(figsize=(10, 6))

# Graficar Mejor Fitness vs Generaciones
ax.plot(df['Generaciones'], df['Mejor Fitness'], marker='o', linestyle='-', color='b', label='Mejor Fitness')

# Configurar ticks del eje X de 25 en 25
max_gen = df['Generaciones'].max()
# Genera ticks desde 0 hasta max_gen con intervalos de 25. Se espera que 5000/25 = 200 puntos.
xticks = np.arange(0, max_gen + 1, 25)
ax.set_xticks(xticks)

ax.set_xlabel('Generaciones')
ax.set_ylabel('Mejor Fitness')
ax.set_title('Evolución del Mejor Fitness vs. Generaciones')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()
