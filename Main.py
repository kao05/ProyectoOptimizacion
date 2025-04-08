"""
Dev:Kevin Vargas Flores
Fecha: 08-04-25
Descripcion: 
Proyecto de optimizacion de imagenes usando algoritmos geneticos y hill climbing.
"""


#Librerias
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
from scipy.ndimage import distance_transform_edt
import pandas as pd

# ---------------------------
# 1. PRECOMPUTACIÓN DE LA TRANSFORMADA DE DISTANCIA
# ---------------------------
def precompute_distance_transform(target_array):
    """
    Precomputa la transformada de distancia de la imagen objetivo.
    recompensa extra en la función fitness si los píxeles 
    generados están cerca del objeto real, incluso si no están exactamente sobre él.
    """
    target_black = (target_array == 0)
    #sirve para identificar los píxeles del objeto objetivo
    distances = distance_transform_edt(~target_black)
    """
    calcula, para cada píxel blanco (fondo), cuál es la distancia al píxel negro más cercano (parte del objeto).
    Esto da como resultado un mapa de distancias, donde cada número representa 
    cuán lejos está ese píxel del objeto negro más cercano.
    """
    return distances
    """
    una matriz del mismo tamaño que la imagen, con números flotantes (float) 
    que indican la distancia euclidiana a la silueta (píxeles negros).
    """

# ---------------------------
# 2. FITNESS MEJORADO: DECAY EXPONENCIAL
# ---------------------------
def fitness_vectorized_improved(canvas, target_array, distance_matrix, alpha=1.0):
    """
    Calcula el fitness del canvas evaluando solo los píxeles pintados (diferentes de 255).
    A cada píxel se le asigna un score según exp(-alpha * distancia) que varía de 1 (error cero)
    a 0 a medida que la distancia aumenta.
    
    Se retorna el promedio del score entre los píxeles pintados.
    """
    painted_mask = canvas != 255
    total_painted = painted_mask.sum()
    if total_painted == 0:
        return 0.0
    distances = distance_matrix[painted_mask]
    # Se asigna score de 1 para distancia 0 y decae exponencialmente con alpha
    scores = np.exp(-alpha * distances)
    return scores.sum() / total_painted

# ---------------------------
# 3. CANVAS Y OPERACIONES SOBRE INDIVIDUOS
# ---------------------------
def create_canvas_from_S(S, alto, ancho):
    """
    Crea un canvas en blanco (valor 255) y pinta en negro (valor 0) las posiciones indicadas en S.
    S es una lista de tuplas (pixIndex, intensidad). En nuestro caso, intensidad = 0.
    """
    canvas = np.full((alto, ancho), 255, dtype=np.uint8)
    for (pixIndex, intensidad) in S:
        y = pixIndex // ancho
        x = pixIndex % ancho
        canvas[y, x] = intensidad
    return canvas

def create_random_individual(Tinta, ancho, alto):
    """
    Crea un individuo aleatorio con Tinta píxeles pintados (sin repetir).
    """
    total_pixels = ancho * alto
    pixel_indices = random.sample(range(total_pixels), Tinta)
    S = [(idx, 0) for idx in sorted(pixel_indices)] #Hace un for para generar los piceles que se pintaran en el 1D
    return S

def fix_individual(S, Tinta, ancho, alto):
    """
    Corrige el individuo S para eliminar duplicados y garantizar que contenga Tinta genes.
    """
    total_pixels = ancho * alto
    unique_indices = list(set([g[0] for g in S]))
    while len(unique_indices) < Tinta:
        candidate = random.randint(0, total_pixels - 1)
        if candidate not in unique_indices:
            unique_indices.append(candidate)
    if len(unique_indices) > Tinta:
        unique_indices = random.sample(unique_indices, Tinta)
    unique_indices.sort()
    new_S = [(idx, 0) for idx in unique_indices]
    return new_S

def evaluate_individual(S, target_array, ancho, alto, distance_matrix, alpha=1.0):
    """
    Evalúa un individuo generando su canvas y calculando su fitness con la función mejorada.
    Retorna un diccionario con 'S', 'fitness' y 'canvas'.
    """
    canvas = create_canvas_from_S(S, alto, ancho) #Manda a la funcion de pasar el arreglo de 1D a 2D
    fit_val = fitness_vectorized_improved(canvas, target_array, distance_matrix, alpha) #Funcion para ver que tan cerca estan los pixeles pintados de los objetivos
    return {"S": S, "fitness": fit_val, "canvas": canvas}

# ---------------------------
# 4. OPERADORES GENÉTICOS
# ---------------------------
def one_point_crossover(S1, S2, crossover_point):
    """
    Cruce one-point: intercambia bloques del genoma a partir de la posición crossover_point.
    """
    offspring1 = S1[:crossover_point] + S2[crossover_point:]
    offspring2 = S2[:crossover_point] + S1[crossover_point:]
    return offspring1, offspring2

def mutate(S, mutation_prob, ancho, alto):
    """
    - `S`: lista de genes (cada uno es una tupla `(pixIndex, intensity)`).
    - `mutation_prob`: probabilidad de que cada gen sufra mutación.
    - `ancho`, `alto`: dimensiones del lienzo para calcular el número total de píxeles.
    """
    total_pixels = ancho * alto
    new_S = [] #Crea una lista vacía para guardar el nuevo genoma mutado.
    for (pixIndex, intensity) in S:
        if random.random() < mutation_prob: #Con cierta probabilidad (`mutation_prob`), se decide si este gen se muta. `random.random()` devuelve un número entre 0 y 1.
            new_index = random.randint(0, total_pixels - 1)
            while new_index == pixIndex:
                new_index = random.randint(0, total_pixels - 1)
            new_S.append((new_index, 0)) #Guarda el nuevo gen mutado (mismo color, solo cambia la posición del píxel).
        else:
            new_S.append((pixIndex, intensity))
    return new_S

def hill_climbing(individual, target_array, ancho, alto, distance_matrix, iterations=10, alpha=1.0):
    """
    Aplica hill climbing (búsqueda local) al individuo intentando mejorar su fitness mediante
    pequeñas mutaciones aceptadas solo si mejoran el fitness.
    """
    """
    Parametros
    individual: el individuo a mejorar ({"S": ..., "fitness": ..., "canvas": ...})
    target_array: la imagen objetivo (como matriz de píxeles).
    ancho, alto: dimensiones del lienzo.
    distance_matrix: matriz de distancias (usada para calcular la similitud de imágenes).
    iterations: cuántas veces intentar mejorar al individuo.
    alpha: peso usado para ajustar el cálculo de fitness.
    """
    best_ind = individual.copy() #Guarda el individuo actual como el "mejor encontrado hasta ahora"
    for _ in range(iterations): #Repite la búsqueda local iterations veces (por defecto, 10 veces).
        mutated = best_ind["S"].copy() #Copia los genes del mejor individuo actual (S) para hacer una mutación sobre él.
        idx = random.randint(0, len(mutated) - 1) # Escoge al azar uno de los genes del individuo para mutar.
        total_pixels = ancho * alto # Calcula el número total de píxeles en el lienzo.
        new_index = random.randint(0, total_pixels - 1) # Genera un nuevo índice aleatorio para el píxel a mutar.
        #Aquí se elige un nuevo índice de píxel que no esté ya usado en S, para evitar duplicados.
        while new_index in [g[0] for g in mutated]:
            new_index = random.randint(0, total_pixels - 1)
        mutated[idx] = (new_index, 0) # Reemplaza el gen seleccionado por uno nuevo, en otra posición.
        mutated = fix_individual(mutated, len(mutated), ancho, alto)# Se asegura de que el individuo no tenga duplicados y tenga exactamente la cantidad de genes correcta (Tinta).
        canvas_temp = create_canvas_from_S(mutated, alto, ancho) #
        fit_temp = fitness_vectorized_improved(canvas_temp, target_array, distance_matrix, alpha) #Calcula el fitness del nuevo individuo mutado.
        #Si el nuevo individuo tiene un mejor fitness, se actualiza como el mejor encontrado hasta ahora.
        if fit_temp > best_ind["fitness"]:
            best_ind = {"S": mutated, "fitness": fit_temp, "canvas": canvas_temp}
    return best_ind

# ---------------------------
# 5. MUTACIÓN ADAPTATIVA
# ---------------------------
def adaptive_mutation_rate(generation, stagnation, initial_rate=0.05, max_rate=0.12, min_rate=0.01):
    """
    Ajusta la tasa de mutación:
      - Si hay estancamiento (generaciones sin mejora) se aumenta la tasa hasta max_rate.
      - Si hay mejoras se reduce gradualmente la tasa.
    """
    if stagnation > 30:
        # Si está estancado, incrementa la tasa en función del número de generaciones sin mejora.
        return min(max_rate, initial_rate * (1 + stagnation / 50.0))
    else:
        # Disminuye la tasa conforme la generación avanza.
        return max(min_rate, initial_rate * (0.99 ** generation))

# ---------------------------
# 6. ALGORITMO GENÉTICO 
# ---------------------------
def genetic_algorithm(
    target_array,
    ancho,
    alto,
    Tinta,
    distance_matrix,
    max_generations=5000,
    pop_size=400,
    selection_percent=0.40,   # Top 40% se consideran padres
    children_best=20,         # Seleccionamos 20 de los hijos
    initial_mutation_prob=0.05,
    alpha=1.0,                # Parámetro para el fitness exponencial
    show_plot=True
):
    """
    Ejecuta el algoritmo genético completo:
      - Se genera población inicial.
      - Se selecciona el top 40% y se cruzan en pares usando crossover one-point.
      - Se aplica mutación adaptativa y se corrigen duplicados.
      - Se evalúan los hijos y se complementa la población con individuos aleatorios.
      - Cada 30 generaciones se aplica hill climbing intensivo al 20% superior.
      - Se visualiza el mejor individuo en tiempo real.
      
    Se detiene cuando se alcanza un fitness de 1 o se llega a max_generations.
    """
    population = [] # Para almacenar la población 
    # Inicializar la población con individuos aleatorios
    for _ in range(pop_size):
        S = create_random_individual(Tinta, ancho, alto) #Pinta un individuo y lo añade a la población
        ind = evaluate_individual(S, target_array, ancho, alto, distance_matrix, alpha)
        population.append(ind) #Agrega el fitness de cada individuo a la población
    
    stagnation_counter = 0 #  lleva la cuenta de cuántas generaciones consecutivas han pasado sin mejora en el mejor fitness de la población.
    best_overall = max(population, key=lambda x: x["fitness"]) #Encuentra al mejor individuo de la población actual basándose en su valor de fitness.
    

    # Lista para almacenar estadísticas: generaciones, mejor fitness y mutación
    generation_stats = []
    if show_plot: # Verifica si la visualización está activada 
        plt.ion() # Activa el modo interactivo de matplotlib para actualizar la visualización en tiempo real.|
        fig, ax = plt.subplots(figsize=(6,6))
        best_ind = best_overall
        im = ax.imshow(best_ind["canvas"], cmap='gray', vmin=0, vmax=255)
        ax.axis('off')
        title = ax.set_title(f"Gen: 0 | Fit: {best_ind['fitness']:.4f}", fontsize=10)
        plt.show()
    
    generation = 0 #Inicializa un contador que lleva el registro del número de generaciones que ha procesado el algoritmo genético.
    last_best = best_overall["fitness"] #Almacena el mejor valor de fitness de la generación anterior para detectar estancamiento.
    
    #NUCLEO PRINCIPAL DEL ALGORITMO GENETICO
    while best_overall["fitness"] < 1.0 and generation < max_generations: #Repite el proceso hasta que se alcance un fitness de 1 o se llegue al límite de generaciones.
        generation += 1 # Incrementa el contador de generaciones
        
        # Ordenar la población (mejores primero)
        population.sort(key=lambda x: x["fitness"], reverse=True)
        current_best = population[0]["fitness"]
        if current_best <= last_best:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
            last_best = current_best
        """
        -Ordenar población: Los individuos se ordenan de mayor a menor fitness.
        -Detectar estancamiento:
            -Si no hay mejora en el fitness (current_best <= last_best), aumenta stagnation_counter.
            -Si hay mejora, reinicia el contador y actualiza last_best.
        """
        # Actualizar tasa de mutación de forma adaptativa
        mutation_prob = adaptive_mutation_rate(generation, stagnation_counter, initial_mutation_prob)
        """
        Aumenta si hay estancamiento (para diversificar la población).
        Disminuye si hay mejoras (para explotar soluciones prometedoras).
        """


        # Seleccionar los padres (top selection_percent)
        n_sel = int(pop_size * selection_percent) # Cantidad de padres que tomara
        parents = population[:n_sel] # Selecciona el top n_sel de la población ordenada
        
        # Cruzamiento (crossover)
        random.shuffle(parents)
        """
        - Modifica la lista original parents "in-place" (no devuelve una nueva lista, sino que reordena la existente).
        -Desordena los elementos de forma aleatoria usando un algoritmo de Fisher-Yates (un método eficiente para permutaciones aleatorias).
        """
        children = [] # Lista para almacenar los hijos generados
        cross_point = int(0.25 * Tinta)  # Punto de corte (25% del genoma)
        
        #Recorre los padres de dos en dos (pares) y aplica el cruce one-point.
        for i in range(0, len(parents) - 1, 2):
            p1 = parents[i]["S"] #Extrae el elemento i de la lista S
            p2 = parents[i+1]["S"]
            #Toma la funcion de cruzar los padres y genera dos descendientes
            c1, c2 = one_point_crossover(p1, p2, cross_point)
            # Aplicar mutación
            c1 = mutate(c1, mutation_prob, ancho, alto)
            c2 = mutate(c2, mutation_prob, ancho, alto)
            #  corrige a un individuo (o cromosoma), si después de la mutación o cruce, tiene píxeles repetidos o no tiene la cantidad exacta de genes que debería.
            c1 = fix_individual(c1, Tinta, ancho, alto)
            c2 = fix_individual(c2, Tinta, ancho, alto)
            #Funcion para evaluar el individuo y calcular su fitness
            child1 = evaluate_individual(c1, target_array, ancho, alto, distance_matrix, alpha)
            child2 = evaluate_individual(c2, target_array, ancho, alto, distance_matrix, alpha)
            #Agrega los hijos a la lista de hijos generados
            children.append(child1)
            children.append(child2)
        
        # Seleccionar los mejores hijos
        children.sort(key=lambda x: x["fitness"], reverse=True)
        best_children = children[:children_best]
        
        # Nueva población: padres + mejores hijos + individuos aleatorios para completar
        new_population = parents + best_children
        while len(new_population) < pop_size:
            S_new = create_random_individual(Tinta, ancho, alto)
            ind_new = evaluate_individual(S_new, target_array, ancho, alto, distance_matrix, alpha)
            new_population.append(ind_new)
        
        population = new_population
        
        # Aplicar hill climbing intensivo al 20% de la elite cada 30 generaciones
        if generation % 30 == 0: #Cada 30 generaciones se aplica el algoritmo
            elite_count = max(1, int(0.20 * pop_size)) #Calcula cuántos individuos forman la élite, es decir, los mejores.
            for i in range(elite_count): #Este bucle recorre los primeros elite_count individuos de la población.
                # a partir de un individuo, se prueban pequeñas modificaciones (vecinos) y se queda con el mejor.
                improved = hill_climbing(population[i], target_array, ancho, alto, distance_matrix, iterations=20, alpha=alpha)
                population[i] = improved #Reemplazas el individuo original por su versión mejorada tras el hill climbing.
                #El hill climbing explota soluciones prometedoras a fondo.

        best_overall = max(population, key=lambda x: x["fitness"])
        
        if show_plot:
            im.set_data(best_overall["canvas"])
            title.set_text(f"Gen: {generation} | Fit: {best_overall['fitness']:.4f} | Mut: {mutation_prob:.4f}")
            plt.draw()
            plt.pause(0.01)
        
        print(f"Gen: {generation} | Mejor fitness: {best_overall['fitness']:.4f} | Mut: {mutation_prob:.4f}")
    
    if show_plot:
        plt.ioff()
        plt.show()
    
    # Exportar las estadísticas a un archivo Excel
    df_stats = pd.DataFrame(generation_stats)
    excel_file = "algorithm_stats.xlsx"
    df_stats.to_excel(excel_file, index=False)
    print(f"Estadísticas exportadas a {excel_file}")

    return best_overall

# ---------------------------
# 7. BLOQUE PRINCIPAL: MAIN
# ---------------------------
if __name__ == "__main__":
    # Cargar la imagen objetivo en escala de grises ("L")
    target_im = Image.open("ImGoal360.png").convert("L")
    target_array = np.array(target_im) #Convertir la imagen en un array 
    
    alto, ancho = target_array.shape # Obtener dimensiones de la imagen objetivo
    # Definir Tinta: número de píxeles negros de la imagen objetivo
    Tinta = int(np.sum(target_array == 0))
    print(Tinta)


    # Precomputar la transformada de distancia de la imagen objetivo
    # Crea una matriz con valores flotantes de la cercania de los pixeles objetivo
    distance_matrix = precompute_distance_transform(target_array)
    
    # Ejecutar el algoritmo genético con las mejoras implementadas
    best_individual = genetic_algorithm(
        target_array=target_array,
        ancho=ancho,
        alto=alto,
        Tinta=Tinta,
        distance_matrix=distance_matrix,
        max_generations=5000, #Limite de generaciones
        pop_size=400, #Tamano de las poblaciones
        selection_percent=0.40, # Porcentaje de la población que se considera "elite" o "apta" y se selecciona para cruzarse.
        children_best=20, #Número de hijos generados a partir de los mejores individuos de cada generación.
        initial_mutation_prob=0.05, #Probabilidad inicial de mutación.
        alpha=1.0, # Peso que se le da al componente de proximidad
        show_plot=True # Visualizar en tiempo real la evolución
    )
    
    print(f"Mejor fitness encontrado = {best_individual['fitness']:.4f}")
    if best_individual["fitness"] >= 0.9999:
        print("Se encontró un individuo con fitness cercano a 1!")
    
    # Guardar la imagen final resultante
    final_canvas = best_individual["canvas"]
    final_img = Image.fromarray(final_canvas, mode="L")
    final_img.save("Imagen_Genetico_Final.png")