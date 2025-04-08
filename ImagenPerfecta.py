import numpy as np
from PIL import Image
import math

# Parámetros
size = 70
center = size // 2   # Centro del lienzo: 350
radius = 20
bg_color = 255
circle_color = 0

# Creamos la matriz llena de blanco (255)
img_array = np.full((size, size), bg_color, dtype=np.uint8)

# Dibujamos el círculo relleno
count_black = 0
for x in range(size):
    for y in range(size):
        # Distancia desde (x, y) al centro
        dist = math.sqrt((x - center)**2 + (y - center)**2)
        
        # Si el píxel está dentro del círculo (incluyendo el borde), se pinta de negro
        if dist <= radius:
            img_array[x, y] = circle_color
            count_black += 1

# Calculamos cuántos píxeles quedaron blancos
count_white = size * size - count_black

print("Píxeles negros (aprox.):", count_black)
print("Píxeles blancos (aprox.):", count_white)

# Guardamos la imagen
img = Image.fromarray(img_array)
img.save("ImgPerfecta.png")

