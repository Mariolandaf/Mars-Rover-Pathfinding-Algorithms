# Mario Alberto Landa Flores
#  BUSQUEDA PROFUNDIZACIÓN ITERATIVA
#------------------------------------------------------------------------------------------------------------------
#   Height map pre-processing
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
#   Imports
#------------------------------------------------------------------------------------------------------------------
import copy
import numpy as np
from skimage.transform import downscale_local_mean

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

import plotly.graph_objects as px

import heapq

#------------------------------------------------------------------------------------------------------------------
#   File names
#------------------------------------------------------------------------------------------------------------------
input_file = "/content/mars_map.IMG"
output_file = "mars_map.npy"

#------------------------------------------------------------------------------------------------------------------
#   Load map data
#------------------------------------------------------------------------------------------------------------------

data_file = open(input_file, "rb")

endHeader = False;
while not endHeader:
    line = data_file.readline().rstrip().lower()

    sep_line = line.split(b'=')

    if len(sep_line) == 2:
        itemName = sep_line[0].rstrip().lstrip()
        itemValue = sep_line[1].rstrip().lstrip()

        if itemName == b'valid_maximum':
            maxV = float(itemValue)
        elif itemName == b'valid_minimum':
            minV = float(itemValue)
        elif itemName == b'lines':
            n_rows = int(itemValue)
        elif itemName == b'line_samples':
            n_columns = int(itemValue)
        elif itemName == b'map_scale':
            scale_str = itemValue.split()
            if len(scale_str) > 1:
                scale = float(scale_str[0])

    elif line == b'end':
        endHeader = True
        char = 0
        while char == 0 or char == 32:
            char = data_file.read(1)[0]
        pos = data_file.seek(-1, 1)

image_size = n_rows*n_columns
data = data_file.read(4*image_size)

image_data = np.frombuffer(data, dtype=np.dtype('f'))
image_data = image_data.reshape((n_rows, n_columns))
image_data = np.array(image_data)
image_data = image_data.astype('float64')

image_data = image_data - minV;
image_data[image_data < -10000] = -1;

#------------------------------------------------------------------------------------------------------------------
#   Subsampling
#------------------------------------------------------------------------------------------------------------------
sub_rate = round(10/scale)

image_data = downscale_local_mean(image_data, (sub_rate, sub_rate))
image_data[image_data<0] = -1

print('Sub-sampling:', sub_rate)

new_scale = scale*sub_rate
print('New scale:', new_scale, 'meters/pixel')

#------------------------------------------------------------------------------------------------------------------
#   Save map
#------------------------------------------------------------------------------------------------------------------
np.save(output_file, image_data)

#------------------------------------------------------------------------------------------------------------------
#   Show 3D surface
#------------------------------------------------------------------------------------------------------------------

x = new_scale*np.arange(image_data.shape[1])
y = new_scale*np.arange(image_data.shape[0])
X, Y = np.meshgrid(x, y)

fig = px.Figure(data = px.Surface(x=X, y=Y, z=np.flipud(image_data), colorscale='hot', cmin = 0,
                           lighting = dict(ambient = 0.0, diffuse = 0.8, fresnel = 0.02, roughness = 0.4, specular = 0.2),
                           lightposition=dict(x=0, y=n_rows/2, z=2*maxV)),

                layout = px.Layout(scene_aspectmode='manual',
                                   scene_aspectratio=dict(x=1, y=n_rows/n_columns, z=max((maxV-minV)/x.max(), 0.2)),
                                   scene_zaxis_range = [0,maxV-minV])
                )



#------------------------------------------------------------------------------------------------------------------
#   Show surface image
#------------------------------------------------------------------------------------------------------------------

cmap = copy.copy(plt.cm.get_cmap('autumn'))
cmap.set_under(color='black')

ls = LightSource(315, 45)
rgb = ls.shade(image_data, cmap=cmap, vmin = 0, vmax = image_data.max(), vert_exag=2, blend_mode='hsv')

fig, ax = plt.subplots()

im = ax.imshow(rgb, cmap=cmap, vmin = 0, vmax = image_data.max(),
                extent =[0, scale*n_columns, 0, scale*n_rows],
                interpolation ='nearest', origin ='upper')

cbar = fig.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Altura (m)')

plt.title('Superficie de Marte')
plt.xlabel('x (m)')
plt.ylabel('y (m)')

plt.show()


# Cargar el mapa de alturas de Marte
mars_map = np.load('mars_map.npy')
nr, nc = mars_map.shape

# Definir función para convertir coordenadas (x, y) a índices de matriz (r, c)
def xy_to_rc(x, y, scale):
    r = nr - round(y / scale)
    c = round(x / scale)
    return r, c

# Definir función para obtener vecinos válidos
#Esta función toma un índice de matriz (r, c) y devuelve una lista de los vecinos válidos en el mapa de alturas de Marte.
#Solo se consideran vecinos que no están fuera de los límites del mapa, que no son obstáculos (-1),
#y cuya diferencia de altura con respecto al nodo actual está dentro del umbral especificado.
def get_neighbors(r, c, mars_map, scale, height_diff_threshold=0.5):
    neighbors = []
    deltas = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for dr, dc in deltas:
        nr, nc = r + dr, c + dc
        if 0 <= nr < mars_map.shape[0] and 0 <= nc < mars_map.shape[1]:
            if mars_map[nr, nc] != -1 and abs(mars_map[r, c] - mars_map[nr, nc]) <= height_diff_threshold:
                neighbors.append((nr, nc))
    return neighbors

# Definir algoritmo de búsqueda en profundización iterativa
#Esta función implementa el algoritmo de búsqueda de profundización iterativa.
#Itera sobre límites de profundidad crecientes, desde 1 hasta el número total de nodos en el mapa.
#Llama a la función dfs para realizar la búsqueda en profundidad limitada con el límite actual y verifica si se encontró una ruta válida.
def iterative_deepening_search(start, goal, mars_map, scale):
    start = xy_to_rc(*start, scale)
    goal = xy_to_rc(*goal, scale)

    for depth_limit in range(1, nr * nc):  # Límite de profundidad incrementado en cada iteración
        visited = set()
        path = dfs(start, goal, mars_map, scale, depth_limit, visited)
        if path:
            return path
    return []

# Definir función de búsqueda en profundidad
#Esta función realiza la búsqueda en profundidad limitada desde un nodo dado hasta el objetivo.
#Si se encuentra el objetivo, devuelve el camino desde el nodo actual hasta el objetivo.
#Si se alcanza el límite de profundidad sin encontrar el objetivo, devuelve una lista vacía.
def dfs(current_node, goal, mars_map, scale, depth_limit, visited):
    if current_node == goal:
        return [current_node]

    if depth_limit == 0:
        return []

    visited.add(current_node)

    for neighbor in get_neighbors(*current_node, mars_map, scale):
        if neighbor not in visited:
            path = dfs(neighbor, goal, mars_map, scale, depth_limit - 1, visited)
            if path:
                return [current_node] + path

    return []

# Coordenadas de inicio y fin
start_point = (2850, 6400)
end_point = (3150, 6800)

# Calcular ruta utilizando búsqueda en profundización iterativa
path = iterative_deepening_search(start_point, end_point, mars_map, 10.0174)

# Mostrar el mapa de alturas y la ruta si existe
if path:
    plt.figure(figsize=(10, 5))
    plt.imshow(mars_map, cmap='hot', extent=(0, nc*10.0174, 0, nr*10.0174))
    plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'bo-')
    plt.plot([p[1] * 10.0174 for p in path], [(nr - p[0]) * 10.0174 for p in path], 'r.-')
    plt.title(f'Ruta desde {start_point} hasta {end_point}. Distancia: {len(path) - 1:.2f} metros')
    plt.xlabel('Coordenada X (metros)')
    plt.ylabel('Coordenada Y (metros)')
    plt.colorbar(label='Altura (metros)')
    plt.grid(True)
    plt.show()
else:
    print("No se pudo encontrar una ruta válida desde el punto de inicio hasta el punto objetivo.")
