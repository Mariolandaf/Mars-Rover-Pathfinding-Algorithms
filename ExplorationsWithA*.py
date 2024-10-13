# Mario Alberto Landa Flores
#  BUSQUEDA A* ----- DISTANCIA MENOR A 500 METROS
#------------------------------------------------------------------------------------------------------------------
#   Height map pre-processing
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

#Esta función convierte coordenadas (en metros) a índices de matriz (filas y columnas)
#basadas en el tamaño de píxel y las dimensiones del mapa de alturas.
def xy_to_rc(x, y, scale):
    r = nr - round(y / scale)
    c = round(x / scale)
    return r, c

# Definir función heurística para A* (distancia de Manhattan ponderada por la diferencia de altura)
#Esta función calcula la distancia de Manhattan entre los nodos a y b (la suma de las diferencias en las coordenadas x y y),
#y también tiene en cuenta la diferencia de altura entre los nodos.
#Esto proporciona una estimación del costo adicional necesario para llegar desde el nodo actual hasta el nodo objetivo,
#lo que guía al algoritmo en la búsqueda de la ruta más corta.
def heuristic(a, b, mars_map, scale):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    dz = abs(mars_map[a[0], a[1]] - mars_map[b[0], b[1]])
    return dx + dy + dz

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

# Definir algoritmo A*
#Esta función implementa el algoritmo A* para encontrar la ruta más corta desde el punto de inicio hasta
#el punto objetivo en el mapa de alturas de Marte. Utiliza una cola de prioridad (frontier) para expandir los nodos,
#manteniendo un registro de los costos acumulados (cost_so_far) y los nodos padres (came_from).
#Calcula la prioridad de cada nodo en función de la suma del costo acumulado y la heurística.
#Una vez que se alcanza el objetivo, reconstruye el camino desde el punto de inicio hasta el punto objetivo.
def astar(start, goal, mars_map, scale):
    start = xy_to_rc(*start, scale)
    goal = xy_to_rc(*goal, scale)

    frontier = [(0, start)]
    came_from = {}
    cost_so_far = {start: 0}

    while frontier:
        current_cost, current_node = heapq.heappop(frontier)

        if current_node == goal:
            break

        for next_node in get_neighbors(*current_node, mars_map, scale):
            new_cost = cost_so_far[current_node] + 1
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost + heuristic(goal, next_node, mars_map, scale)
                heapq.heappush(frontier, (priority, next_node))
                came_from[next_node] = current_node

    # Reconstruir el camino si existe
    #Esta sección del código comienza desde el nodo objetivo y sigue retrocediendo a lo largo de la ruta,
    #agregando cada nodo al camino hasta que llega al nodo de inicio.
    #Luego invertimos el camino para que esté en el orden correcto (desde el punto de inicio hasta el punto objetivo).
    #Finalmente, convertimos los índices de matriz de la ruta en coordenadas (x, y) para que sea más fácil de entender y visualizar.
    if goal in came_from:
        path = [goal]
        while path[-1] != start:
            path.append(came_from[path[-1]])
        path = [(rc[1] * scale, (nr - rc[0]) * scale) for rc in reversed(path)]
        distance = cost_so_far[goal]
    else:
        print("No se pudo encontrar una ruta válida desde el punto de inicio hasta el punto objetivo.")
        path = []
        distance = None

    return path, distance

# Coordenadas de inicio y fin
start_point = (3000, 6000)
end_point = (3150, 7500)

# Calcular ruta utilizando A*
path, distance = astar(start_point, end_point, mars_map, 10.0174)

# Mostrar el mapa de alturas y la ruta si existe
if path:
    plt.figure(figsize=(10, 5))
    plt.imshow(mars_map, cmap='hot', extent=(0, nc*10.0174, 0, nr*10.0174))
    plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'bo-')
    plt.plot([p[0] for p in path], [p[1] for p in path], 'r.-')
    plt.title(f'Ruta desde {start_point} hasta {end_point}. Distancia: {distance:.2f} metros')
    plt.xlabel('Coordenada X (metros)')
    plt.ylabel('Coordenada Y (metros)')
    plt.colorbar(label='Altura (metros)')
    plt.grid(True)
    plt.show()

fig.show()

#  BUSQUEDA A* ----- DISTANCIA MAYOR A 1000 METROS Y MENOR A 1500 METROS
#------------------------------------------------------------------------------------------------------------------
#   Height map pre-processing 
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

#Esta función convierte coordenadas (en metros) a índices de matriz (filas y columnas)
#basadas en el tamaño de píxel y las dimensiones del mapa de alturas.
def xy_to_rc(x, y, scale):
    r = nr - round(y / scale)
    c = round(x / scale)
    return r, c

# Definir función heurística para A* (distancia de Manhattan ponderada por la diferencia de altura)
#Esta función calcula la distancia de Manhattan entre los nodos a y b (la suma de las diferencias en las coordenadas x y y),
#y también tiene en cuenta la diferencia de altura entre los nodos.
#Esto proporciona una estimación del costo adicional necesario para llegar desde el nodo actual hasta el nodo objetivo,
#lo que guía al algoritmo en la búsqueda de la ruta más corta.
def heuristic(a, b, mars_map, scale):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    dz = abs(mars_map[a[0], a[1]] - mars_map[b[0], b[1]])
    return dx + dy + dz

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

# Definir algoritmo A*
#Esta función implementa el algoritmo A* para encontrar la ruta más corta desde el punto de inicio hasta
#el punto objetivo en el mapa de alturas de Marte. Utiliza una cola de prioridad (frontier) para expandir los nodos,
#manteniendo un registro de los costos acumulados (cost_so_far) y los nodos padres (came_from).
#Calcula la prioridad de cada nodo en función de la suma del costo acumulado y la heurística.
#Una vez que se alcanza el objetivo, reconstruye el camino desde el punto de inicio hasta el punto objetivo.
def astar(start, goal, mars_map, scale):
    start = xy_to_rc(*start, scale)
    goal = xy_to_rc(*goal, scale)

    frontier = [(0, start)]
    came_from = {}
    cost_so_far = {start: 0}

    while frontier:
        current_cost, current_node = heapq.heappop(frontier)

        if current_node == goal:
            break

        for next_node in get_neighbors(*current_node, mars_map, scale):
            new_cost = cost_so_far[current_node] + 1
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost + heuristic(goal, next_node, mars_map, scale)
                heapq.heappush(frontier, (priority, next_node))
                came_from[next_node] = current_node

    # Reconstruir el camino si existe
    #Esta sección del código comienza desde el nodo objetivo y sigue retrocediendo a lo largo de la ruta,
    #agregando cada nodo al camino hasta que llega al nodo de inicio.
    #Luego invertimos el camino para que esté en el orden correcto (desde el punto de inicio hasta el punto objetivo).
    #Finalmente, convertimos los índices de matriz de la ruta en coordenadas (x, y) para que sea más fácil de entender y visualizar.
    if goal in came_from:
        path = [goal]
        while path[-1] != start:
            path.append(came_from[path[-1]])
        path = [(rc[1] * scale, (nr - rc[0]) * scale) for rc in reversed(path)]
        distance = cost_so_far[goal]
    else:
        print("No se pudo encontrar una ruta válida desde el punto de inicio hasta el punto objetivo.")
        path = []
        distance = None

    return path, distance

# Coordenadas de inicio y fin
start_point = (5000, 2000)
end_point = (4500, 10000)

# Calcular ruta utilizando A*
path, distance = astar(start_point, end_point, mars_map, 10.0174)

# Mostrar el mapa de alturas y la ruta si existe
if path:
    plt.figure(figsize=(10, 5))
    plt.imshow(mars_map, cmap='hot', extent=(0, nc*10.0174, 0, nr*10.0174))
    plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'bo-')
    plt.plot([p[0] for p in path], [p[1] for p in path], 'r.-')
    plt.title(f'Ruta desde {start_point} hasta {end_point}. Distancia: {distance:.2f} metros')
    plt.xlabel('Coordenada X (metros)')
    plt.ylabel('Coordenada Y (metros)')
    plt.colorbar(label='Altura (metros)')
    plt.grid(True)
    plt.show()

fig.show()

#  BUSQUEDA A* ----- DISTANCIA MAYOR A 10000
#------------------------------------------------------------------------------------------------------------------
#   Height map pre-processing 
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

#Esta función convierte coordenadas (en metros) a índices de matriz (filas y columnas)
#basadas en el tamaño de píxel y las dimensiones del mapa de alturas.
def xy_to_rc(x, y, scale):
    r = nr - round(y / scale)
    c = round(x / scale)
    return r, c

# Definir función heurística para A* (distancia de Manhattan ponderada por la diferencia de altura)
#Esta función calcula la distancia de Manhattan entre los nodos a y b (la suma de las diferencias en las coordenadas x y y),
#y también tiene en cuenta la diferencia de altura entre los nodos.
#Esto proporciona una estimación del costo adicional necesario para llegar desde el nodo actual hasta el nodo objetivo,
#lo que guía al algoritmo en la búsqueda de la ruta más corta.
def heuristic(a, b, mars_map, scale):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    dz = abs(mars_map[a[0], a[1]] - mars_map[b[0], b[1]])
    return dx + dy + dz

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

# Definir algoritmo A*
#Esta función implementa el algoritmo A* para encontrar la ruta más corta desde el punto de inicio hasta
#el punto objetivo en el mapa de alturas de Marte. Utiliza una cola de prioridad (frontier) para expandir los nodos,
#manteniendo un registro de los costos acumulados (cost_so_far) y los nodos padres (came_from).
#Calcula la prioridad de cada nodo en función de la suma del costo acumulado y la heurística.
#Una vez que se alcanza el objetivo, reconstruye el camino desde el punto de inicio hasta el punto objetivo.
def astar(start, goal, mars_map, scale):
    start = xy_to_rc(*start, scale)
    goal = xy_to_rc(*goal, scale)

    frontier = [(0, start)]
    came_from = {}
    cost_so_far = {start: 0}

    while frontier:
        current_cost, current_node = heapq.heappop(frontier)

        if current_node == goal:
            break

        for next_node in get_neighbors(*current_node, mars_map, scale):
            new_cost = cost_so_far[current_node] + 1
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost + heuristic(goal, next_node, mars_map, scale)
                heapq.heappush(frontier, (priority, next_node))
                came_from[next_node] = current_node

    # Reconstruir el camino si existe
    #Esta sección del código comienza desde el nodo objetivo y sigue retrocediendo a lo largo de la ruta,
    #agregando cada nodo al camino hasta que llega al nodo de inicio.
    #Luego invertimos el camino para que esté en el orden correcto (desde el punto de inicio hasta el punto objetivo).
    #Finalmente, convertimos los índices de matriz de la ruta en coordenadas (x, y) para que sea más fácil de entender y visualizar.
    if goal in came_from:
        path = [goal]
        while path[-1] != start:
            path.append(came_from[path[-1]])
        path = [(rc[1] * scale, (nr - rc[0]) * scale) for rc in reversed(path)]
        distance = cost_so_far[goal]
    else:
        print("No se pudo encontrar una ruta válida desde el punto de inicio hasta el punto objetivo.")
        path = []
        distance = None

    return path, distance

# Coordenadas de inicio y fin
start_point = (6000, 1250)
end_point = (1000, 17000)

# Calcular ruta utilizando A*
path, distance = astar(start_point, end_point, mars_map, 10.0174)

# Mostrar el mapa de alturas y la ruta si existe
if path:
    plt.figure(figsize=(10, 5))
    plt.imshow(mars_map, cmap='hot', extent=(0, nc*10.0174, 0, nr*10.0174))
    plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'bo-')
    plt.plot([p[0] for p in path], [p[1] for p in path], 'r.-')
    plt.title(f'Ruta desde {start_point} hasta {end_point}. Distancia: {distance:.2f} metros')
    plt.xlabel('Coordenada X (metros)')
    plt.ylabel('Coordenada Y (metros)')
    plt.colorbar(label='Altura (metros)')
    plt.grid(True)
    plt.show()

fig.show()
