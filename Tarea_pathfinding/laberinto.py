"""
Importamos todos los modulos para utilizarlos en los algoritmos de busqueda de un camino para 
salir de un laberinto.
networkx y matplotlib.pyplot son para crear y dibujar el grafo (laberinto)
random es para generar los laberintos aleatoriamente
deque se utiliza para crear una cola para BFS
heapq se utiliza en dijistra y en A* para asi pocer ver cual tiene mas prioridad
timeit y tracemalloc se utilizan para ver la eficiencia empirica de cada Busqueda de este algoritmo
"""
import networkx as nx
import matplotlib.pyplot as plt
import random
from collections import deque
import tracemalloc
import timeit
import heapq


class Laberinto:
    """
    Función init donde estan los principales aspectos de un laberinto
    """
    def __init__(self, filas, columnas,):
        self.filas = filas
        self.columnas = columnas
        self.grafo = nx.Graph()
        self.pos = {}
        self.entrada = (0, 0)# el nodo donde empezamos estara en coordenadas (0,0)
        self.salida = (filas - 1, columnas - 1)# el fin del laberinto sera (filas - 1, columnas - 1) que es el nodo derecha abajo
    
    """Crea nodos y aristas para el grafo del laberinto."""
    def crear_nodos(self):
        for i in range(self.filas):
            for j in range(self.columnas):
                nodo = (i, j)
                self.grafo.add_node(nodo)
                self.pos[nodo] = (j, -i)
    def crear_aristas(self):
        aristas = []
        for i in range(self.filas):
            for j in range(self.columnas):
                if j < self.columnas - 1:#tenemos que ver que para cada j menor a n-1(columnas) podemos movernos a la derecha
                    aristas.append(((i, j), (i, j + 1)))#añadimos una tupla con los dos nodos que esta conectando la arista
                if i < self.filas - 1:#Aqui lo que hacemos es ver que la i cuando es menor a m-1(filas) podemos conectar con el nodo de abajo
                    aristas.append(((i, j), (i + 1, j)))#añadimos una tupla con los dos nodos que esta conectando la arista


        # devolvemos la lista de aristas
        return aristas





    '''
    Generar el laberinto usando el algoritmo de Prim
    Haciendo un MST para que no haya ciclos en el laberinto'''
    def generar_laberinto(self):
    
        edges = self.crear_aristas()#coje las aristas anteriores
        random.shuffle(edges)  # Mezclamos las aristas para aleatorizar el proceso
        visited = set([self.entrada])# la entrada la marcamos como ya  visitado
        maze_edges = []

        while len(visited) < self.filas * self.columnas:#mientras no hayamos visitado todos los nodos
            for edge in edges:
                u, v = edge # los nodos de que conecta cada arista
                if u in visited and v not in visited:# si u esta visitado y v no
                    maze_edges.append(edge)
                    visited.add(v)#añadir a la lista y como visitado
                    break

        self.grafo.add_edges_from(maze_edges)#añadir al grafo

    
        


    def resolver_laberinto_BFS(self):
        """Resuelve el laberinto utilizando BFS y devuelve el camino."""
        cola = deque([self.entrada])
        visited = {self.entrada: None}

        while cola: # Mientras cola este llena
            current_node = cola.popleft()#llamamos a current_node el primer nodo de la lista queue
            '''
            Este bucle for recorre todas las vecindades (neighbors) para cada neighbor:
            luego vemos si NO esta vecindad(hijo de current_node) esta en el dictionario visited
            Entonces si no esta en visited añadimos a visited un dictionario:
            ((el nodo del vecino), (el nodo que estamos))
            El nodo del vecino lo añadimos a la lista cola
            
            '''
            for vecino in self.grafo.neighbors(current_node):
                if vecino not in visited:
                    visited[vecino] = current_node
                    cola.append(vecino)

            if current_node == self.salida:
                camino = []
                while current_node is not None:# para crear el camino haciendo backtracing hasta llegar a NONE
                    camino.append(current_node)
                    current_node = visited[current_node]
                return camino[::-1]# Invertir el camino para que sea de entrada a salida
    
    """Resuelve el laberinto utilizando DFS y devuelve el camino."""
    def resolver_laberinto_DFS(self):
        
        stack = [self.entrada]# crear un stack con la entrada
        visitados = {self.entrada: None}
        while stack:
            nodo_actual = stack.pop()# LIFO cojemos el ULTIMO nodo
            if nodo_actual == self.salida:# SI LLEAGAMOS A SALIDA BACKTRACING DEL DICIONARIO
                camino = []
                while nodo_actual is not None:
                    camino.append(nodo_actual)
                    nodo_actual = visitados[nodo_actual]
                return camino[::-1]

            for vecino in self.grafo.neighbors(nodo_actual):# luego añaimos los vecinos al stack 
                if vecino not in visitados:
                    stack.append(vecino)
                    visitados[vecino] = nodo_actual
    """Resuelve el laberinto utilizando Dijkstra y devuelve el camino."""
    def resolver_laberinto_Dijkstra(self):
       
        # Inicialización de distancias
        distancias = {nodo: float('inf') for nodo in self.grafo.nodes()}#TODAS SON INFINITAS 
        distancias[self.entrada] = 0  # La distancia a la entrada es 0
        
        # Cola de prioridad (min-heap)
        cola = [(0, self.entrada)]  # (distancia, nodo)
        
        # Diccionario para almacenar el predecesor de cada nodo
        predecesores = {self.entrada: None}
        
        while cola:# Mientras que la cola este llena
            # Extraemos el nodo_actual con la distancia más pequeña
            distancia, nodo_actual = heapq.heappop(cola)
            
            # Si llegamos a la salida, reconstruimos el camino
            if nodo_actual == self.salida:
                camino = []
                while nodo_actual is not None:
                    camino.append(nodo_actual)
                    nodo_actual = predecesores[nodo_actual]
                return camino[::-1]  # Invertimos el camino al reconstruirlo
            
            # Exploramos los vecinos
            for vecino in self.grafo.neighbors(nodo_actual):
                nueva_distancia = distancia + 1  # El peso de cada arista es 1
                if nueva_distancia < distancias[vecino]:
                    distancias[vecino] = nueva_distancia#añadir a distancia (vecino:nueva_distancia)
                    predecesores[vecino] = nodo_actual# añadir a predecesores (vecino:nodo_actual)
                    heapq.heappush(cola, (nueva_distancia, vecino))# añadimos a la cola la [(nueva_distancia , vecino)]
        
        return None  # Si no se encuentra un camino

    """Resuelve el laberinto utilizando un A* más eficiente."""
    def resolver_laberinto_A_star(self):
  
        open_set = []  # Cola de prioridad
        heapq.heappush(open_set, (0, self.entrada))  # (f_score, nodo actual)
        came_from = {}  # Para reconstruir el camino
        g_score = {self.entrada: 0}  # Costos reales desde el inicio
        closed_set = set()  # Nodos ya procesados
        """Heurística mejorada combinando Manhattan y una penalización por desviación."""
        def heuristica_optimizada(nodo):
            #nodo (u,v) donde u=nodo[0] y v=nodo[1]
            """
            Distancia manhatan entre el nodo y la salida
            """
            dx = abs(nodo[0] - self.salida[0])
            dy = abs(nodo[1] - self.salida[1])

            return dx + dy + 0.1 * abs(dx - dy)  # Penaliza caminos menos directos con 0.1 * abs(dx - dy)

        while open_set:
            # Extraemos el nodo con el menor f_score
            _, current = heapq.heappop(open_set)

            if current == self.salida:
                # Reconstruimos el camino si llegamos a la salida
                camino = []
                while current in came_from:
                    camino.append(current)
                    current = came_from[current]
                camino.append(self.entrada)
                return camino[::-1]

            closed_set.add(current)  # Marcamos el nodo como procesado

            for vecino in self.grafo.neighbors(current):
                if vecino in closed_set:
                    continue

                tentative_g_score = g_score[current] + 1  # Peso de la arista es 1

                # Si encontramos un mejor camino hacia el vecino
                if vecino not in g_score or tentative_g_score < g_score[vecino]:
                    g_score[vecino] = tentative_g_score
                    f_score = tentative_g_score + heuristica_optimizada(vecino)#f_score distancia tentativa que es la misma que la dijistra de antes mas la heristica
                    heapq.heappush(open_set, (f_score, vecino))
                    came_from[vecino] = current

        return None  # Si no se encuentra un camino


    def dibujar_laberinto(self, camino=None):
        """Dibuja el laberinto y opcionalmente su solución."""
        plt.figure(figsize=(8, 8))
        # Asegurarse de que las posiciones estén completas antes de dibujar
        if not self.pos:
            self.crear_nodos()  # Asegurarse de que las posiciones de los nodos estén listas
        nx.draw(self.grafo, self.pos, with_labels=False, node_size=100, node_color='skyblue', font_size=10)
        nx.draw_networkx_nodes(self.grafo, self.pos, nodelist=[self.entrada], node_color='red')  # Entrada en rojo
        nx.draw_networkx_nodes(self.grafo, self.pos, nodelist=[self.salida], node_color='green')  # Salida en verde

        if camino:
            aristas_camino = [(camino[i], camino[i + 1]) for i in range(len(camino) - 1)]
            nx.draw_networkx_edges(self.grafo, self.pos, edgelist=aristas_camino, width=4, edge_color='blue')  # Camino en azul

        plt.show()


# Ejecución principal
if __name__ == "__main__":
    filas = int(input('Número de filas: '))
    columnas = int(input('Número de columnas: '))
    laberinto = Laberinto(filas, columnas)
    print("1. Resolver por DFS")
    print("2. Resolver por BFS")
    print("3. Resolver por DIJISTRA")
    print("4. Resolver por A*")
    print("5. Dar tiempo y memoria de cada programa")#analisis
    print("6. Todas los resultados")
    opcion = input("Seleccione una opción: ")

    if opcion == "1":
        laberinto.generar_laberinto()
        camino = laberinto.resolver_laberinto_DFS()
        laberinto.dibujar_laberinto(camino)

    elif opcion == "2":
        laberinto.generar_laberinto()
        camino = laberinto.resolver_laberinto_BFS()
        laberinto.dibujar_laberinto(camino)

    elif opcion == "3":
        laberinto.generar_laberinto()
        camino = laberinto.resolver_laberinto_Dijkstra()
        laberinto.dibujar_laberinto(camino)  
    elif opcion == "4":
        laberinto.generar_laberinto()
        camino = laberinto.resolver_laberinto_A_star()
        if camino:
            laberinto.dibujar_laberinto(camino)
        else:
            print("No se encontró un camino.")

    elif opcion=="5":
        laberinto.generar_laberinto()

        # Medición de tiempo y memoria para A* utilizando timeit
        tracemalloc.start()
        sal = timeit.timeit(lambda: laberinto.resolver_laberinto_A_star(), number=100)  # Ejecuta A* 100 veces
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Medición de tiempo y memoria para Dijkstra utilizando timeit
        tracemalloc.start()
        fal = timeit.timeit(lambda: laberinto.resolver_laberinto_Dijkstra(), number=100)  # Ejecuta Dijkstra 100 veces
        currents, peaks = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Medición de tiempo y memoria para BFS utilizando timeit
        tracemalloc.start()
        cal = timeit.timeit(lambda: laberinto.resolver_laberinto_BFS(), number=100)  # Ejecuta BFS 100 veces
        currentes, peakes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Medición de tiempo y memoria para DFS utilizando timeit
        tracemalloc.start()
        dal = timeit.timeit(lambda: laberinto.resolver_laberinto_DFS(), number=100)  # Ejecuta DFS 100 veces
        currenta, peaka = tracemalloc.get_traced_memory()  # Obtiene la memoria utilizada
        tracemalloc.stop()  # Detiene el rastreo de la memoria

        # Imprimir resultados de cada algoritmo
        print(f"Tiempo de ejecución DFS (100 repeticiones): {dal:.6f} segundos")
        print(f"Memoria usada por DFS: {currenta / 1024:.2f} KB (actual), {peaka / 1024:.2f} KB (pico)")

        print(f"Tiempo de ejecución BFS (100 repeticiones): {cal:.6f} segundos")
        print(f"Memoria usada por BFS: {currentes / 1024:.2f} KB (actual), {peakes / 1024:.2f} KB (pico)")

        print(f"Tiempo de ejecución Dijkstra (100 repeticiones): {fal:.6f} segundos")
        print(f"Memoria usada por Dijkstra: {currents / 1024:.2f} KB (actual), {peaks / 1024:.2f} KB (pico)")

        print(f"Tiempo de ejecución A* (100 repeticiones): {sal:.6f} segundos")
        print(f"Memoria usada por A*: {current / 1024:.2f} KB (actual), {peak / 1024:.2f} KB (pico)")
    elif opcion == "6":
        laberinto.generar_laberinto()
        camino = laberinto.resolver_laberinto_DFS()
        caminos = laberinto.resolver_laberinto_BFS()
        caminot = laberinto.resolver_laberinto_A_star()
        caminod = laberinto.resolver_laberinto_Dijkstra()
        
        laberinto.dibujar_laberinto(camino)
        laberinto.dibujar_laberinto(caminos)
        laberinto.dibujar_laberinto(caminot)
        laberinto.dibujar_laberinto(caminod)
        