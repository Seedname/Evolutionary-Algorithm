@ -2,36 +2,54 @@ import numpy as np
import cv2
import random
import time
import math
# import multiprocessing

def generate_regular_polygon(num_sides, radius, center, rotation_angle_deg):
    vertices = []
    angle_increment = 2 * math.pi / num_sides
    rotation_matrix = cv2.getRotationMatrix2D(center=center, angle=rotation_angle_deg, scale=1.0)
    
    for i in range(num_sides):
        x = radius * math.cos(i * angle_increment)
        y = radius * math.sin(i * angle_increment)
        
        # Apply rotation and translation
        transformed_point = np.dot(rotation_matrix, np.array([x, y, 1]))
        vertices.append(transformed_point[:2] + center)
    
    return np.array(vertices, dtype=np.int32)

class Shape:
    def __init__(self, w, h, points=None):
        # self.w1 = random.randint(5, w // 2)
        # self.h1 = random.randint(5, h // 2)
    def __init__(self, w, h, generate=False):
        if generate:
            self.x = random.randint(0, w)
            self.y = random.randint(0, h)
            self.sides = random.randint(3, 5)
            self.radius = random.randint(1, w//2)
            self.angle = random.randint(0, 360)
            self.shape = generate_regular_polygon(self.sides, self.radius, (self.x, self.y), self.angle)

        # self.x1 = random.randint(0, w-self.w1)
        # self.y1 = random.randint(0, h-self.h1)
        if points:
            self.points = [[random.randint(0, w), random.randint(0, h)] for _ in range(points)]
            # self.points = np.random.randint(0, high=(w, h), size=(points, 2))

        self.w = w
        self.h = h

        self.mutation_rate = 25
        self.mutation_rate = 150

    def get_fitness(self, current_image, target_image):

        # Apply shape to image
        mask = np.zeros(current_image.shape[:2], dtype="uint8")
        cv2.fillPoly(mask, pts=[np.array(self.points)], color=255)
        cv2.fillPoly(mask, pts=[self.shape], color=255)
        # cv2.rectangle(mask, (self.x1, self.y1), (self.x1+self.w1, self.y1+self.h1), 255, -1)

        color = tuple(map(int, cv2.mean(target_image, mask)[:3]))
        # cv2.rectangle(current_image, (self.x1, self.y1), (self.x1+self.w1, self.y1+self.h1), color, -1)
        cv2.fillPoly(current_image, pts=[np.array(self.points)], color=color)
        cv2.fillPoly(current_image, pts=[self.shape], color=color)

        # find fitness with color difference
        color_difference = np.sum(np.abs(target_image - current_image))
        color_difference = np.sum(np.square(target_image - current_image))
 
        # store the image
        self.image = current_image
@ -43,16 +61,21 @@ class Shape:

        for _ in range(num_children):
            child = Shape(self.w, self.h)
            child.points = [[self.points[i][0] + random.randint(-10, 10) *  self.mutation_rate, self.points[i][1] + random.randint(-10, 10) * self.mutation_rate] for i in range(len(self.points))]
            # child.x1 = self.x1 + random.randint(-10, 10) * self.mutation_rate
            # child.y1 = self.y1 + random.randint(-10, 10) * self.mutation_rate

            # child.w1 = self.w1 + random.randint(-10, 10) * self.mutation_rate
            # child.h1 = self.h1 + random.randint(-10, 10) * self.mutation_rate
            child.sides = self.sides
            child.x = self.x + random.randint(-10, 10) * self.mutation_rate
            child.y = self.x + random.randint(-10, 10) * self.mutation_rate
            child.radius = self.x + random.randint(-10, 10) * self.mutation_rate
            child.angle = self.x + random.randint(-10, 10) * self.mutation_rate

            child.shape = generate_regular_polygon(child.sides, child.radius, (child.x, self.y), child.angle)
            
            # mutation = np.random.randint(-10, 11, size=(len(self.points), 2)) * self.mutation_rate
            # child.points = np.clip(self.points + mutation, 0, [self.w, self.h])
            children.append(child)

        return children
        return np.array(children)


class Evolution:
    def __init__(self, current_image, target_image, population:int):
@ -61,29 +84,54 @@ class Evolution:

        h, w = current_image.shape[:2]

        self.population = [Shape(w, h, random.randint(3, 5)) for _ in range(population)]
        self.population = np.array([Shape(w, h, True) for _ in range(population)])
        
    def evolve(self, generations):
        for generation in range(generations):
            # get fitness for all members of population
            fitness = [shape.get_fitness(self.current_image.copy(), self.target_image.copy()) for shape in self.population]
        for _ in range(generations):
            # Get fitness for all members of the population
            fitness = np.array([shape.get_fitness(self.current_image.copy(), self.target_image.copy()) for shape in self.population])
            

            # Get the indices that would sort the fitness array
            sorted_indices = np.argsort(fitness)

            # Rearrange the population list based on sorted_indices
            culled_population = [self.population[sorted_indices[i]] for i in range(len(sorted_indices)) if sorted_indices[i] < len(fitness) // 2]

            # # Calculate the median fitness
            # median = np.median(fitness)
            # # print(median)
            # # print(fitness)
            # # Create a boolean mask to filter bad members
            # mask = fitness > median

            # # Apply the mask to the population to get the culled population
            # culled_population = self.population[mask]

            # kill off bad members
            median = np.median(fitness)
            culled_population = [self.population[i] for i in range(len(self.population)) if fitness[i] >= median]
            # print(culled_population.shape)

            # Have the surviving members reproduce and mutate
            self.population = []
            self.population = np.array(culled_population)
            # self.population = culled_population

            for shape in culled_population:
                self.population.extend(shape.reproduce(3))
                self.population = np.concatenate((self.population, shape.reproduce(1)))

            # print(self.population.shape)

        fitness = [shape.get_fitness(self.current_image.copy(), self.target_image.copy()) for shape in self.population]
        best_fit = fitness.index(max(fitness))
        return self.population[best_fit].image
        best_fitness = min(fitness)
        best_fit = fitness.index(best_fitness)
        return [best_fitness, self.population[best_fit].image]

# def parallel_evolve(args):
#     return args[0].evolve(args[1])

if __name__ == "__main__":
    scale = 16
    shapes = 20000
    shapes = 10000
    shape_counter = 0

    startTime = time.time()

    image = cv2.imread('image.jpg')
@ -93,15 +141,31 @@ if __name__ == "__main__":
    average_color = tuple(map(int, cv2.mean(image)))[:3]
    base_image = np.full(shape=image.shape, fill_value=average_color, dtype=np.uint8)

    # pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    population_size = 50
    generations_per_step = 5

    last_fitness = float('inf')

    for i in range(1, shapes+1):
        object = Evolution(base_image, image, 20)
        base_image = object.evolve(3)
        print(i)
        if i % 1000 == 0:
            print(i)
        object = Evolution(base_image, image, population_size)
        current_fitness, current_image = object.evolve(generations_per_step)
        if current_fitness < last_fitness:
            base_image = current_image
        # base_image = pool.map(parallel_evolve, [(object, generations_per_step)])[0]

            shape_counter += 1
            # print(i)
            print(shape_counter)
            

        if i % 100 == 0:
            if i > 0:
                delay = time.time() - startTime
                print(f"{delay * ((shapes - i)/ 1000)} more seconds")
                elapsed_time = time.time() - startTime
                remaining_time = (elapsed_time / i) * (shapes - i)
                print(f"Time Remaining: {remaining_time}")
        last_fitness = current_fitness

    cv2.imwrite('finished.jpg', base_image)
    cv2.imshow('Final result', base_image)
