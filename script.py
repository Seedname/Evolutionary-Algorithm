import numpy as np
import cv2
import random

class Shape:
    def __init__(self, w, h):
        self.x1 = random.randint(0, w)
        self.x2 = random.randint(0, w)
        self.y1 = random.randint(0, h)
        self.y2 = random.randint(0, h)

        self.w = w
        self.h = h

        self.mutation_rate = 10

    def get_fitness(self, current_image, target_image):

        # Apply shape to image
        mask = np.zeros(current_image.shape[:2], dtype="uint8")
        cv2.rectangle(mask, (self.x1, self.y1), (self.x2, self.y2), 255, -1)

        color = tuple(map(int, cv2.mean(target_image, mask)[:3]))
        cv2.rectangle(current_image, (self.x1, self.y1), (self.x2, self.y2), color, -1)

        # find fitness with color difference
        color_difference = np.sum(np.abs(target_image - current_image))
 
        # store the image
        self.image = current_image

        return color_difference
    
    def reproduce(self, num_children):
        children = []

        for _ in range(num_children):
            child = Shape(self.w, self.h)
            child.x1 = self.x1 + random.randint(-10, 10) * self.mutation_rate
            child.x2 = self.x2 + random.randint(-10, 10) * self.mutation_rate
            child.y1 = self.y1 + random.randint(-10, 10) * self.mutation_rate
            child.y2 = self.y2 + random.randint(-10, 10) * self.mutation_rate

            children.append(child)

        return children

class Evolution:
    def __init__(self, current_image, target_image, population:int):
        self.current_image = current_image
        self.target_image = target_image

        h, w = current_image.shape[:2]

        self.population = [Shape(w, h) for _ in range(population)]
        
    def evolve(self, generations):
        for generation in range(generations):
            # get fitness for all members of population
            fitness = [shape.get_fitness(self.current_image.copy(), self.target_image.copy()) for shape in self.population]

            # kill off bad members
            median = np.median(fitness)
            culled_population = [self.population[i] for i in range(len(self.population)) if fitness[i] >= median]

            # Have the surviving members reproduce and mutate
            self.population = []
            for shape in culled_population:
                self.population.extend(shape.reproduce(2))

        fitness = [shape.get_fitness(self.current_image.copy(), self.target_image.copy()) for shape in self.population]
        best_fit = fitness.index(max(fitness))
        return self.population[best_fit].image

    
image = cv2.imread('image.jpg')
average_color = tuple(map(int, cv2.mean(image)))[:3]
base_image = np.full(shape=image.shape, fill_value=average_color, dtype=np.uint8)

for i in range(500):
    object = Evolution(base_image, image, 20)
    base_image = object.evolve(5)
    print(i)

cv2.imshow('Final result', base_image)
cv2.waitKey(0)
