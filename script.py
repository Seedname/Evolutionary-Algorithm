import numpy as np
import cv2
import random
import time


class Shape:
    def __init__(self, w, h, points=None):
        # self.w1 = random.randint(5, w // 2)
        # self.h1 = random.randint(5, h // 2)

        # self.x1 = random.randint(0, w-self.w1)
        # self.y1 = random.randint(0, h-self.h1)
        if points:
            self.points = [[random.randint(0, w), random.randint(0, h)] for _ in range(points)]

        self.w = w
        self.h = h

        self.mutation_rate = 25

    def get_fitness(self, current_image, target_image):

        # Apply shape to image
        mask = np.zeros(current_image.shape[:2], dtype="uint8")
        cv2.fillPoly(mask, pts=[np.array(self.points)], color=255)
        # cv2.rectangle(mask, (self.x1, self.y1), (self.x1+self.w1, self.y1+self.h1), 255, -1)

        color = tuple(map(int, cv2.mean(target_image, mask)[:3]))
        # cv2.rectangle(current_image, (self.x1, self.y1), (self.x1+self.w1, self.y1+self.h1), color, -1)
        cv2.fillPoly(current_image, pts=[np.array(self.points)], color=color)

        # find fitness with color difference
        color_difference = np.sum(np.abs(target_image - current_image))
 
        # store the image
        self.image = current_image

        return color_difference
    
    def reproduce(self, num_children):
        children = []

        for _ in range(num_children):
            child = Shape(self.w, self.h)
            child.points = [[self.points[i][0] + random.randint(-10, 10) *  self.mutation_rate, self.points[i][1] + random.randint(-10, 10) * self.mutation_rate] for i in range(len(self.points))]
            # child.x1 = self.x1 + random.randint(-10, 10) * self.mutation_rate
            # child.y1 = self.y1 + random.randint(-10, 10) * self.mutation_rate

            # child.w1 = self.w1 + random.randint(-10, 10) * self.mutation_rate
            # child.h1 = self.h1 + random.randint(-10, 10) * self.mutation_rate

            children.append(child)

        return children

class Evolution:
    def __init__(self, current_image, target_image, population:int):
        self.current_image = current_image
        self.target_image = target_image

        h, w = current_image.shape[:2]

        self.population = [Shape(w, h, random.randint(3, 5)) for _ in range(population)]
        
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
                self.population.extend(shape.reproduce(3))

        fitness = [shape.get_fitness(self.current_image.copy(), self.target_image.copy()) for shape in self.population]
        best_fit = fitness.index(max(fitness))
        return self.population[best_fit].image

if __name__ == "__main__":
    scale = 16
    shapes = 20000
    startTime = time.time()

    image = cv2.imread('image.jpg')

    image = cv2.resize(image, (int(image.shape[1]/scale), int(image.shape[0]/scale)))

    average_color = tuple(map(int, cv2.mean(image)))[:3]
    base_image = np.full(shape=image.shape, fill_value=average_color, dtype=np.uint8)

    for i in range(1, shapes+1):
        object = Evolution(base_image, image, 20)
        base_image = object.evolve(3)
        print(i)
        if i % 1000 == 0:
            print(i)
            if i > 0:
                delay = time.time() - startTime
                print(f"{delay * ((shapes - i)/ 1000)} more seconds")

    cv2.imwrite('finished.jpg', base_image)
    cv2.imshow('Final result', base_image)
    cv2.waitKey(0)