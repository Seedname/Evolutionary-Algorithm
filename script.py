import numpy as np
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
    def __init__(self, w, h, generate=False):
        if generate:
            self.x = random.randint(0, w)
            self.y = random.randint(0, h)
            self.sides = random.randint(3, 5)
            self.radius = random.randint(1, w//2)
            self.angle = random.randint(0, 360)
            self.shape = generate_regular_polygon(self.sides, self.radius, (self.x, self.y), self.angle)
            self.color = 0
            # self.points = np.random.randint(0, high=(w, h), size=(points, 2))

        self.w = w
        self.h = h

        self.mutation_rate = 150

    def get_fitness(self, current_image, target_image):

        # Apply shape to image
        mask = np.zeros(current_image.shape[:2], dtype="uint8")
        cv2.fillPoly(mask, pts=[self.shape], color=255)
        # cv2.rectangle(mask, (self.x1, self.y1), (self.x1+self.w1, self.y1+self.h1), 255, -1)

        self.color = tuple(map(int, cv2.mean(target_image, mask)[:3]))
        # cv2.rectangle(current_image, (self.x1, self.y1), (self.x1+self.w1, self.y1+self.h1), color, -1)
        cv2.fillPoly(current_image, pts=[self.shape], color=self.color)

        # find fitness with color difference
        color_difference = np.sum(np.square(target_image - current_image))
 
        # store the image
        self.image = current_image

        return color_difference
    
    def reproduce(self, num_children):
        children = []

        for _ in range(num_children):
            child = Shape(self.w, self.h)

            child.sides = self.sides
            child.x = self.x + random.randint(-10, 10) * self.mutation_rate
            child.y = self.x + random.randint(-10, 10) * self.mutation_rate
            child.radius = self.x + random.randint(-10, 10) * self.mutation_rate
            child.angle = self.x + random.randint(-10, 10) * self.mutation_rate

            child.shape = generate_regular_polygon(child.sides, child.radius, (child.x, self.y), child.angle)
            

            # mutation = np.random.randint(-10, 11, size=(len(self.points), 2)) * self.mutation_rate
            # child.points = np.clip(self.points + mutation, 0, [self.w, self.h])
            children.append(child)

        return np.array(children)


class Evolution:
    def __init__(self, current_image, target_image, population:int):
        self.current_image = current_image
        self.target_image = target_image

        h, w = current_image.shape[:2]

        self.population = np.array([Shape(w, h, True) for _ in range(population)])
        
    def evolve(self, generations):
        for _ in range(generations):
            # Get fitness for all members of the population
            fitness = np.array([shape.get_fitness(self.current_image.copy(), self.target_image.copy()) for shape in self.population])
            

            # Get the indices that would sort the fitness array
            sorted_indices = np.argsort(fitness)

            # Rearrange the population list based on sorted_indices
            culled_population = [self.population[sorted_indices[i]] for i in range(len(sorted_indices)) if sorted_indices[i] < len(fitness) // 2]

            # Have the surviving members reproduce and mutate
            self.population = np.array(culled_population)
            # self.population = culled_population

            for shape in culled_population:
                self.population = np.concatenate((self.population, shape.reproduce(1)))


        fitness = [shape.get_fitness(self.current_image.copy(), self.target_image.copy()) for shape in self.population]
        best_fitness = min(fitness)
        best_fit = fitness.index(best_fitness)
        return [best_fitness, self.population[best_fit].image, self.population[best_fit].shape, self.population[best_fit].color]

# def parallel_evolve(args):
#     return args[0].evolve(args[1])

if __name__ == "__main__":
    scale = 4
    shapes = 10000
    shape_counter = 0

    startTime = time.time()

    image = cv2.imread('mit.png')
    final_image_shape = tuple(image.shape)

    video = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 5, final_image_shape[:-1])

    image = cv2.resize(image, (int(image.shape[1]/scale), int(image.shape[0]/scale)))
    cv2.imshow('Preview', image)
    cv2.waitKey(0)
    average_color = tuple(map(int, cv2.mean(image)))[:3]
    base_image = np.full(shape=image.shape, fill_value=average_color, dtype=np.uint8)
    final_image = np.full(shape=final_image_shape, fill_value=average_color, dtype=np.uint8)

    population_size = 50
    generations_per_step = 5

    last_fitness = float('inf')

    all_shapes = []


    for i in range(1, shapes+1):
        object = Evolution(base_image, image, population_size)
        current_fitness, current_image, best_fit_shape, best_fit_color = object.evolve(generations_per_step)
        if current_fitness < last_fitness:
            base_image = current_image
        # base_image = pool.map(parallel_evolve, [(object, generations_per_step)])[0]
            cv2.fillPoly(final_image, pts=[scale * best_fit_shape], color=best_fit_color)

            video.write(final_image)
            shape_counter += 1
            # print(i)
            print(shape_counter)
            

        if i % 100 == 0:
            if i > 0:
                elapsed_time = time.time() - startTime
                remaining_time = (elapsed_time / i) * (shapes - i)
                print(f"Time Remaining: {remaining_time}")
        last_fitness = current_fitness
    video.release()
    cv2.imwrite('finished.png', base_image)
    cv2.imwrite('final.png', final_image)
    cv2.imshow('Final result', final_image)
    cv2.waitKey(0)
    