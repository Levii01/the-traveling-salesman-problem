from math import degrees, acos
from turtle import Turtle
import matplotlib.pyplot as plt
import numpy as np
import operator
import pandas as pd
import random


# This will be class to handle cities data
class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        x_distance = abs(self.x - city.x)
        y_distance = abs(self.y - city.y)
        return np.sqrt((x_distance ** 2) + (y_distance ** 2))

    def angle(self, city):
        try:
            result = degrees(acos((city.x - self.x) / (self.distance(city))))
        except:
            result = 0
        if self.y > city.y:
            result = 360 - result
        return result

    def __repr__(self):
        return f"({str(self.x)},{str(self.y)})"


# Class to show how root is good, it help to find the best routes match
class RouteMatcher:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def route_distance(self):
        if self.distance == 0:
            for i in range(len(self.route)):
                from_city = self.route[i]
                to_city = self.route[i + 1] if (i + 1 < len(self.route)) else self.route[0]
                self.distance += from_city.distance(to_city)
        return self.distance

    def match(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.route_distance())
        return self.fitness


# take random city to create route
def create_route(cities):
    return random.sample(cities, len(cities))


# create generation of routes to all cities
def initial_possible_routes(generatio_size, cities):
    possible_routes = []
    for i in range(generatio_size):
        possible_routes.append(create_route(cities))
    return possible_routes


# returns sorted list of all possible routes
def sorted_rank_routes(possible_routes):
    matched_results = {}
    for i in range(0, len(possible_routes)):
        matched_results[i] = RouteMatcher(possible_routes[i]).match()
    return sorted(matched_results.items(), key=operator.itemgetter(1), reverse=True)


def selection(possible_routes_ranked, elite_size):
    selection_results = []
    df = pd.DataFrame(np.array(possible_routes_ranked), columns=["Index", "Match"])
    df['cum_sum'] = df.Match.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Match.sum()

    for i in range(elite_size):
        selection_results.append(possible_routes_ranked[i][0])

    for i in range(len(possible_routes_ranked) - elite_size):
        pick = 100 * random.random()
        for i in range(len(possible_routes_ranked)):
            if pick <= df.iat[i, 3]:
                selection_results.append(possible_routes_ranked[i][0])
                break
    return selection_results


# it returns collection of routes, used to create next generation of routes
def mating_pool(possible_routes, selection_results):
    matingpool = []
    for i in range(len(selection_results)):
        id = selection_results[i]
        matingpool.append(possible_routes[id])
    return matingpool


# creaate next generation with mating pool (parents), method return new child
def crossover(parent1, parent2):
    # we need child_p1, child_p2 as array[]
    child_p1 = []
    gene_a = int(random.random() * len(parent1))
    gene_b = int(random.random() * len(parent1))
    start_gene = min(gene_a, gene_b)
    end_gene = max(gene_a, gene_b)

    for i in range(start_gene, end_gene):
        child_p1.append(parent1[i])
    child_p2 = [item for item in parent2 if item not in child_p1]

    return child_p1 + child_p2

# method to create offspring routes collection from crossover method
# and also we used elitism to save the best routes collection (the best solutions)
def generate_offspring_generation(matingpool, elite_size):
    children = []
    length = len(matingpool) - elite_size
    pool = random.sample(matingpool, len(matingpool))

    for i in range(elite_size):
        children.append(matingpool[i])

    for i in range(length):
        child = crossover(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children

# method to swap city with low probability, we set mutation_rate on the begin of our alghoritm
def swap_cities(individual, mutation_rate):
    for swapped in range(len(individual)):
        if random.random() < mutation_rate:
            swap_with = int(random.random() * len(individual))
            individual[swapped], individual[swap_with] = individual[swap_with], individual[swapped]
    return individual


def mutate_generation(possible_routes, mutation_rate):
    mutated_generation = []

    for id in range(len(possible_routes)):
        mutated_id = swap_cities(possible_routes[id], mutation_rate)
        mutated_generation.append(mutated_id)
    return mutated_generation


def next_generation(current_gen, elite_size, mutation_rate):
    possible_routes_ranked = sorted_rank_routes(current_gen)
    selection_results = selection(possible_routes_ranked, elite_size)
    matingpool = mating_pool(current_gen, selection_results)
    children = generate_offspring_generation(matingpool, elite_size)
    return mutate_generation(children, mutation_rate)


def city_coordinates_list(city):
    return [city.x, city.y]


def show_coordinates_graph(cities):
    result = map(city_coordinates_list, cities)
    plt.title('Graph with Cities for salesman')
    plt.scatter(*zip(*list(result)))
    plt.show()


def show_distance_to_generation_plot(progress):
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()


def draw_salesman_route(best_routes):
    ang = 0
    travel_path = Turtle()
    travel_path.hideturtle()
    travel_path.screen.bgcolor("yellow")
    travel_path.color("blue")

    for i in range(len(best_routes)):
        travel_path.left(-ang)
        d = best_routes[i - 1].distance(best_routes[i])
        ang = best_routes[i - 1].angle(best_routes[i])

        travel_path.left(ang)
        travel_path.forward(d)
        travel_path.circle(2)
    travel_path.screen.exitonclick()


def tsp_algorithm(cities, generatio_size, elite_size, mutation_rate, generations):
    show_coordinates_graph(cities)
    possible_routes = initial_possible_routes(generatio_size, cities)
    progress = [1 / sorted_rank_routes(possible_routes)[0][1]]

    print(f"Initial distance: {str(1 / sorted_rank_routes(possible_routes)[0][1])}")

    for i in range(generations):
        possible_routes = next_generation(possible_routes, elite_size, mutation_rate)
        progress.append(1 / sorted_rank_routes(possible_routes)[0][1])

    print(f"Final distance: {str(1 / sorted_rank_routes(possible_routes)[0][1])}")
    show_distance_to_generation_plot(progress)
    best_route_id = sorted_rank_routes(possible_routes)[0][0]
    draw_salesman_route(possible_routes[best_route_id])


city_list = []
for i in range(30):
    city_list.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))

tsp_algorithm(cities=city_list, generatio_size=100, elite_size=20, mutation_rate=0.01, generations=500)
