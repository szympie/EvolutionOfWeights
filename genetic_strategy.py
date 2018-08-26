import random


class Individual:
    def __init__(self,
                 classifiers,
                 classifiers_weights,
                 class_weights):
        self.classifiers = classifiers
        self.classifiers_weights = classifiers_weights
        self.class_weights = class_weights
        self.score = 0.0

    def mutate_classifiers_weights(self):
        random_index = random.randint(0, len(self.classifiers_weights) - 1)
        new_value = random.random()
        self.classifiers_weights[random_index] = new_value

    def mutate_class_weights(self):
        random_index = random.randint(0, len(self.class_weights) - 1)
        new_value = random.random()
        self.class_weights[random_index] = new_value


def crossover_classifier_weights(first_individual, second_individual):
    crossover_point = random.randint(0,
                                     len(first_individual.classifiers_weights))
    crossed_weights = first_individual.classifiers_weights[0:crossover_point]
    crossed_weights.extend(
        second_individual.classifiers_weights[crossover_point:])
    child = Individual(None,
                       crossed_weights,
                       None)
    return child


def roullete_choice(scores):
    max = sum(scores)
    pick = random.uniform(0, max)
    current = 0
    for index, value in enumerate(scores):
        current += value
        if current > pick:
            return index
