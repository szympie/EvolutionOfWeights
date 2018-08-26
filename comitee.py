import random
import copy

import numpy as np

import genetic_strategy


class Comitee:
    def __init__(self, classifiers, classifiers_weights, class_weights):
        self.classifiers = classifiers
        self.classifiers_weights = classifiers_weights
        self.class_weights = class_weights

    def fit(self, X, y):
        for classifier in self.classifiers:
            classifier.fit(X, y)

    def predict(self, X):
        predictions = []
        for instance_index, instance in enumerate(X):
            instance = instance.reshape(1, -1)
            classes_scores = np.zeros(len(self.class_weights))
            for classifier_index, classifier in enumerate(self.classifiers):
                prediction = classifier.predict(instance)
                weighted_prediction = self.class_weights[int(prediction)] * \
                                      self.classifiers_weights[
                                          classifier_index]
                classes_scores[int(prediction)] += weighted_prediction
            predictions.append(classes_scores.argmax())
        return np.array(predictions)

    def score(self, X, y):
        predictions = self.predict(X)
        return sum(np.equal(predictions, y)) / len(X)

    def optimize_classifier_weights(self, X, y, number_of_individuals=20,
                                    number_of_epochs=100,
                                    elitism_coefficient=0.1):
        def update_individuals_scores(population, X, y):
            for individual in population:
                self.classifiers_weights = individual.classifiers_weights
                individual.score = self.score(X, y)

        def crossover(population, scores, number_of_childs):
            new_population = []
            for x in range(number_of_childs):
                parent1_index = genetic_strategy.roullete_choice(scores)
                parent2_index = genetic_strategy.roullete_choice(scores)
                child = genetic_strategy.crossover_classifier_weights(
                    population[parent1_index],
                    population[parent2_index])
                new_population.append(child)
            return new_population

        def mutate(population, mutation_probability):
            for individual in population:
                if random.random() < mutation_probability:
                    individual.mutate_classifiers_weights()

        def save_elite(population, elitism_coefficient):
            population = sorted(population, key=lambda x: x.score,
                                reverse=True)
            length_of_elite = int(len(population) * elitism_coefficient)
            elite = [copy.deepcopy(individual) for individual in
                     population[0:length_of_elite]]
            return elite

        individuals = [genetic_strategy.Individual(None,
                                                   [random.random() for _ in
                                                    range(len(
                                                        self.classifiers))],
                                                   None) for _ in
                       range(number_of_individuals)]
        for epoch in range(number_of_epochs):
            update_individuals_scores(individuals, X, y)
            individuals = sorted(individuals, key=lambda x: x.score,
                                 reverse=True)
            scores = [individual.score for individual in individuals]
            if (np.max(scores)) == 1.0:
                break
            print(create_description(epoch, scores))
            elite = save_elite(individuals, elitism_coefficient)
            individuals = crossover(individuals, scores,
                                    len(individuals) - len(elite))
            individuals.extend(elite)
            mutate(individuals, 0.01)


def create_description(epoch, scores):
    return f'Epoch: {epoch}, ' \
           f'Best score: {np.max(scores)}, ' \
           f'Mean score: {np.mean(scores)}, ' \
           f'Std score: {np.std(scores)}, '
