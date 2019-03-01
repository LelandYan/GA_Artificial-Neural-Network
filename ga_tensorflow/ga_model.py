# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2018/11/17 23:02'

import numpy as np
import pandas as pd
from ga_tensorflow.Classifier_model.svm_model import *
from ga_tensorflow.Classifier_model.RandomForest_model import *
from ga_tensorflow.Classifier_model.knn_model import *
from ga_tensorflow.Classifier_model.Quadratic_Discriminant_Analysis_model import *
from ga_tensorflow.Classifier_model.Multinomial_Native_bayes_Classifier import *
from ga_tensorflow.Classifier_model.logistic_regression_model import *
from ga_tensorflow.Classifier_model.Linaer_Discriminant_Analysis import *
from ga_tensorflow.Classifier_model.GaussianNB_model import *
from ga_tensorflow.Classifier_model.GBDT_model import *
from ga_tensorflow.Classifier_model.AdaBoost_model import *
# the path and name of file
# CSV_FILE_PATH = 'csv_result-ALL-AML_train.csv'
CSV_FILE_PATH = 'csv_result-colonTumor.csv'
# read the file
df = pd.read_csv(CSV_FILE_PATH)
shapes = df.values.shape
# the eigenvalue of file
input_data = df.values[:, 1:shapes[1] - 1]
# the result of file
result = df.values[:, shapes[1] - 1:shapes[1]]
# the length of eigenvalue
value_len = input_data.shape[1]
# the length of result
# pop_len = 20
pop_len = result.shape[0]
# DNA length
DNA_SIZE = value_len
# population size
POP_SIZE = pop_len
# mating probability(DNA crossover)
CROSS_RATE = 0.8
# mutation probability
MUTATION_RATE = 0.005  # 0.02 0.006 0.005
# the times of generations
N_GENERATIONS = 300


# find non-zero fitness for selection
def get_fitness(pred):
    return pred


# convert binary DNA to decimal and normalize it to a rang(0,5)
def translateDNA(pop):
    index_list = []
    for i in range(len(pop)):
        if pop[i] == 1:
            index_list.append(i)
    return index_list


# nature selection wrt pop fitness
def select(pop, fitness):
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=fitness / fitness.sum())
    return pop[idx]


# roulette wheel selection
def select_gamble(pop, fitness):
    # sort by fitness
    sorted_index = np.argsort(fitness)
    sorted_pop = pop[sorted_index]  # 100,22
    sorted_fitness = fitness[sorted_index]  # 100,
    # out the time queue
    total_fitness = np.sum(sorted_fitness)

    accumulation = [None for col in range(len(sorted_fitness))]
    accumulation[0] = sorted_fitness[0] / total_fitness
    for i in range(1, len(sorted_fitness)):
        accumulation[i] = accumulation[i - 1] + sorted_fitness[i] / total_fitness
    accumulation = np.array(accumulation)

    # roulette wheel selection
    roulette_index = []
    for i in range(POP_SIZE):
        p = np.random.rand()
        for j in range(len(accumulation)):
            if float(accumulation[j]) >= p:
                roulette_index.append(j)
                break
    roulette_dict = {}
    for i in range(len(roulette_index)):
        value = roulette_dict.get(roulette_index[i], 0)
        value += 1
        roulette_dict[roulette_index[i]] = value
    # print(roulette_dict)
    roulette_dict = sorted(roulette_dict.items(), key=lambda x: x[1], reverse=True)
    best_index = []
    # print(roulette_dict[0][0])
    for one in roulette_dict:
        best_index.append(one[0])
    new_pop = []
    new_fitness = []
    for i in range(len(pop)):
        if i not in best_index:
            new_pop.append(sorted_pop[best_index[0]])
            new_fitness.append(sorted_fitness[best_index[0]])
        else:
            new_pop.append(sorted_pop[i])
            new_fitness.append(sorted_fitness[i])

    new_pop = np.array(new_pop)
    return new_pop


# mating process (genes crossover)
def crossover(parent, pop):
    if np.random.rand() < CROSS_RATE:
        # select another individual from pop
        i_ = np.random.randint(0, POP_SIZE, size=1)
        # choose crossover points
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)
        # mating and produce one child
        parent[cross_points] = pop[i_, cross_points]
    return parent


# generation
def mutate(child):
    for i in range(int(DNA_SIZE/POP_SIZE*0.01)):
        rand_index = np.random.randint(DNA_SIZE)
        if np.random.rand() < MUTATION_RATE:
            child[rand_index] = 1 if child[rand_index] == 0 else 0
    return child


# initialize the pop DNA
# pop = np.zeros((POP_SIZE, DNA_SIZE))
pop = np.zeros((POP_SIZE, DNA_SIZE))
# count = 1

# init DNA
for i in range(len(pop)):
    for j in range(len(pop[i])):
        # if count < int(0.005 * DNA_SIZE):
        if np.random.rand() < 0.002:
            pop[i][j] = 1
            # count += 1
# print(pop)
# the training of ga
accuracy_gen = []
features_gen = []
for _ in range(N_GENERATIONS):
    accuracy_list = []
    feature_list = []
    for i in range(POP_SIZE):
        data = input_data[:, translateDNA(pop[i])]
        # data = data[:, pop[i]]
        feature_list.append(np.sum(pop, axis=1)[i])
        #accuracy_list.append(RandomForest_model(data, result)* 0.99 + 0.01 * (1 - np.sum(pop, axis=1)[i] / DNA_SIZE))
        accuracy_list.append(knn_model(data, result) * 0.99 + 0.01 * (1 - np.sum(pop, axis=1)[i] / DNA_SIZE))
        #accuracy_list.append(svm_model(data, result)* 0.99 + 0.01 * (1 - np.sum(pop, axis=1)[i] / DNA_SIZE))
        #accuracy_list.append(QuadraticDiscriminantAnalysis_model(data, result) * 0.99 + 0.01 * (1 - np.sum(pop, axis=1)[i] / DNA_SIZE))
        #accuracy_list.append(Neural_Network().__int__(data, result)[0])
    # GA part(evolution)
    fitness = np.array(accuracy_list)
    features = np.array(feature_list)
    best_features = features[np.argmax(accuracy_list)]
    best_accuracy = (np.max(accuracy_list) - 0.01 * (1 - best_features / DNA_SIZE))/0.99
    accuracy_gen.append(best_accuracy)
    features_gen.append(best_features)
    # print("accuracy: ",best_accuracy, " features: ",best_features )
    pop = select_gamble(pop, fitness)
    # print(pop.sum(axis=1))
    pop_copy = pop.copy()
    #print("the old value",pop.sum(axis=1))
    for parent in pop:
        child = crossover(parent, pop_copy)
        child = mutate(child)
        #print(parent[:].shape)
        parent[:] = child
    #print("the new value",pop.sum(axis=1))
    # print(accuracy_list)
    # print(feature_list)

np.savetxt("",np.max(accuracy_gen))
print(np.max(accuracy_gen))
print(features_gen[np.argsort(accuracy_gen)[-1]])
