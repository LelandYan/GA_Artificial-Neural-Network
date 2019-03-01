# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/3/1 8:50'
import numpy as np
from load_data import *
from Classifer_model import *


class GA_model:
    def __init__(self, CROSS_RATE, MUTATION_RATE, N_GENERATIONS):
        self.CROSS_RATE = CROSS_RATE
        self.MUTATION_RATE = MUTATION_RATE
        self.N_GENERATIONS = N_GENERATIONS

    # find non-zero fitness for selection
    def get_fitness(self, accuracy, i):
        return accuracy * 0.99 + 0.01 * (1 - np.sum(self.pop_, axis=1)[i] / self.DNA_SIZE_)

    # convert binary DNA to decimal
    def translateDNA(self, pop):
        index_list = []
        for i in range(len(pop)):
            if pop[i] == 1:
                index_list.append(i)
        return index_list

    # nature selection wrt pop fitness
    def select(self, pop, fitness):
        idx = np.random.choice(np.arange(self.POP_SIZE_), size=self.POP_SIZE_, replace=True, p=fitness / fitness.sum())
        return pop[idx]

    # roulette wheel selection
    def select_gamble(self, pop, fitness):
        np.random.seed(6)
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
        for i in range(self.POP_SIZE_):
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
        roulette_dict = sorted(roulette_dict.items(), key=lambda x: x[1], reverse=True)
        best_index = []
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
    def crossover(self, parent, pop):
        np.random.seed(6)
        if np.random.rand() < self.CROSS_RATE:
            # select another individual from pop
            i_ = np.random.randint(0, self.POP_SIZE_, size=1)
            # choose crossover points
            cross_points = np.random.randint(0, 2, size=self.DNA_SIZE_).astype(np.bool)
            # mating and produce one child
            parent[cross_points] = pop[i_, cross_points]
        return parent

    # generation
    def mutate(self, child):
        np.random.seed(6)
        for i in range(int(self.DNA_SIZE_ / self.POP_SIZE_ * 0.01)):
            rand_index = np.random.randint(self.DNA_SIZE_)
            if np.random.rand() < self.MUTATION_RATE:
                child[rand_index] = 1 if child[rand_index] == 0 else 0
        return child

    def search(self, input_data, label,model):
        np.random.seed(6)
        self.DNA_SIZE_ = input_data.shape[1]
        self.POP_SIZE_ = label.shape[0]
        self.pop_ = np.zeros((self.POP_SIZE_, self.DNA_SIZE_))
        for i in range(len(self.pop_)):
            for j in range(len(self.pop_[i])):
                # if count < int(0.005 * DNA_SIZE):
                if np.random.rand() < 0.002:
                    self.pop_[i][j] = 1
        # the training of ga
        accuracy_gen = []
        features_gen = []
        for _ in range(self.N_GENERATIONS):
            accuracy_list = []
            feature_list = []
            for i in range(self.POP_SIZE_):
                data = input_data[:, self.translateDNA(self.pop_[i])]
                feature_list.append(np.sum(self.pop_, axis=1)[i])
                accuracy_list.append(self.get_fitness(Classifer_model(data, label,model),i))
            # GA part(evolution)
            fitness = np.array(accuracy_list)
            features = np.array(feature_list)
            best_features = features[np.argmax(accuracy_list)]
            best_accuracy = (np.max(accuracy_list) - 0.01 * (1 - best_features / self.DNA_SIZE_)) / 0.99
            accuracy_gen.append(best_accuracy)
            features_gen.append(best_features)
            pop = self.select_gamble(self.pop_, fitness)
            pop_copy = pop.copy()
            for parent in pop:
                child = self.crossover(parent, pop_copy)
                child = self.mutate(child)
                parent[:] = child
        accuracy = np.max(accuracy_gen)
        features = features_gen[np.argsort(accuracy_gen)[-1]]
        return accuracy,features


if __name__ == '__main__':
    data, label = load_data("csv_result-colonTumor.csv")
    model = GA_model(CROSS_RATE=0.8, MUTATION_RATE=0.005, N_GENERATIONS=300)
    classifier_model = [KNeighborsClassifier, GaussianNB, SVC, RandomForestClassifier, LogisticRegression]
    # accuracy, features = model.search(data, label, KNeighborsClassifier)
    # print(accuracy,features)
    # classifier_model = [KNeighborsClassifier]
    with open("GA_result.txt",'w') as f:
        for classifer in classifier_model:
            accuracy,features = model.search(data, label,classifer)
            f.write(str(classifer)+" accuracy:"+str(accuracy)+" features:"+str(features)+"\n")
