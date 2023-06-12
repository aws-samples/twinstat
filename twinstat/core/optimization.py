# -*- coding: utf-8 -*-
######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: MIT-0                                     #
######################################################################


import os
from scipy.stats import qmc
import numpy as np
import pandas
import matplotlib.pyplot as plot
from tqdm import tqdm
from joblib import Parallel, delayed


class GeneticAlgorithm:
    def __init__(self, fitness_function, population_size:int, generations:int, bounds:list[tuple],
                       record_method:str = 'csv',
                       sql_info:dict = {},
                       mutation_rate:float=0.1,
                       best_fraction:float = 0.5,
                       crossover_method:str = 'chromosome'):
        '''

        Create object for genetic algorithm global heuristic optimization.

        Parameters
        ----------
        fitness_function :
            Python exectuable function that accepts an array X and ouputs a fitness
            score.  The algorithm seeks to minimize this function. To perform a maximization,
            users should output the negative score.
        population_size : int
            Total sample size for individual chromosomes
        generations : int
            Number of generations to refine the population
        bounds : list[tuple]
            The population will be prevented from going outside
            of these bounds. The location in the list must correspond
            to the input variables.

            E.g. bounds = [(0,1)]

        record_method : str, optional
            Only csv file record is setup. All data saved to
            'GA_generation_data.csv'.  Eventually SQL support will be added.
            The default is 'csv'.
        mutation_rate : float, optional
            What fraction of the population will experience a random mutation.
            The default is 0.1.
        best_fraction : float, optional
            The 'best_fraction' of the population will be kept and the rest removed
            from the population. The default is 0.5.
        crossover_method : str, optional
            chromosome | hillclimbing | switch.

            'Chromosome' uses a random switch point in the two parent vectors to create
            new offspring.

            'Hilleclimbing' uses the random weighted average to create offspring.

            'Switch' will use chromosome until the last 10 generations and then the
            method is switched to hillclimbing to fine tune the remainder.

            The default is 'chromosome'.

        Methods
        -------

        genetic_algorithm:
            perform the optimization
        create_images:
            create a scatter plot for each generation to show the evolution of the population
            also creates a convergence plot after each generation to provide some understanding
            of how the optimization progress.

        Returns
        -------
        None.

        '''
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = []
        self.fitness_scores = []
        self.bounds = np.array(bounds)
        self.chromosome_length = self.bounds.shape[0]
        self.fitness_function = fitness_function
        self.record_method = record_method
        self.generation_count = 0
        self.best_fraction = best_fraction
        self.csv_record = "GA_generation_data.csv"
        self.crossover_method = crossover_method


        #TODO: not setup yet
        if record_method =='sql':
            if len(sql_info) == 0:
                raise ValueError("ERROR: require SQL inputs to record to sql database.")
        else:
            if os.path.isfile(self.csv_record):
                os.remove(self.csv_record)

    def _record_generation_results(self, generation_count):
        #TODO: add sql options
        if self.record_method != 'sql':
            if os.path.isfile(self.csv_record):
                previous_data = pandas.read_csv(self.csv_record)
            else:
                previous_data = None
            data = pandas.DataFrame(self.population)
            data.columns = [f'X{i}' for i in range(data.shape[1])]
            #print(data, self.fitness_scores)
            data['fitness_score'] = self.fitness_scores
            data['generation'] = generation_count
            if previous_data is not None:
                data = pandas.concat([previous_data,data],axis=0)
            data.to_csv(self.csv_record, index = False)

        self.create_images(only_convergence = True)


    def _initialize_population(self):
        cube = qmc.LatinHypercube(self.chromosome_length)
        self.population = cube.random(self.population_size)
        self.population = qmc.scale(self.population, self.bounds[:,0], self.bounds[:,1])


    def _evaluate_population(self):
        #TODO: allow user to change thread count
        # self.fitness_scores = []
        # for chromosome in self.population:
        #     self.fitness_scores.append(self.fitness_function(chromosome))

        self.fitness_scores = \
            Parallel(n_jobs=self.population_size,
                      prefer='threads')(
            delayed(self.fitness_function)(chromosome) for chromosome in self.population
            )


    def _selection(self):
        selected_population = []
        rank = np.argsort(self.fitness_scores)
        n_offspring = int(len(self.fitness_scores)*self.best_fraction)
        #select the smallest fitness functions for proliferation
        best_offspring = rank[:n_offspring]
        selected_population = np.random.choice( best_offspring, size =  self.population_size)
        self.population = np.array(self.population)[selected_population]

    def _crossover(self, method = "chromosome"):
        new_population = []
        while len(new_population) < self.population_size:
            parent1, parent2 = np.random.choice(len(self.population), 2, replace = False)
            parent1, parent2 = self.population[parent1], self.population[parent2]

            if method == 'hillclimbing':
                mixture = np.random.uniform(size=self.chromosome_length)
                child1 = mixture*parent1 + (1-mixture) * parent2
                new_population.extend([child1])
            else:
                crossover_point = np.random.randint(1, self.chromosome_length)
                child1 = np.concatenate( (parent1[:crossover_point], parent2[crossover_point:]))
                child2 = np.concatenate( ( parent2[:crossover_point] , parent1[crossover_point:]))
                new_population.extend([child1, child2])
        self.population = new_population

    def _mutate(self):
        for i in range(len(self.population)):
            for j in range(self.chromosome_length):
                if np.random.uniform() < self.mutation_rate:
                    # normally disturb the gene
                    #TODO: play with this more?
                    self.population[i][j] = self.population[i][j] * (1 + np.random.normal(0,0.1))

    def _bound_population(self):
        self.population = np.clip(self.population, self.bounds[:,0], self.bounds[:,1])

    def genetic_algorithm(self) -> np.array and float:
        '''
        Perform the genetic algorithm optimization.

        Returns
        -------
        np.array
            Best vector
        float
            Best fitness score

        '''
        self._initialize_population()
        for gen in range(self.generations):
            self._evaluate_population()
            self._record_generation_results(gen)
            self._selection()
            if self.crossover_method.lower() == 'switch':
                #TODO: add more options here to manipulate method switch
                if (self.generations - gen ) <=10:
                    self._crossover(method='hillclimbing')
                else:
                    self._crossover(method='chromosome')
            else:
                self._crossover(method=self.crossover_method)
            self._mutate()
            self._bound_population()

            best = np.min(self.fitness_scores)
            print(f"Generation {gen} best fitness {best}")

        best_arg = np.argmin(self.fitness_scores)
        #best_chromosome = max(self.population, key=self.fitness_function)
        return self.population[best_arg], self.fitness_scores[best_arg]


    # TODO: add option to change which variables are plotted
    def create_images(self, only_convergence:bool = False) -> None:
        '''

        Parameters
        ----------
        only_convergence : bool, optional
            If false, a scatter plot is created for each
            generation of the optimization.

            In additional the minimum fitness score is
            plotted for each generation.

            If true, only the convergence figure is
            generated.

            The default is False.


        Returns
        -------
        None.

        '''

        if not os.path.isdir("images"):
            os.mkdir("images")

        #TODO: not setup yet
        if self.record_method =='sql':
            raise ValueError("ERROR: Not setup yet.")
        else:
            df = pandas.read_csv(self.csv_record)

        print("Plotting data.")

        plot.ioff()
        groups = df.groupby(by="generation")
        x_limits = df['X0'].min(), df['X0'].max()
        y_limits = df['X1'].min(), df['X1'].max()
        convergence = []
        for gen, group in tqdm(groups):
            #gen, group = list(groups)[0]
            if not only_convergence:
                plot.figure()
                plot.title(gen)
                plot.scatter(group['X0'], group['X1'])
                plot.xlabel("X0")
                plot.ylabel("X1")
                plot.xlim(x_limits[0], x_limits[1])
                plot.ylim(y_limits[0], y_limits[1])
                plot.tight_layout()
                plot.savefig(f'./images/GA_results_{gen}.png')
                plot.close()

            convergence.append(np.min(group['fitness_score']))

        plot.figure()
        plot.plot(convergence)
        plot.ylabel("Fitness Score")
        plot.xlabel("Iterations")
        plot.tight_layout()
        plot.savefig('./images/GA_convergence.png')
        plot.close()
        plot.ion()


#%% main
if __name__ == '__main__':

    ''' Example.'''

    # Example usage
    population_size = 100
    generations = 100
    bounds = [(0,1),
              (0,10),
              (0,100)]

    def objf(x):
        return x[0] + x[1]**2 + x[2]**3

    ga = GeneticAlgorithm(objf, population_size, generations, bounds, crossover_method="switch")
    best_solution, best_fitness = ga.genetic_algorithm()

    ga.create_images()