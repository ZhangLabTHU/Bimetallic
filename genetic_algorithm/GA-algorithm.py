# copyright THU-Zhanglab
# coding:utf-8 

import random
from operator import itemgetter
import numpy as np
from ase.io import read
from ase.neighborlist import NeighborList
import pickle
from random import sample
from scipy.stats import norm
import os
import warnings
warnings.filterwarnings("ignore")

# extract neighborlist based on pure.traj
atoms = read('pure.traj')
radius = [1.5]*64 # set 1.5 to make sure surf atoms have 10 neighbors and bulk atoms have 14 neighbors.  
nl = NeighborList(radius, self_interaction=False, bothways=True)
nl.update(atoms)
# neighbor store the neighbor atoms index in PdAu.
neighbor_list = {}
coordination_list = {}
adsorption_list = {}
for index in range(len(atoms)):
    indices, offsets = nl.get_neighbors(index)
    indices = indices.tolist()
    neighbor_list[index] = indices 
    coordination_list[index] = len(indices)

# extract adsorptionlist based on pure.traj
# adsorption_list保存每一个位点对应的三层shell
adsorption_coordination_list = {} #按照配位数划分

# 读取pure.traj，扩展3倍
atoms = read('pure.traj') 
extend_atoms = atoms*[3,3,1]

# fcc_indices记录扩胞后16个组成fcc位点的3个原子标号
fcc_indices = [[268,270,269],[270,300,271],[300,302,301],[302,460,303],
              [269,271,284],[271,301,286],[301,303,316],[303,461,318],
              [284,286,285],[286,316,287],[316,318,317],[318,319,476],
              [285,287,332],[287,317,334],[317,319,364],[319,477,366]]

# 通过近邻判断原子数
radius = [1.5]*576 # set 1.5 to make sure surf atoms have 10 neighbors and bulk atoms have 14 neighbors.  
nl = NeighborList(radius, self_interaction=False, bothways=True)
nl.update(extend_atoms)
neighbor_extend_list = {}
coordination_extend_list ={}
for index in range(len(extend_atoms)):
    indices, offsets = nl.get_neighbors(index)
    indices = indices.tolist()
    neighbor_extend_list[index] = indices
    coordination_extend_list[index] = len(indices)

# 每一个fcc位点划分6层特征，6层原子数分别为3，3，3，6，3，40
for num in range(16):
    first_shell = fcc_indices[num]
    
    second_shell = []
    for i in first_shell:
        neighbor_i = neighbor_extend_list[i]
        for j in neighbor_i:
            if j not in first_shell and j not in second_shell:
                second_shell.append(j)
    
    third_shell = []
    for i in second_shell:
        neighbor_i = neighbor_extend_list[i]
        for j in neighbor_i:
            if j not in second_shell and j not in third_shell:
                third_shell.append(j)
    
    position1 = extend_atoms[first_shell[0]].position+[0,0,1.2]
    position2 = extend_atoms[first_shell[1]].position+[0,0,1.2]
    position3 = extend_atoms[first_shell[2]].position+[0,0,1.2]
    center_position = (position1+position2+position3)/3
    second_shell_distance = {}
    for index in second_shell:
        second_shell_distance[index] = np.linalg.norm(center_position-extend_atoms[index].position)
    second_shell_distance_ordered = sorted(second_shell_distance.items(), key=lambda item:item[1])
    second_shell_distance_sub1 = [second_shell_distance_ordered[i][0] for i in range(0,3)] 
    second_shell_distance_sub2 = [second_shell_distance_ordered[i][0] for i in range(3,6)]
    second_shell_distance_sub3 = [second_shell_distance_ordered[i][0] for i in range(6,12)]
    second_shell_distance_sub4 = [second_shell_distance_ordered[i][0] for i in range(12,15)]
    
    adsorption_shell = []
    adsorption_shell.append([i%64 for i in first_shell])
    adsorption_shell.append([i%64 for i in second_shell_distance_sub1])
    adsorption_shell.append([i%64 for i in second_shell_distance_sub2])
    adsorption_shell.append([i%64 for i in second_shell_distance_sub3])
    adsorption_shell.append([i%64 for i in second_shell_distance_sub4])
    adsorption_coordination_list[num] = adsorption_shell


def add_to_list(list_target, shell_pd, shell_coord_pd, shell_au, shell_coord_au):
    if len(shell_pd) !=0:
        list_target.append(len(shell_pd))
        list_target.append(np.mean(shell_coord_pd))
    else:
        list_target.append(0)
        list_target.append(0)

    if len(shell_au) !=0:
        list_target.append(len(shell_au))   # number  
        list_target.append(np.mean(shell_coord_au)) # mean of coordination number 

    else:
        list_target.append(0)
        list_target.append(0)
        
    return list_target


def feature_from_config_int(config_int):
    Feature = []
    au_atom_list = [i for i,x in enumerate(config_int) if x==1]
    ratio = len(au_atom_list)/len(config_int)
    for i in range(16):
        adsorption_coordination_feature = []
        for j in range(5):
            shell = adsorption_coordination_list[i][j]
            shell_pd = [k for k in shell if k not in au_atom_list]
            shell_pd_coord = [coordination_list[k] for k in shell_pd]
            shell_au = [k for k in shell if k in au_atom_list]
            shell_au_coord = [coordination_list[k] for k in shell_au]
            adsorption_coordination_feature = add_to_list(adsorption_coordination_feature, shell_pd, shell_pd_coord, shell_au, shell_au_coord)
        Feature.append(adsorption_coordination_feature + [ratio])
    return Feature

class Gene:
    """
    This is a class to represent individual(Gene) in GA algorithom
    each object of this class have two attribute: data, size
    """
    def __init__(self, **data):
        self.__dict__.update(data)
        self.size = len(data['data'])  # length of gene 
    
class GA:
    """
    This is a class of GA algorithm.
    """

    def __init__(self, parameter):
        """
        Initialize the pop of GA algorithom and evaluate the pop by computing its' fitness value.
        The data structure of pop is composed of several individuals which has the form like that:

        {'Gene':a object of class Gene, 'fitness': 1.02(for example)}
        Representation of Gene is a list: [b s0 u0 sita0 s1 u1 sita1 s2 u2 sita2]

        """
        # parameter = [CXPB, MUTPB, NGEN, popsize, low, up]
        self.parameter = parameter
        self.CXPB = parameter[0]
        self.MUTPB = parameter[1]
        self.NGEN = parameter[2]
        self.popsize = parameter[3]
        self.length = parameter[4]
        self.total_gene = parameter[5]
        self.max_lastgen = parameter[6]
        self.initial_gene_path = parameter[7]
        self.out_path =parameter[8]
        self.preprocessing = pickle.load(open("Preprocessing.pkl", "rb"))
        self.model = pickle.load(open("GPRmodel.model", "rb"))
        
        print('Initilize...')
        
        ### initialize population by random or given structure! ###
        pop = []
        pop_max_fitness = []
        if self.initial_gene_path == '':
            for i in range(self.popsize):
                geneinfo = sample(self.total_gene, self.length) # initialise popluation
                fitness, activity, sigma = self.evaluate(geneinfo)  # evaluate each chromosome
                pop.append({'Gene':Gene(data=geneinfo), 'fitness':fitness, 'activity':activity, 'sigma':sigma})
                pop_max_fitness.append({'Gene':Gene(data=geneinfo), 'fitness':fitness, 'activity':activity, 'sigma':sigma})  # store the gene and its fitness
        else:
            Geneinfo = np.loadtxt(self.initial_gene_path,delimiter=',',encoding='UTF-8-sig')
            for i in range(0,len(Geneinfo)):
                geneinfo = Geneinfo[i]
                fitness,activity,sigma = self.evaluate(geneinfo)  # evaluate each chromosome
                pop.append({'Gene':Gene(data=geneinfo), 'fitness':fitness, 'activity':activity, 'sigma':sigma})  # store the gene and its fitness
                pop_max_fitness.append({'Gene':Gene(data=geneinfo), 'fitness':fitness, 'activity':activity, 'sigma':sigma}) 

        self.pop = pop
        self.pop_max_fitness = pop_max_fitness
        self.best_fit_individual = self.selectBestFit(self.pop) # store the chromosome with highest fitness in the population
        self.best_aci_individual = self.selectBestAci(self.pop) # store the chromosome with highest activity in the population
        print('Construct Initial Population!')

    def evaluate(self, geneinfo, epsilon=0.0001):
        """
        fitness function
        """
        # translate the chromosome to the structure code
        config_int = []
        for i in range(len(self.total_gene)):
            if i in geneinfo:
                config_int.append(1)
            else:
                config_int.append(0)
        feature = feature_from_config_int(config_int)

        # Predict the adsorption energy by ML model
        feature = self.preprocessing.transform(feature)
        ads_energy_predict, sigma_predict = self.model.predict(feature,return_std=True)
        Activity = [0.6*(eo+0.23)-1.376 if eo < 1.58 else -1.645*(eo+0.23)+2.688 for eo in ads_energy_predict]
        
        # u:average activity of 16 fcc sites on PtNi surface
        u = np.mean(Activity)
        sigma = (np.mean([i**2 for i in sigma_predict])) ** 0.5
        Z = (u - self.max_lastgen)/sigma
        EI = (u - self.max_lastgen - epsilon) * norm.cdf(Z) + sigma * norm.pdf(Z)
        return EI, u, sigma

    def selectBestFit(self, pop):
        """
        select the best individual from pop
        """
        s_inds = sorted(pop, key=itemgetter('fitness'), reverse=True)
        return s_inds[0]
    
    def selectBestAci(self, pop):
        """
        select the best individual from pop
        """
        s_inds = sorted(pop, key=itemgetter('activity'), reverse=True)
        return s_inds[0]
    
    def selection(self, individuals, k):
        """
        select some good individuals from pop, note that good individuals have greater probability to be choosen
        for example: a fitness list like that:[5, 4, 3, 2, 1], sum is 15,
        [-----|----|---|--|-]
        012345|6789|101112|1314|15
        we randomly choose a value in [0, 15],
        it belongs to first scale with greatest probability
        """
        s_inds = sorted(individuals, key=itemgetter("fitness"), reverse=True)  # sort the pop by the reference of fitness
        sum_fits = sum(ind['fitness'] for ind in individuals)  # sum up the fitness of the whole pop

        chosen = []
        for _ in range(k):                  # k means that how many individuals are selected which has high fitness
            u = random.random() * sum_fits  # randomly produce a num in the range of [0, sum_fits], as threshold
            sum_ = 0
            for ind in s_inds:
                sum_ += ind['fitness']  # sum up the fitness
                if sum_ >= u:
                    # when the sum of fitness is bigger than u, choose the one, which means u is in the range of
                    # [sum(1,2,...,n-1),sum(1,2,...,n)] and is time to choose the one ,namely n-th individual in the pop
                    chosen.append(ind)
                    break
        # from small to large, due to list.pop() method get the last element
        chosen = sorted(chosen, key=itemgetter("fitness"), reverse=False)
        return chosen

    def crossoperate(self, offspring):
        """
        cross operation
        here we use two points crossoperate
        for example: gene1: [5, 2, 4, 7], gene2: [3, 6, 9, 2], if pos1=1, pos2=2
        5 | 2 | 4  7
        3 | 6 | 9  2
        =
        3 | 2 | 9  2
        5 | 6 | 4  7
        """
        dim = len(offspring[0]['Gene'].data)

        geninfo1 = offspring[0]['Gene'].data  # Gene's data of first offspring chosen from the selected pop
        geninfo2 = offspring[1]['Gene'].data  # Gene's data of second offspring chosen from the selected pop

        if dim == 1:
            pos1 = 0
            pos2 = 1
        else:
            pos1 = random.randrange(1, dim)  # select a position in the range from 0 to dim-1,
            pos2 = random.randrange(1, dim)

        newoff1 = Gene(data=[])  # offspring1 produced by cross operation
        newoff2 = Gene(data=[])  # offspring2 produced by cross operation       
        temp1 = []
        temp2 = []

        geninfo1_avoid = [geninfo1[k] for k in range(dim) if k not in range(min(pos1,pos2),max(pos1,pos2))]
        geninfo2_avoid = [geninfo2[k] for k in range(dim) if k not in range(min(pos1,pos2),max(pos1,pos2))]

        geninfo1_cross = [geninfo1[k] for k in range(min(pos1,pos2),max(pos1,pos2))]
        geninfo2_cross = [geninfo2[k] for k in range(min(pos1,pos2),max(pos1,pos2))]

        gene_all_avail = list(set(geninfo1_cross + geninfo2_cross))
        gene_1_avail = [i for i in gene_all_avail if i not in geninfo1_avoid]
        gene_2_avail = [i for i in gene_all_avail if i not in geninfo2_avoid]

        for i in range(dim):
            if min(pos1, pos2) <= i < max(pos1, pos2):
                temp2.append(gene_2_avail.pop(random.randint(0,len(gene_2_avail)-1)))
                temp1.append(gene_1_avail.pop(random.randint(0,len(gene_1_avail)-1)))            
            else:
                temp2.append(geninfo2[i])
                temp1.append(geninfo1[i])

        newoff1.data = temp1
        newoff2.data = temp2
        return newoff1, newoff2

    def mutation(self, crossoff):
        """
        mutation operation
        """
        dim = len(crossoff.data)

        if dim == 1:
            pos = 0
        else:
            pos = random.randrange(0, dim)  # chose a position in crossoff to perform mutation. 
        available_site = [i for i in self.total_gene if i not in crossoff.data]

        random_number = sample(available_site, 1)
        for item in random_number:
            random_number_item = item
        crossoff.data[pos] = random_number_item
        return crossoff

    def GA_main(self):
        """
        main frame work of GA
        """
        Activity_max_global = []
        Activity_max_current = []
        Activity_fitness_max = []
        Sigma_fitness_max = []
        Fitness_max_global = []
        Fitness_max_current = []
        print("Start of evolution")

        # Begin the evolution
        for g in range(NGEN):

            print("############### Generation {} ###############".format(g))

            # Apply selection based on their converted fitness
            selectpop = self.selection(self.pop, self.popsize)
            nextoff = []
            while len(nextoff) != self.popsize:
                # Apply crossover and mutation on the offspring
                # Select two individuals
                offspring = [selectpop.pop() for _ in range(2)]
                if random.random() < CXPB:  # cross two individuals with probability CXPB
                    crossoff1, crossoff2 = self.crossoperate(offspring)
                    if random.random() < MUTPB:  # mutate an individual with probability MUTPB
                        muteoff1 = self.mutation(crossoff1)
                        muteoff2 = self.mutation(crossoff2)
                        fit_muteoff1, aci_muteoff1, sig_muteoff1 = self.evaluate(muteoff1.data)  # Evaluate the individuals
                        fit_muteoff2, aci_muteoff2, sig_muteoff2 = self.evaluate(muteoff2.data)  # Evaluate the individuals
                        nextoff.append({'Gene': muteoff1, 'fitness': fit_muteoff1, 'activity': aci_muteoff1, 'sigma':sig_muteoff1})
                        nextoff.append({'Gene': muteoff2, 'fitness': fit_muteoff2, 'activity': aci_muteoff2, 'sigma':sig_muteoff2})
                    else:
                        fit_crossoff1, aci_crossoff1, sig_muteoff1 = self.evaluate(crossoff1.data)  # Evaluate the individuals
                        fit_crossoff2, aci_crossoff2, sig_muteoff2 = self.evaluate(crossoff2.data)
                        nextoff.append({'Gene': crossoff1, 'fitness': fit_crossoff1, 'activity': aci_crossoff1, 'sigma':sig_muteoff1})
                        nextoff.append({'Gene': crossoff2, 'fitness': fit_crossoff2, 'activity': aci_crossoff2, 'sigma':sig_muteoff2})
                else:
                    nextoff.extend(offspring)

            # The population is entirely replaced by the offspring
            self.pop = nextoff
            for num in range(len(nextoff)):
                self.pop_max_fitness.append(nextoff[num])

            # order the pop_max_fitness and save half of them by fitness
            sorted_pop_max_fitness = sorted(self.pop_max_fitness, key=itemgetter("fitness"), reverse=True)
            self.pop_max_fitness = []
            for num in range(int(len(sorted_pop_max_fitness)/2)):
                self.pop_max_fitness.append(sorted_pop_max_fitness[num])
            # Store top fitness structure


            # Gather all the fitnesses in one list and print the stats
            fits = [ind['fitness'] for ind in self.pop]
            acis = [ind['activity'] for ind in self.pop]

            best_fit_ind = self.selectBestFit(self.pop) # update best fitness
            if best_fit_ind['fitness'] > self.best_fit_individual['fitness']:
                self.best_fit_individual = best_fit_ind
            
            best_aci_ind = self.selectBestAci(self.pop)
            if best_aci_ind['activity'] > self.best_aci_individual['activity']:
                self.best_aci_individual = best_aci_ind

            Activity_max_current.append(max(acis))
            Activity_max_global.append(self.best_aci_individual['activity'])
            Activity_fitness_max.append(self.best_fit_individual['activity'])
            Sigma_fitness_max.append(self.best_fit_individual['sigma'])
            Fitness_max_current.append(max(fits))
            Fitness_max_global.append(self.best_fit_individual['fitness'])
            #Gene_fitness_max_global.append(self.best_fit_individual['Gene'].data)

            # print("Best individual found is {}, {}".format(self.best_fit_individual['Gene'].data,self.best_fit_individual['fitness']))
            # print("Max fitness of current pop: {}".format(max(fits)))
        
        print("------ End of (successful) evolution ------")
        try:
            os.makedirs(self.out_path, exist_ok=True)
        except Exception as e:
            print(f"An error occurred while creating the directory: {e}")
        np.savetxt(self.out_path+'Activity_max_global.csv',Activity_max_global,delimiter=',')
        np.savetxt(self.out_path+'Activity_max_current.csv',Activity_max_current,delimiter=',')
        np.savetxt(self.out_path+'Activity_fitness_max.csv',Activity_fitness_max,delimiter=',')
        np.savetxt(self.out_path+'Fitness_max_global.csv',Fitness_max_global,delimiter=',')
        np.savetxt(self.out_path+'Fitness_max_current.csv',Fitness_max_current,delimiter=',')
        np.savetxt(self.out_path+'Sigma_fitness_max.csv',Sigma_fitness_max,delimiter=',')
        np.savetxt(self.out_path+'Gene_fitness_max_global.csv',self.best_fit_individual['Gene'].data,delimiter=',')

if __name__ == "__main__":
    LEN = [8, 16, 24, 32, 40, 48, 56]
    CXPB, MUTPB, NGEN, popsize = 0.5, 0.2, 400, 200 # popsize must be even number
    for Len in range(1, 16):
        for i in range(3):
            config_int = [i for i in range(64)]
            max_pre_gen = -0.37
            initial_gene_path = ''
            out_path = str(Len) + '/' + str(i) + '/' 
            parameter = [CXPB, MUTPB, NGEN, popsize, Len, config_int, max_pre_gen, initial_gene_path, out_path]
            run = GA(parameter)
            run.GA_main()
