import axelrod as axl
from axelrod.action import Action
from axelrod.player import Player
import neat
import visualize 
import numpy as np
import matplotlib.pyplot as plt
import os, glob
import datetime
import pickle


current_dir = os.path.dirname(os.path.abspath(__file__))
date_str = datetime.datetime.now().strftime("%d-%m-%Y--%H.%M.%S")
checkpoints_path = os.path.join(current_dir, "checkpoints", date_str)
plots_path = os.path.join(current_dir, "plots", date_str)
summaries_path = os.path.join(current_dir, "summaries", date_str)
nodecounts_path = os.path.join(current_dir, "node_counts", date_str)

os.mkdir(checkpoints_path)
os.mkdir(plots_path)
os.mkdir(summaries_path)
os.mkdir(nodecounts_path)

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

## Load the configuration file for NEAT library

CONFIG_FILE = 'my-config.ini'
config = neat.Config(
    genome_type=neat.DefaultGenome,
    reproduction_type=neat.DefaultReproduction,
    species_set_type=neat.DefaultSpeciesSet,
    stagnation_type=neat.DefaultStagnation,
    filename=CONFIG_FILE
)

## Set the parameters of the game
#default
default = axl.game.Game()
GAME = default

# snowdrift = axl.game.Game(r=2, s=1, t=3, p=0)
# (self, r: Score = 3, s: Score = 0, t: Score = 5, p: Score = 1) standard default settings
# GAME = snowdrift
#t__(self, r: Score = 3, s: Score = 0, t: Score = 5, p: Score = 1)

#test1 = axl.game.Game(r=12, s=0, t=20, p=4)
#GAME = test1

# After how many generations, save the generations
# If it is set to 0, saving is off
SAVE_GENERATIONS_INTERVAL = 1

## If saving is on, remove old checkpoints
# if SAVE_GENERATIONS_INTERVAL:
#        files = glob.glob('checkpoints/*')
#        for f in files:
#            os.remove(f)

PLOT_ON_CONSOLE = TrueSAVE_PLOTS = True

## How many matches each agent play with each other at one generation
TOURNAMENT_TURNS = 30


# How many generations should the code run
MAX_GENERATIONS = 2

# Read input size of network to use it in the code, do not change.
NEAT_INPUT_SIZE = config.genome_config.num_inputs  ## 2

# It should be an even number since it will take last N action of both of the
#   agents and total input size will be 2 * N
assert NEAT_INPUT_SIZE % 2 == 0, 'Input size should not be an odd number.'


###############################################################################
############### FUNCTIONS AND CLASSES TO INTEGRATE TWO LIBRARIES. #############
####################### DON'T CHANGE ##########################################
###############################################################################

def action_to_binary(action):
    '''

    Conversion from Axelrod action object to -1 or 1

    '''
    if action == Action.C:
        return 1
    elif action == Action.D:
        return -1
    else:
        raise Exception('Invalid action type')


def net_output_to_action(net_output):
    '''

    Conversion from neural network output to Axelrod action object

    '''
    if net_output > 0:
        return Action.D
    else:
        return Action.C


def create_node_names(input_size):
    '''

    Create node names for visualization, regarding the input size of network
    This works for 2, 4, 6 and 8 inputs.

    '''

    if input_size == 2:
        node_names = {-1: 'My last action',
                      -2: 'Opponent\'s last action',
                      0: 'My decision'}

    elif input_size == 4:
        node_names = {-1: 'My last action',
                      -2: 'My second last action',
                      -3: 'Opponent\'s last action',
                      -4: 'Opponent\'s second last action',
                      0: 'My decision'}

    elif input_size == 6:
        node_names = {-1: 'My last action',
                      -2: 'My second last action',
                      -3: 'My third last action',
                      -4: 'Opponent\'s last action',
                      -5: 'Opponent\'s second last actionn',
                      -6: 'Opponent\'s third last action',
                      0: 'My decision'}

    elif input_size == 8:
        node_names = {-1: 'My last action',
                      -2: 'My second last action',
                      -3: 'My third last action',
                      -4: 'My fourth last action',
                      -5: 'Opponent\'s last actionn',
                      -6: 'Opponent\'s second last action',
                      -7: 'Opponent\'s third last action',
                      -8: 'Opponent\'s fourth last action',
                      0: 'My decision'}
    elif input_size == 16:
        node_names = {-1: 'My last action',
                      -2: 'My second last action',
                      -3: 'My third last action',
                      -4: 'My fourth last action',
                      -5: 'Opponent\'s last actionn',
                      -6: 'Opponent\'s second last action',
                      -7: 'Opponent\'s third last action',
                      -8: 'Opponent\'s fourth last action',
                      -9: 'My last action',
                      -10: 'My second last action',
                      -11: 'My third last action',
                      -12: 'My fourth last action',
                      -13: 'Opponent\'s last actionn',
                      -14: 'Opponent\'s second last action',
                      -15: 'Opponent\'s third last action',
                      0: 'My decision'}
    return node_names


class NeatAgent(Player):
    '''
    
    Class utility for encapsulating a NEAT neural net as Axelrod agent
    
    '''

    name = "NeatAgent"

    def __init__(self, neural_net):
        super().__init__()
        self.neural_net = neural_net

    def strategy(self, opponent):
        '''
        
        Overridden strategy function, this must be implemented for Axelrod custom
        agents to work.
        
        '''
        opponent_history_binary = list(map(action_to_binary, opponent.history))
        self_history_binary = list(map(action_to_binary, self.history))

        if len(opponent_history_binary) < NEAT_INPUT_SIZE / 2:
            opponent_history_binary = [-1] * NEAT_INPUT_SIZE
            self_history_binary = [-1] * NEAT_INPUT_SIZE

        single_history_size = NEAT_INPUT_SIZE // 2
        net_input = opponent_history_binary[-single_history_size:] \
                    + self_history_binary[-single_history_size:]
        output = self.neural_net.activate(net_input)[0]
        action = net_output_to_action(output)
        return action

    def __repr__(self):
        '''
        
        This is just for prettier console output if you need to print it.
        
        '''

        return 'NEAT agent'


def create_players_from_genomes(genomes, config):
    '''
    
    Takes a population of NEAT genomes, makes it a player pool for 
    Axelrod tournament. 
    
    '''

    agents = []
    for _, genome in genomes:
        neural_net = neat.nn.FeedForwardNetwork.create(genome, config)
        agent = NeatAgent(neural_net)
        agents.append(agent)
    return agents

#


def cooperation_ratio(result_set):
    '''
    
    Calculates ratio of cooperation actions to the all actions in a tournament
    
    '''

    #cooperations = result_set.cooperating_rating
    # return np.mean(cooperations)
    ##### cooperation normalised cooperation.
    cooperations = result_set.normalised_cooperation
    return np.mean(cooperations)


def evaluate_genomes(genomes, config):
    '''
   
    Assign fitness value to each genome as its score in the tournament.
    This must be implemented for NEAT library to work.
    It also saves results of each tournament to a global variable to summarize 
    and plot the results later
    
    '''

    global results_per_generation
    global matches_played_per_player_per_generation
    agents = create_players_from_genomes(genomes, config)
    tournament = axl.Tournament(agents, game=GAME, turns=TOURNAMENT_TURNS)
    results = tournament.play()
    results_per_generation.append(results)




    matches_played_per_player = []
    for n in range(len(agents)):
        matches_played_per_player.append(len(agents) - 1)

    matches_played_per_player_per_generation.append(matches_played_per_player)

    mean_scores = np.mean(results.scores, axis=1)
    for n in range(len(genomes)):
        genome = genomes[n][1]
        genome.fitness = mean_scores[n]


    ###



###############################################################################
###########  END OF THE FUNCTIONS AND THE CLASSES   ###########################
#############  TO INTEGRATE TWO LIBRARIES.  ###################################
###############################################################################
###############################################################################


'''
    Create an empty list to store the tournament results for each generation
'''
results_per_generation = []
matches_played_per_player_per_generation = []

'''
    Create the population, which is the top-level object for a NEAT run.
'''
p = neat.Population(config)

'''
    Create a stats object to use for showing fitness per generation graph.
'''
stats = neat.StatisticsReporter()
p.add_reporter(stats)

'''
    If you also want outputs on console, uncomment this.
'''
# p.add_reporter(neat.StdOutReporter(True))


'''
    Save checkpoints to checkpoints directory. Do not delete the checkpoints directory
    or it will complain about it can not find the directory.
'''
if SAVE_GENERATIONS_INTERVAL > 0:
    p.add_reporter(neat.Checkpointer(SAVE_GENERATIONS_INTERVAL,
                                     filename_prefix=os.path.join(checkpoints_path, 'neat-checkpoint-')))

'''
    Run the NEAT algorithm for MAX_GENERATIONS time and select the best performing
    agent from the last generation
'''
best_individual = p.run(evaluate_genomes, MAX_GENERATIONS)

'''
    Extract the neural net from genotype of best_indiviual
'''
best_net = neat.nn.FeedForwardNetwork.create(best_individual, config)

'''
    Visualizing the network topology
'''
node_names = create_node_names(NEAT_INPUT_SIZE)
visualize.draw_net(config, best_individual, view=True, node_names=node_names)
visualize.plot_stats(stats, ylog=False, view=True)

'''
    Plotting the cooperation ratio per generation
'''
cooperation_ratio_per_generation = [cooperation_ratio(results) for results in results_per_generation]
plt.plot(cooperation_ratio_per_generation)
plt.xlabel('Generations')
plt.ylabel('Average cooperation ratio')
plt.show()


'''
    Summarize each generation's tournament results and save it to a Excel document
    in the summaries folder. It deletes the old summaries inside the folder first.
    
    Note: Do not delete summaries folder, this saves results in it and complains
     if it can't find the folder.
'''

# files = glob.glob('summaries/*')
# for f in files:
#    os.remove(f)
for n, results in enumerate(results_per_generation):
    results.write_summary(os.path.join(summaries_path, 'summary' + str(n) + '.csv'))

'''
    Summarize each generation's tournament results and save it in the 
     plots folder if SAVE_PLOTS is True. It also shows it on the console if 
     PLOT_ON_CONSOLE is True. It deletes the old plots inside 
     the folder before saving.
    
    Note: Do not delete plots folder, this saves results in it and complains
     if it can't find the folder.
'''
# if SAVE_PLOTS:
#     files = glob.glob('plots/*')
#     for f in files:
#         os.remove(f)

# for n, results in enumerate(results_per_generation):
#    wins = np.array([summary.Wins for summary in results.summarise()])
#    matches_played = np.array(matches_played_per_player_per_generation[n])
#
#    win_ratio = wins / matches_played
#    coop_ratio = [summary.Cooperation_rating for summary in results.summarise()]
#
#    plt.plot(win_ratio, 'ro')
#    plt.plot(coop_ratio, 'b^')
#    plt.xlabel('n_th Individual')
#    plt.ylabel('Win ratio & Cooperation ratio')
#    plt.legend(('Win ratio', 'Cooperation ratio'))
#    if SAVE_PLOTS:
#        plt.savefig(os.path.join(plots_path, 'tournament' + str(n) + '.png'))
#    if PLOT_ON_CONSOLE:
#        plt.show()

'''
    Print and save the count of the nodes
'''

node_counts = []
final_genomes = list(p.population.values())
for n in range(len(final_genomes)):
    individual = final_genomes[n]

    print('There are %d nodes at the %d. individual' % (len(individual.nodes) - 1, n + 1))
    node_counts.append(len(individual.nodes) - 1)

with open(os.path.join(nodecounts_path, 'list'), 'wb') as f:
    pickle.dump(node_counts, f)

####

