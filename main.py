import axelrod as axl 
from axelrod.action import Action
from axelrod.player import Player
import neat
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import visualize
import numpy as np


CONFIG_FILE = 'config-my'
config = neat.Config(
    genome_type = neat.DefaultGenome,
    reproduction_type = neat.DefaultReproduction, 
    species_set_type = neat.DefaultSpeciesSet,
    stagnation_type = neat.DefaultStagnation,
    filename = CONFIG_FILE
    )

TOURNAMENT_TURNS = 6
MAX_GENERATIONS = 20

INT_DEFECT = -1
INT_COOPERATION = 1

NEAT_INPUT_SIZE = config.genome_config.num_inputs ## 2
assert NEAT_INPUT_SIZE % 2 == 0, 'Input size should not be an odd number.'



def action_to_binary(action):
        if action == Action.C:
            return INT_COOPERATION
        elif action == Action.D:
            return INT_DEFECT
        else:
            raise Exception('Invalid action type')

def net_output_to_action(net_output):
    if net_output > 0:
        return Action.D
    else:
        return Action.C
    



class NeatAgent(Player):
    
    name = "NeatAgent"
    
    def __init__(self, neural_net):
        super().__init__()
        self.neural_net = neural_net
        
        
    def strategy(self, opponent):
        opponent_history_binary = list(map(action_to_binary, opponent.history))
        self_history_binary = list(map(action_to_binary, self.history))
        if len(opponent_history_binary) < NEAT_INPUT_SIZE / 2:
            opponent_history_binary = [INT_DEFECT] * NEAT_INPUT_SIZE
            self_history_binary = [INT_DEFECT] * NEAT_INPUT_SIZE
        
        single_history_size = NEAT_INPUT_SIZE // 2
        net_input = opponent_history_binary[-single_history_size:] + self_history_binary[-single_history_size:]
        output = self.neural_net.activate(net_input)[0]
        action = net_output_to_action(output)
        return action
        
        
    
def create_players_from_genomes(genomes, config):
    agents = []
    for genome_id, genome in genomes:
        neural_net = neat.nn.RecurrentNetwork.create(genome, config)
        agent = NeatAgent(neural_net)
        agents.append(agent)
    return agents
    



def evaluate_genomes(genomes, config):

    agents = create_players_from_genomes(genomes, config)
    tournament = axl.Tournament(agents, turns = TOURNAMENT_TURNS)
    results = tournament.play()
    mean_scores = np.mean(results.scores, axis = 1)
    for n in range(len(genomes)):
        genome = genomes[n][1]
        genome.fitness = mean_scores[n]
#        print(genome.fitness)








# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)


best_individual = p.run(evaluate_genomes, MAX_GENERATIONS)
best_net = neat.nn.RecurrentNetwork.create(best_individual, config)
visualize.draw_net(config, best_individual, True)
visualize.plot_stats(stats, ylog=False, view=True)

