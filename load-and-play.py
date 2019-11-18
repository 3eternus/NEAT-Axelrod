import axelrod as axl
from axelrod.action import Action
from axelrod.player import Player
import neat
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


CHECKPOINT_FILE_PATH = 'checkpoints/neat-checkpoint-3992neuron400gens'

## Load the configuration file for NEAT library

CONFIG_FILE = 'my-config.ini'
config = neat.Config(
    genome_type=neat.DefaultGenome,
    reproduction_type=neat.DefaultReproduction,
    species_set_type=neat.DefaultSpeciesSet,
    stagnation_type=neat.DefaultStagnation,
    filename=CONFIG_FILE
)

## How many matches each agent play with each other at one generation
TOURNAMENT_TURNS = 100

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


def create_players_from_checkpoint(p, config):
    '''

    Takes a population of NEAT genomes in loaded population format, makes it a
    player pool for Axelrod tournament.

    '''
    genomes = [(key, p.population[key]) for key in p.population.keys()]
    return create_players_from_genomes(genomes, config)


###############################################################################
###########  END OF THE FUNCTIONS AND THE CLASSES   ###########################
#############  TO INTEGRATE TWO LIBRARIES.  ###################################
###############################################################################


# Load the generation from CHECKPOINT_FILE_PATH
checkpoint = neat.Checkpointer.restore_checkpoint(CHECKPOINT_FILE_PATH)

# Create players from the checkpoint
neat_players = create_players_from_checkpoint(checkpoint, config)
axel_players = [
    axl.Cooperator(),
]

#OLD


players = neat_players + axel_players
# tournament = axl.Tournament(players, turns = TOURNAMENT_TURNS)
# results = tournament.play()

for i, player in enumerate(neat_players):
    tf = axl.TransitiveFingerprint(player, opponents=axel_players)
    data = tf.fingerprint(turns=10, repetitions=10)
    ...  # Write data to disk for analysis
    p = tf.plot(display_names=True)
    p.savefig(f"plot_{i}.png")  # Saves a plot for each player of interest
    p.show()


############ ANALYSIS PLOTS
#
# cooperation_df_nc1 = pd.DataFrame(results.cooperation,
#                                   index=results.players,
#                                   columns=results.players)
# plt.subplots(figsize=(20,15))
# sns.heatmap(cooperation_df_nc1, annot=True).set(
#     xlabel="Cooperations received",
#     ylabel="Cooperations emitted",
#
# )
# cooperation_df_nc1
#
# plt.show()
#
# good_partner_df_nc1 = pd.DataFrame(results.good_partner_matrix,
#                                   index=results.players,
#                                   columns=results.players)
# plt.subplots(figsize=(20,15))
# sns.heatmap(good_partner_df_nc1).set(
#     xlabel="Adversary",
#     ylabel="Reference player",
# )
# good_partner_df_nc1
# plt.show()
#
# payoff_matrix_df_nc1 = pd.DataFrame(results.payoff_matrix,
#                                     index=results.players,
#                                     columns=results.players)
# plt.subplots(figsize=(20,15))
# sns.heatmap(payoff_matrix_df_nc1, annot=True).set(
#     xlabel="Payoff offered",
#     ylabel="Payoff received",
# )
# payoff_matrix_df_nc1
# plt.show()




#print(results.cooperation,
# "cooperation")
#print(results.good_partner_rating, "good partner rating")


##



# tournament = axl.AshlockFingerprint(NeatAgent)
# # #results = tournament.play()
# data = tournament.fingerprint(turns=TOURNAMENT_TURNS, repetitions=10)
# p = tournament.plot(display_names=True)
# p.show()

# strategy = axl.Cooperator()
# probe = neat_players
# af = axl.AshlockFingerprint(strategy, probe)
# data = af.fingerprint(turns=10, repetitions=2, step=0.2)
# p = af.plot()
# p.show()

# OLD PLOTS FOR REFRACT
# # plot
# #plot = axl.Plot(results)
# import matplotlib.style as style
# style.use('seaborn-paper')  # sets the size of the charts
# style.use('ggplot')
# palette = ("plasma_r")
#
# _, ax = plt.subplots()
# title = ax.set_title('Payoff')
# xlabel = ax.set_xlabel('Strategies'
# #)
# #p = plot.boxplot(ax=ax)
#
# ##
# plt.show()
# plt.savefig('test.png')
# results.write_summary('summary.csv')
