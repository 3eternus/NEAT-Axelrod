import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SUMMARIES_DIRECTORY_PATH = "summaries" 
NODECOUNTS_DIRECTORY_PATH = "node_counts"

current_dir = os.path.dirname(os.path.abspath(__file__))
SUMMARIES_DIRECTORY_PATH = os.path.join(current_dir, "summaries") 
NODECOUNTS_DIRECTORY_PATH = os.path.join(current_dir, "node_counts") 
summaries_path_list = os.listdir(SUMMARIES_DIRECTORY_PATH)
nodecounts_path_list = os.listdir(NODECOUNTS_DIRECTORY_PATH)

coop_ratios_list = []
for summary_path in summaries_path_list:
    coop_ratio_list = []
    abs_summaries_path = os.path.join(SUMMARIES_DIRECTORY_PATH, summary_path)
    summaries = os.listdir(abs_summaries_path)
    for n in range(len(summaries)):
        abs_summary_path_generation = os.path.join(abs_summaries_path, "summary" + str(n) + ".csv")
        summary = pd.read_csv(abs_summary_path_generation)
        coop_ratio = np.mean(summary.Cooperation_rating)
        coop_ratio_list.append(coop_ratio)
    coop_ratios_list.append(coop_ratio_list)


coop_ratios_array = np.array(coop_ratios_list, dtype=np.float64)
average_coop_ratios = np.mean(coop_ratios_array, axis=0)
std_coop_ratios     = np.std(coop_ratios_array, axis=0)
 
plt.errorbar(x=np.arange(len(average_coop_ratios)), 
             y=average_coop_ratios, 
             yerr=std_coop_ratios, 
             uplims=True, 
             lolims=True,
             ecolor='red',
             elinewidth=0.4,
             capsize=1,
             errorevery=2
             )
plt.xlabel('Generation')
plt.ylabel('Average cooperation ratio')    
plt.show()
    


node_counts_list = []
for nodecount_path in nodecounts_path_list:
    abs_nodecount_path = os.path.join(NODECOUNTS_DIRECTORY_PATH, nodecount_path)
    with open(os.path.join(abs_nodecount_path, 'list'), 'rb') as f:
        nodecountlist = pickle.load(f)
        node_counts_list.append(nodecountlist)
        
mean_nodecounts = np.mean(node_counts_list)
std_nodecounts = np.std(node_counts_list)

print("Average node count: %.3f" % (mean_nodecounts))
print("Std of node count: %.3f" % (std_nodecounts))
