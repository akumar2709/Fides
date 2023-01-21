import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import torch.nn.functional as F
import sys
from scipy.special import rel_entr, kl_div, softmax
from scipy.stats import wasserstein_distance, energy_distance
import seaborn as sns

'''
This script calculates de KL Divergence between two probability vectors.
Additional, it performs Attack Scenario #1 over the main model.
A threshold is also calculated between similartities close to 0 and those with a high dissimilarity.
A plot is created.

[05/10/2022] Cases 1 and 5 are put together as Case A; same for cases 2, 3 and 4 as case B.
[05/12/2022] The Wasserstein distance was added as a second distance value besides KLD. calculate_distance() function

Date: 04/26/2022
Author: Mike Guirao
'''


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def calculate_kldiv(pv1, pv2):
    kldiv = F.kl_div(F.softmax(pv1), F.softmax(pv2), reduction='batchmean')
    kldiv_norm = (1 - np.exp(-(kldiv.item())))
    return kldiv_norm


def calculate_distance(pv1, pv2, distance_type = "KLD"):
    '''
    distance_type indicates which distance value is going to be calculated:
        KLD:    Kullback–Leibler divergence (default)
        W:      wasserstein_distance
    '''
    if distance_type == "KLD":
        distance1 = sum(rel_entr(pv1, pv2))#, kl_div(pv1, pv2)
        distance2 = sum(rel_entr(pv2, pv1))
        distance =  (distance1 + distance2)/2
    elif distance_type == "W":
        dists = [i for i in range(len(pv1))]
        distance = energy_distance(dists, dists, pv2, pv1)
    return distance


def get_distance_metric_name(distance_type):
    if distance_type == "KLD":
        distance_metric_name = "JSD"
    elif distance_type == "W":
        distance_metric_name = "Wasserstein Distance"
    return distance_metric_name


def convert_list_from_string_to_double(line):
    #['num_test', 'recordid', 'rn56', 'rn20', 'ground_true']
    dlist = []
    v = line
    #print(len(v56))
    for el in v:
        if(el != ''):
            el = float(el)
            dlist.append(float(el))
    #print(type(dlist), len(dlist), type(dlist[5]), dlist[5])
    return np.exp(dlist)/sum(np.exp(dlist))



def get_detailed_data_for_all_cases(box_plot_data):
    data = box_plot_data
    bp = plt.boxplot(data, showmeans=True)
    medians = [round(item.get_ydata()[0], 10) for item in bp['medians']]
    means = [round(item.get_ydata()[0], 10) for item in bp['means']]
    minimums = [round(item.get_ydata()[0], 10) for item in bp['caps']][::2]
    maximums = [round(item.get_ydata()[0], 10) for item in bp['caps']][1::2]
    q1 = [round(min(item.get_ydata()), 10) for item in bp['boxes']]
    q3 = [round(max(item.get_ydata()), 10) for item in bp['boxes']]
    fliers = [item.get_ydata() for item in bp['fliers']]
    lower_outliers = []
    upper_outliers = []
    for i in range(len(fliers)):
        lower_outliers_by_box = []
        upper_outliers_by_box = []
        for outlier in fliers[i]:
            if outlier < q1[i]:
                lower_outliers_by_box.append(round(outlier, 10))
            else:
                upper_outliers_by_box.append(round(outlier, 10))
        lower_outliers.append(lower_outliers_by_box)
        upper_outliers.append(upper_outliers_by_box)    
        
    # New code
    stats = [medians, means, minimums, maximums, q1, q3, lower_outliers, upper_outliers]
    stats_names = ['Median', 'Mean', 'Minimum', 'Maximum', 'Q1', 'Q3', 'Lower outliers', 'Upper outliers']
    categories = ['CASE 1', 'CASE 2', 'CASE 3', 'CASE 4', 'CASE 5'] # to be updated
    for i in range(len(categories)):
        print(f'\033[1m{categories[i]}\033[0m')
        for j in range(len(stats)):
            print(f'{stats_names[j]}: {stats[j][i]}')
        print('\n')


def find_threshold_for_case(kldiv_pre, kldiv_pos):
    print()
    print("********************* CASE ANALYSIS *********************")
    total_pre = []
    total_pos = []
    for t in np.arange(0, 1, 0.01):
        percentagePre = np.count_nonzero(kldiv_pre >= t)/kldiv_pre.size #<=
        total_pre.append(percentagePre)
        percentagePos = np.count_nonzero(kldiv_pos < t)/kldiv_pos.size #>
        total_pos.append(percentagePos)
        print("Pre: {0}% out of {1}; Post: {2}% out of {3} when t={4}".format(round(percentagePre, 3), kldiv_pre.size, round(percentagePos, 3  ), kldiv_pos.size, t))
    return total_pre, total_pos


def plot_threshold_vs_percentage_for_acase(total_pre, total_pos, case, distance_type):
    plt.plot(np.arange(0, 1, 0.01), total_pre)
    plt.plot(np.arange(0, 1, 0.01), total_pos)
    plt.title("Comparison for " + case + " Pre and Post Attack")
    plt.legend(["Pre Attack", "Post Attack"], loc="lower right")
    plt.xlabel("Threshold", fontsize=16)
    plt.ylabel("Distance metric values (%)" , fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig("pre_vs_post_comparison_" + case + "_" + distance_type + "_" + dataset + "_attack1" ".png")
    plt.show()


def get_detailed_data_from_caseB(box_plot_data):
    # GET METRICS ABOUT CASES C2, C3 AND C4 POS ATTACK
    data = box_plot_data
    bp = plt.boxplot(data, showmeans=True)
    medians = [round(item.get_ydata()[0], 10) for item in bp['medians']]
    means = [round(item.get_ydata()[0], 10) for item in bp['means']]
    minimums = [round(item.get_ydata()[0], 10) for item in bp['caps']][::2]
    maximums = [round(item.get_ydata()[0], 10) for item in bp['caps']][1::2]
    q1 = [round(min(item.get_ydata()), 10) for item in bp['boxes']]
    q3 = [round(max(item.get_ydata()), 10) for item in bp['boxes']]
    fliers = [item.get_ydata() for item in bp['fliers']]
    lower_outliers = []
    upper_outliers = []
    for i in range(len(fliers)):
        lower_outliers_by_box = []
        upper_outliers_by_box = []
        for outlier in fliers[i]:
            if outlier < q1[i]:
                lower_outliers_by_box.append(round(outlier, 10))
            else:
                upper_outliers_by_box.append(round(outlier, 10))
        lower_outliers.append(lower_outliers_by_box)
        upper_outliers.append(upper_outliers_by_box)    
        
    # New code
    stats = [medians, means, minimums, maximums, q1, q3, lower_outliers, upper_outliers]
    stats_names = ['Median', 'Mean', 'Minimum', 'Maximum', 'Q1', 'Q3', 'Lower outliers', 'Upper outliers']
    categories = ['DATASET 1', 'DATASET 2', 'DATASET 3'] # to be updated
    for i in range(len(categories)):
        print(f'\033[1m{categories[i]}\033[0m')
        for j in range(len(stats)):
            print(f'{stats_names[j]}: {stats[j][i]}')
        print('\n')

        
'''
In attack1 the attacker is only interested in the softmax output of the main model, ResNet56 Model.
The attacker will take the first and second highest probabilities from the softmax vector and switch them.
'''
def read_process_csv_file(filename):
    matches_before_attack = 0
    matches_after_attack = 0
    total_lines_processed = 0
    CONSIDERED_CASES = ['C1', 'C2', 'C3', 'C4', 'C5'] # Modify this for those clases that need to be considered in the attack

    kldiv_preC1 = np.zeros(0) #<class 'numpy.ndarray'>
    kldiv_preC2 = np.zeros(0)
    kldiv_preC3 = np.zeros(0)
    kldiv_preC4 = np.zeros(0)
    kldiv_preC5 = np.zeros(0)

    kldiv_posC1 = np.zeros(0) #<class 'numpy.ndarray'>
    kldiv_posC2 = np.zeros(0)
    kldiv_posC3 = np.zeros(0)
    kldiv_posC4 = np.zeros(0)
    kldiv_posC5 = np.zeros(0)

    kldiv_posC1_reclassified = np.zeros(0)
    kldiv_posC2_reclassified = np.zeros(0)
    kldiv_posC3_reclassified = np.zeros(0)
    kldiv_posC4_reclassified = np.zeros(0)
    kldiv_posC5_reclassified = np.zeros(0)

    with open(filename, mode ='r')as file:
        csvFile = csv.reader(file)
        header = next(csvFile)
        for line in csvFile:
            #total_lines_processed += 1

            # Get the probability vectors for RN56 and RN20
            #ddrn56_vector = line[1].replace("tf.Tensor(",'')
            rn56_vector = line[1].replace("'",'')
            rn56_vector = rn56_vector.replace("tf.Tensor(",'')
            rn56_vector = rn56_vector.replace("\n",'')
            rn56_vector = rn56_vector.replace("]",'')
            rn56_vector = rn56_vector.replace("[",'')
            rn56_vector = rn56_vector.replace(read_var,'')
            #rn56 = rn56_vector.strip(']')
            rn56 = rn56_vector.split(" ")
            rn20_vector = line[2].replace("'",'')
            rn20_vector = rn20_vector.replace("tf.Tensor(",'')
            rn20_vector = rn20_vector.replace("\n",'')
            rn20_vector = rn20_vector.replace("]",'')
            rn20_vector = rn20_vector.replace("[",'')
            rn20_vector = rn20_vector.replace(read_var,'')
            rn20 = rn20_vector.split(" ")
            # Get ground true
            ground_true = line[3]

            y = float(ground_true.strip("[]"))

            # Get the case
            case = line[4]
            
            # Now we have lists of doubles, for RN56 and RN20
            rn56 = convert_list_from_string_to_double(rn56)
            rn56 = np.asarray(rn56)
            rn20 = convert_list_from_string_to_double(rn20)
            rn20 = np.asarray(rn20)

            #print(type(rn56), type(rn20), round(np.sum(rn56), 3), round(np.sum(rn20), 3)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>

            # Calculate the distance metric between RN56 and RN20 before the attack.
            kldiv_preattack1 = calculate_distance(rn56, rn20, distance_type)

            if case in CONSIDERED_CASES:
                total_lines_processed += 1

                #****************************** ATTACK SCENARIO #1 INITIATED! ******************************
                firsthighest = np.max(rn56)
                firsthighest_loc = np.argmax(rn56)
                if y == firsthighest_loc:
                    text = 'MATCH'
                    matches_before_attack += 1
                else:
                    text = "NOMATCH"
                #print("Before Attack Scenario 1:", text)
                #print(distance_metric_name + " is:", round(kldiv_preattack1,6))
                array_copy = rn56.copy()
                #print("Firsthighest:", firsthighest, " Location:", firsthighest_loc)
                array_copy[firsthighest_loc] = -10000
                secondhighest = np.max(array_copy)
                secondhighest_loc = np.argmax(array_copy)
                #print("Secondhighest:", secondhighest, " Location:", secondhighest_loc)
                # Let's make the switch in the original ndarray, proceed with attack scenario #1
                rn56[firsthighest_loc] = secondhighest
                rn56[secondhighest_loc] = firsthighest
                rn20_firsthighest_loc = np.argmax(rn20)
                firsthighest_loc = np.argmax(rn56)
                if(firsthighest_loc == y and rn20_firsthighest_loc == y):
                    reclassified_case = "C1"
                    #print(reclassified_case)
                elif(firsthighest_loc == y and rn20_firsthighest_loc != y):
                    reclassified_case = "C2"
                    #print(reclassified_case)
                elif(firsthighest_loc != y and rn20_firsthighest_loc == y):
                    reclassified_case = "C3"
                    #print(reclassified_case)
                elif(firsthighest_loc != y and rn20_firsthighest_loc != y and (firsthighest_loc != rn20_firsthighest_loc)):
                    reclassified_case = "C4"
                    #print(reclassified_case)
                elif(firsthighest_loc != y and rn20_firsthighest_loc != y and (firsthighest_loc == rn20_firsthighest_loc)):
                    reclassified_case = "C5"


                    #print(reclassified_case)
                #****************************** ATTACK SCENARIO #1 FINALIZED! ******************************

                # Calculate the distance metric between RN56 and RN20 after the attack.
                kldiv_posattack1 = calculate_distance(rn56, rn20, distance_type)

                rn56[firsthighest_loc] = (secondhighest + firsthighest)/2 - 0.05
                rn56[secondhighest_loc] = (secondhighest + firsthighest)/2 + 0.05

                kldiv_posattack2 = calculate_distance(rn56, rn20, distance_type)
                # We save each individual distance metric on it's respective array.
                if case == "C1":
                    kldiv_preC1 = np.insert(kldiv_preC1, 0, kldiv_preattack1)
                    kldiv_posC1 = np.insert(kldiv_posC1, 0, kldiv_posattack1)
                    #kldiv_posC1 = np.insert(kldiv_posC1, 0, kldiv_posattack2)
                elif case == "C2":
                    kldiv_preC2 = np.insert(kldiv_preC2, 0, kldiv_preattack1)
                    kldiv_posC2 = np.insert(kldiv_posC2, 0, kldiv_posattack1)
                    kldiv_posC2 = np.insert(kldiv_posC2, 0, kldiv_posattack2)
                elif case == "C3":
                    kldiv_preC3 = np.insert(kldiv_preC3, 0, kldiv_preattack1)
                    kldiv_posC3 = np.insert(kldiv_posC3, 0, kldiv_posattack1)
                    kldiv_posC3 = np.insert(kldiv_posC3, 0, kldiv_posattack2)
                elif case == "C4":
                    kldiv_preC4 = np.insert(kldiv_preC4, 0, kldiv_preattack1)
                    kldiv_posC4 = np.insert(kldiv_posC4, 0, kldiv_posattack1)
                    kldiv_posC4 = np.insert(kldiv_posC4, 0, kldiv_posattack2)
                elif case == "C5":
                    kldiv_preC5 = np.insert(kldiv_preC5, 0, kldiv_preattack1)
                    kldiv_posC5 = np.insert(kldiv_posC5, 0, kldiv_posattack1)
                    kldiv_posC5 = np.insert(kldiv_posC5, 0, kldiv_posattack2)

                # We save each individual distance metric on it's respective array.
                if reclassified_case == "C1":
                    kldiv_posC1_reclassified = np.insert(kldiv_posC1_reclassified, 0, kldiv_posattack1)
                    kldiv_posC1_reclassified = np.insert(kldiv_posC1_reclassified, 0, kldiv_posattack2)
                elif reclassified_case == "C2":
                    kldiv_posC2_reclassified = np.insert(kldiv_posC2_reclassified, 0, kldiv_posattack1)
                    kldiv_posC2_reclassified = np.insert(kldiv_posC2_reclassified, 0, kldiv_posattack2)
                elif reclassified_case == "C3":
                    kldiv_posC3_reclassified = np.insert(kldiv_posC3_reclassified, 0, kldiv_posattack1)
                    kldiv_posC3_reclassified = np.insert(kldiv_posC3_reclassified, 0, kldiv_posattack2)
                elif reclassified_case == "C4":
                    kldiv_posC4_reclassified = np.insert(kldiv_posC4_reclassified, 0, kldiv_posattack1)
                    kldiv_posC4_reclassified = np.insert(kldiv_posC4_reclassified, 0, kldiv_posattack2)
                elif reclassified_case == "C5":
                    kldiv_posC5_reclassified = np.insert(kldiv_posC5_reclassified, 0, kldiv_posattack1)
                    kldiv_posC5_reclassified = np.insert(kldiv_posC5_reclassified, 0, kldiv_posattack2)


            highest_loc = np.argmax(rn56)
            if y == highest_loc:
                text = "MATCH"
                matches_after_attack += 1
            else: 
                text = "NOMATCH"

            #print("After Attack Scenario 1:", text)
            #print(distance_metric_name + " is:", round(kldiv_posattack1, 6))
            #print("\nDetection phase at the client...")
            #print("Calculating KL Divergence over the main probability vector...")
            #print("Calculating KL Divergence over the secondary probability vector...")
            #print("By using a Threshold value of {0} an attack has been {1}".format(threshold, detection_msg))
            #print("="*45)
        # [END] of file processing

        #print("Total lines processed:", total_lines_processed, 
        #        "Number of Matches before the attack:", matches_before_attack, 
        #        "Number of Matches after the attack:", matches_after_attack, 
        #        "KL Divergence:", kldiv_pre.size, kldiv_pos.size)
        #print()

        #print(kldiv_pre, kldiv_pos)


        print("Q1 for preC1 : ", np.quantile(kldiv_preC1, .25))
        print("Q1 for preC1 : ", np.quantile(kldiv_preC1, .50))
        print("Q1 for preC2 : ", np.quantile(kldiv_preC2, .25))
        print("Q1 for preC3 : ", np.quantile(kldiv_preC3, .25))
        print("Q1 for preC4 : ", np.quantile(kldiv_preC4, .25))
        print("Q1 for preC5 : ", np.quantile(kldiv_preC5, .25))
        print()
        print("Q3 for preC1 : ", np.quantile(kldiv_preC1, .75))
        print("Q3 for preC2 : ", np.quantile(kldiv_preC2, .75))
        print("Q3 for preC3 : ", np.quantile(kldiv_preC3, .75))
        print("Q3 for preC4 : ", np.quantile(kldiv_preC4, .75))
        print("Q3 for preC5 : ", np.quantile(kldiv_preC5, .75))
        kldiv_posC1 = kldiv_posC1
        kldiv_posC2 = kldiv_posC2
        kldiv_posC3 = kldiv_posC3
        kldiv_posC4 = kldiv_posC4
        kldiv_posC5 = kldiv_posC5
        
        print()
        print("Q1 for posC1 : ", np.quantile(kldiv_posC1, .25))
        print("Q1 for posC2 : ", np.quantile(kldiv_posC2, .25))
        print("Q1 for posC3 : ", np.quantile(kldiv_posC3, .25))
        print("Q1 for posC4 : ", np.quantile(kldiv_posC4, .25))
        print("Q1 for posC5 : ", np.quantile(kldiv_posC5, .25))
        print()
        print("Q3 for posC1 : ", np.quantile(kldiv_posC1, .75))
        print("Q3 for posC2 : ", np.quantile(kldiv_posC2, .75))
        print("Q3 for posC3 : ", np.quantile(kldiv_posC3, .75))
        print("Q3 for posC4 : ", np.quantile(kldiv_posC4, .75))
        print("Q3 for posC5 : ", np.quantile(kldiv_posC5, .75))
        print()
        kldiv_posC1_reclassified = kldiv_posC1_reclassified
        kldiv_posC2_reclassified = kldiv_posC2_reclassified
        kldiv_posC3_reclassified = kldiv_posC3_reclassified
        kldiv_posC4_reclassified = kldiv_posC4_reclassified
        kldiv_posC5_reclassified = kldiv_posC5_reclassified
        print()

        print("Q1 for posC1 (Reclassified): ", np.quantile(kldiv_posC1_reclassified, .25))
        print("Q1 for posC2 (Reclassified): ", np.quantile(kldiv_posC2_reclassified, .25))
        print("Q1 for posC3 (Reclassified): ", np.quantile(kldiv_posC3_reclassified, .25))
        print("Q1 for posC4 (Reclassified): ", np.quantile(kldiv_posC4_reclassified, .25))
        print("Q1 for posC5 (Reclassified): ", np.quantile(kldiv_posC5_reclassified, .25))
        print()
        print("Q3 for posC1 (Reclassified): ", np.quantile(kldiv_posC1_reclassified, .75))
        print("Q3 for posC2 (Reclassified): ", np.quantile(kldiv_posC2_reclassified, .75))
        print("Q3 for posC3 (Reclassified): ", np.quantile(kldiv_posC3_reclassified, .75))
        print("Q3 for posC4 (Reclassified): ", np.quantile(kldiv_posC4_reclassified, .75))
        print("Q3 for posC5 (Reclassified): ", np.quantile(kldiv_posC5_reclassified, .75))
        print(" ")
        """
        kldiv_preC1 = NormalizeData(kldiv_preC1)
        kldiv_preC2 = NormalizeData(kldiv_preC2)
        kldiv_preC3 = NormalizeData(kldiv_preC3)
        kldiv_preC4 = NormalizeData(kldiv_preC4)
        kldiv_preC5 = NormalizeData(kldiv_preC5)

        kldiv_posC1 = NormalizeData(kldiv_posC1)
        kldiv_posC2 = NormalizeData(kldiv_posC2)
        kldiv_posC3 = NormalizeData(kldiv_posC3)
        kldiv_posC4 = NormalizeData(kldiv_posC4)
        kldiv_posC5 = NormalizeData(kldiv_posC5)

        kldiv_posC1_reclassified = NormalizeData(kldiv_posC1_reclassified)
        kldiv_posC2_reclassified = NormalizeData(kldiv_posC2_reclassified)
        kldiv_posC3_reclassified = NormalizeData(kldiv_posC3_reclassified)
        kldiv_posC4_reclassified = NormalizeData(kldiv_posC4_reclassified)
        kldiv_posC5_reclassified = NormalizeData(kldiv_posC5_reclassified)
        """

        # Joing in one array C1 and C5; and in another one C2, C3 abd C4.
        kldiv_pos_CA = np.concatenate((kldiv_posC1, kldiv_posC5))
        kldiv_pos_CB = np.concatenate((kldiv_posC2, kldiv_posC3, kldiv_posC4))
        kldiv_pre_CA = np.concatenate((kldiv_preC1, kldiv_preC5))
        kldiv_pre_CB = np.concatenate((kldiv_preC2, kldiv_preC3, kldiv_preC4))
        
        print("Q1 for preCA : ", np.quantile(kldiv_pre_CA, .25))
        print("Q1 for preCB : ", np.quantile(kldiv_pre_CB, .25))

        print("Q2 for preCA : ", np.quantile(kldiv_pre_CA, .50))
        print("Q2 for preCB : ", np.quantile(kldiv_pre_CB, .50))

        print("Q3 for preCA : ", np.quantile(kldiv_pre_CA, .75))
        print("Q3 for preCB : ", np.quantile(kldiv_pre_CB, .75))
        print(" ")
        print("Q1 for posCA : ", np.quantile(kldiv_pos_CA, .25))
        print("Q1 for posCB : ", np.quantile(kldiv_pos_CB, .25))

        print("Q2 for posCA : ", np.quantile(kldiv_pos_CA, .50))
        print("Q2 for posCB : ", np.quantile(kldiv_pos_CB, .50))

        print("Q3 for posCA : ", np.quantile(kldiv_pos_CA, .75))
        print("Q3 for posCB : ", np.quantile(kldiv_pos_CB, .75))

        # Debugging 
        #print("[Pre Case B] ======>> ", np.max(kldiv_pre_CaseB), np.min(kldiv_pre_CaseB)) # 1.0 0.0
        #print("[Pos Case B] ======>> ", np.max(kldiv_pos_CaseB), np.min(kldiv_pos_CaseB)) # 1.0 0.0

        
        from matplotlib.pyplot import figure
        import pandas as pd
        import seaborn as sns
        figure(figsize=(12,7))
        sns.set_theme(style="whitegrid")
        Entropy_concat = np.concatenate((kldiv_preC1, kldiv_preC2, kldiv_preC3, kldiv_preC4, kldiv_preC5,
                                        kldiv_posC1, kldiv_posC2, kldiv_posC3, kldiv_posC4, kldiv_posC5))
                                        #, 
                                        #kldiv_posC1_reclassified, kldiv_posC2_reclassified, kldiv_posC3_reclassified, 
                                        ##kldiv_posC4_reclassified, kldiv_posC5_reclassified))
        Entropy_case = np.concatenate((np.full(len(kldiv_preC1), "C1"), np.full(len(kldiv_preC2), "C2"),
                                        np.full(len(kldiv_preC3), "C3"), np.full(len(kldiv_preC4), "C4"), np.full(len(kldiv_preC5), "C5"),
                                        np.full(len(kldiv_posC1), "C1"), np.full(len(kldiv_posC2), "C2"),
                                        np.full(len(kldiv_posC3), "C3"), np.full(len(kldiv_posC4), "C4"), np.full(len(kldiv_posC5), "C5")))
                                        #np.full(len(kldiv_posC1_reclassified), "C1"), np.full(len(kldiv_posC2_reclassified), "C2"),
                                        #np.full(len(kldiv_posC3_reclassified), "C3"), np.full(len(kldiv_posC4_reclassified), "C4"), np.full(len(kldiv_posC5_reclassified), "C5")))
        entropy_phase = np.concatenate((np.full(len(kldiv_preC1), "Pre-Attack"), np.full(len(kldiv_preC2), "Pre-Attack"),
                                        np.full(len(kldiv_preC3), "Pre-Attack"), np.full(len(kldiv_preC4), "Pre-Attack"), np.full(len(kldiv_preC5), "Pre-Attack"),
                                        np.full(len(kldiv_posC1), "Post-Attack"), np.full(len(kldiv_posC2), "Post-Attack"),
                                        np.full(len(kldiv_posC3), "Post-Attack"), np.full(len(kldiv_posC4), "Post-Attack"), np.full(len(kldiv_posC5), "Post-Attack")))
                                        #np.full(len(kldiv_posC1), "Classified Post-Attack"), np.full(len(kldiv_posC2), "Classified Post-Attack"),
                                        #np.full(len(kldiv_posC3), "Classified Post-Attack"), np.full(len(kldiv_posC4), "Classified Post-Attack"), np.full(len(kldiv_posC5), "Classified Post-Attack")))
        data = {'Entropy':Entropy_concat, 'Case':Entropy_case, 'Phase': entropy_phase}
        
        violin_data = pd.DataFrame(data)
        print(" ")
        print(violin_data)
        sns.color_palette("bright", 1)
        ax = sns.boxplot(x="Case", y="Entropy", hue="Phase", data=data, palette=["#83ff52","#e8000b"])
# Draw a nested barplot by species and sex
        plt.yscale("log")
        plt.yticks(fontsize=30)
        plt.xticks(fontsize=30)
        plt.legend(loc="lower right", fontsize=35)
        plt.ylim(10**-24, 10**3)
        plt.title(dataset, fontsize=45)
        plt.xlabel("Cases" , fontsize=45)
        plt.ylabel(distance_metric_name, fontsize=38)
        plt.tight_layout()
        plt.savefig(model_name + "_" + dataset + "_"+ distance_type + ".pdf")
        plt.show()
        
        # # PLOT #1
        # ##### [BEGIN] PLOT ALL CASES DISTANCE METRIC VALUES BEFORE THE ATTACK
        # plt.rcParams.update({'font.size': 14})
        # from matplotlib.pyplot import figure
        # figure(figsize=(15,8))

        # box_plot_pre_data=[kldiv_preC1, kldiv_preC2, kldiv_preC3, kldiv_preC4, kldiv_preC5]
        # box_plot_pos_data=[kldiv_posC1, kldiv_posC2, kldiv_posC3, kldiv_posC4, kldiv_posC5]
        # box_plot_pos_reclassified_data=[kldiv_posC1_reclassified, kldiv_posC2_reclassified, kldiv_posC3_reclassified, kldiv_posC4_reclassified, kldiv_posC5_reclassified]
        
        
        # #box = plt.boxplot(box_plot_data, patch_artist=True,labels=['C1', 'C2', 'C3', 'C4', 'C5'], showmeans=True)

        # box_pre = plt.boxplot(box_plot_pre_data, positions=np.array(np.arange(len(box_plot_pre_data)))*2.5-0.35,
        #                        widths=0.6, patch_artist=True , showmeans=True)
        # box_pos = plt.boxplot(box_plot_pos_data, positions=np.array(np.arange(len(box_plot_pre_data)))*2.5+1.05,
        #                        widths=0.6, patch_artist=True , showmeans=True)                     
        # plt.semilogy()

        # def define_box_properties(plot_name, color_code, label):
        #     for k, v in plot_name.items():
        #         plt.setp(plot_name.get(k), color=color_code )
                
        #     for patch in plot_name['boxes']:
        #         patch.set(facecolor=color_code) 
        #     plt.plot([], c=color_code, label=label) 
        #     plt.legend(loc="lower right" , fontsize=25)
        # define_box_properties(box_pre, '#38b058', 'Pre-Attack')
        # define_box_properties(box_plot_pos_reclassified, '#3098ff', 'Post-Attack(Reclassified)')
        # define_box_properties(box_pos, '#ff5133', 'Post-Attack')
        
        # ticks = ['C1', 'C2', 'C3', 'C4', 'C5']
        # colors = ['cyan', 'red', 'green', 'blue', 'yellow']
        # plt.xticks([0.35, 2.85, 5.35, 7.85, 10.35], ticks, fontsize=21)
        # plt.yticks(fontsize=21)
        # #plt.tight_layout()
        # plt.title("Assesment of entropy before and after Attack 1 on " + dataset + " dataset" , fontsize=27)
        # plt.xlabel("Cases" , fontsize=32)
        # plt.ylabel(distance_metric_name + " (Norm.)" , fontsize=27)
        # plt.tight_layout()
        # #plt.savefig("Attack_on_" + dataset + "_dataset_" + distance_type + ".pdf")
        # plt.show()
        # ##### [END] PLOT ALL CASES DISTANCE METRIC VALUES BEFORE THE ATTACK

        # #print("Pre Attack Data:")
        # #get_detailed_data_for_all_cases(box_plot_data)

        # #Trying violin plot

        # # PLOT #2
        # ##### [BEGIN] PLOT ALL CASES DISTANCE METRIC VALUES AFTER THE ATTACK
        # from matplotlib.pyplot import figure
        # figure(figsize=(15,8))

        # box_plot_data=[kldiv_posC1, kldiv_posC2, kldiv_posC3, kldiv_posC4, kldiv_posC5]
        # box = plt.boxplot(box_plot_data, patch_artist=True,labels=['C1', 'C2', 'C3', 'C4', 'C5'], showmeans=True)
        # plt.semilogy()
        # colors = ['cyan', 'red', 'green', 'blue', 'yellow']
        # for patch, color in zip(box['boxes'], colors):
        #     patch.set_facecolor(color)
        # #plt.tight_layout()
        # plt.title("Comparison between each of the different cases after an attack")
        # plt.xlabel("Cases")
        # plt.ylabel("Distance value")
        # plt.savefig("cases_after_the_attack_" + distance_type + ".png")
        # plt.show()
        # ##### [END] PLOT ALL CASES DISTANCE METRIC VALUES AFTER THE ATTACK

        
        # # PLOT #3
        # ##### [BEGIN] PLOT CASES A & B DISTANCE METRIC VALUES BEFORE THE ATTACK
        # from matplotlib.pyplot import figure
        # figure(figsize=(15,8))

        # box_plot_data=[kldiv_pos_CaseA, kldiv_pos_CaseB]
        # box = plt.boxplot(box_plot_data, patch_artist=True,labels=['A', 'B'], showmeans=True)
        # plt.semilogy()
        # colors = ['cyan', 'red']
        # for patch, color in zip(box['boxes'], colors):
        #     patch.set_facecolor(color)
        # #plt.tight_layout()
        # plt.title("Comparison between each of the different cases after an attack")
        # plt.xlabel("Cases")
        # plt.ylabel("Distance value")
        # plt.savefig("cases_after_the_attack_Cases_AB_" + distance_type + ".png")
        # plt.show()
        # ##### [END] PLOT CASES A & B DISTANCE METRIC VALUES BEFORE THE ATTACK



               
        # # Now let's find a acceptable threshold for each class
        # total_pre, total_pos = find_threshold_for_case(kldiv_pre_CaseA, kldiv_pos_CaseA)
        # plot_threshold_vs_percentage_for_acase(total_pre, total_pos, "CA", distance_type)

        # total_pre, total_pos = find_threshold_for_case(kldiv_pre_CaseB, kldiv_pos_CaseB)
        # plot_threshold_vs_percentage_for_acase(total_pre, total_pos, "CB", distance_type)


        # #get_detailed_data_from_caseB([kldiv_preC2, kldiv_preC3, kldiv_preC4])
        # #get_detailed_data_from_caseB([kldiv_posC2, kldiv_posC3, kldiv_posC4])

        # # Do we have any distance metric value equal to zero?
        # print()
        # print(f"# of zeros in CA Pre-Attack: {np.count_nonzero(kldiv_pre_CaseA == 0.0)}")
        # print(f"# of zeros in CA Post-Attack: {np.count_nonzero(kldiv_pos_CaseA == 0.0)}")
        # print(f"# of zeros in CB Pre-Attack: {np.count_nonzero(kldiv_pre_CaseB == 0.0)}")
        # print(f"# of zeros in CB Post-Attack: {np.count_nonzero(kldiv_pos_CaseB == 0.0)}")

        # '''
        # total_pre, total_pos = find_threshold_for_case(kldiv_preC3, kldiv_posC3)
        # plot_threshold_vs_percentage_for_acase(total_pre, total_pos, "C3")        

        # total_pre, total_pos = find_threshold_for_case(kldiv_preC4, kldiv_posC4)
        # plot_threshold_vs_percentage_for_acase(total_pre, total_pos, "C4")

        # total_pre, total_pos = find_threshold_for_case(kldiv_preC5, kldiv_posC5)
        # plot_threshold_vs_percentage_for_acase(total_pre, total_pos, "C5")


        
        # total_pre = 0 
        # total_pos = 0
        # supertotal_pre = [] 
        # supertotal_pos = []
        # max_pre = 0
        # max_pos = 0
        # threshold = 0
        # for t in np.arange(0, 1, 0.01):
        #     total_pre = np.count_nonzero(kldiv_pre <= t)/total_lines_processed
        #     supertotal_pre.append(total_pre)
        #     if total_pre > max_pre:
        #         max_pre = total_pre
        #         threshold = t
            
        #     total_pos = np.count_nonzero(kldiv_pos <= t)/total_lines_processed
        #     supertotal_pos.append(total_pos)
        #     if total_pos > max_pos:
        #         max_pos = total_pos
            
        #     if max_pos <= 0.9:
        #         print(">> Pre-attack:", round(max_pre,3)*100, " - Pos-attack:", round(max_pos,3)*100, round(1-max_pos,3)*100, threshold)

        
        # #print(supertotal)

        # # Pre-attack: 92.2  - Pos-attack: 10.5 89.5 0.1
        # plt.plot(np.arange(0, 1, 0.01), supertotal_pre)
        # plt.plot(np.arange(0, 1, 0.01), supertotal_pos)
        # plt.plot((0.1, 0.1), (0, 1), linewidth=4)
        # plt.title("Comparison Pre vs Pos Attack")
        # plt.legend(["Pre Attack", "Post Attack"], loc="lower right")
        # plt.xlabel("threshold")
        # plt.ylabel("%")
        # plt.savefig("pre_pos_comparison.png")
        # plt.show()
        # '''




    file.close()


#*******************************************
#*                                         *
#************** MAIN SECTION ***************
#*                                         *
#*******************************************



filenames_cifar10 = ['prediction_csv/prediction_resnet_cifar10.csv', 'prediction_csv/prediction_densenet_cifar10.csv', 'prediction_csv/prediction_efficientnet_cifar10.csv']
filenames_cifar100 = ['prediction_csv/prediction_resnet_cifar100.csv', 'prediction_csv/prediction_densenet_cifar100.csv', 'prediction_csv/prediction_efficientnet_cifar100.csv']
#filename = 'dummy_data.csv'
classes = ['[0]', '[1]', '[2]', '[3]', '[4]', '[5]', '[6]', '[7]', '[8]', '[9]']
dataset = sys.argv[2].lower()
print(dataset)
if(dataset == "resnet10"):
    model_name = "ResNet"
    dataset = "CIFAR-10"
    filename = filenames_cifar10[0]
    read_var = ", shape=(1, 10), dtype=float32)"
elif(dataset == "densenet10"):
    model_name = "DenseNet"
    dataset = "CIFAR-10"
    filename = filenames_cifar10[1]
    read_var = ", shape=(1, 10), dtype=float32)"
elif(dataset == "efficientnet10"):
    model_name = "EfficientNet"
    dataset = "CIFAR-10"
    filename = filenames_cifar10[2]
    read_var = ", shape=(1, 10), dtype=float32)"
elif(dataset == "resnet100"):
    model_name = "ResNet"
    dataset = "CIFAR-100"
    filename = filenames_cifar100[0]
    read_var = ", shape=(1, 100), dtype=float32)"
elif(dataset == "densenet100"):
    model_name = "DenseNet"
    dataset = "CIFAR-100"
    filename = filenames_cifar100[1]
    read_var = ", shape=(1, 100), dtype=float32)"
elif(dataset == "efficientnet100"):
    model_name = "EfficientNet"
    dataset = "CIFAR-100"
    filename = filenames_cifar100[2]
    read_var = ", shape=(1, 100), dtype=float32)"
    #             0       1       2      3       4      5       6       7        8       9
    #classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
distance_type = sys.argv[1].upper()
distance_metric_name = get_distance_metric_name(distance_type)

print("The {0} distance metric will be used!\n".format(distance_metric_name))
csvfile = read_process_csv_file(filename)
