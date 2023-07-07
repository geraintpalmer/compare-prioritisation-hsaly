import pandas as pd
import ciw
import numpy as np
from collections import namedtuple
import math
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats

def calculate_hsaly(start, intervention, observation_period, pre, post):
    if intervention - start < observation_period:
        return (pre * (intervention - start)) + (post * (start + observation_period - intervention))
    return pre * observation_period

class Patient(ciw.Individual):
    def __init__(self, id_number, customer_class=0, priority_class=0, simulation=False):
        super().__init__(id_number, customer_class, priority_class, simulation)
        person = self.data.loc[ciw.random.choice(self.data.index)]
        self.pre_q = person['Hip Replacement Pre-Op Q Score']
        self.post_q = person['Hip Replacement Post-Op Q Score']
        self.customer_class = int(person[self.priority_column])
        self.priority_class = self.simulation.network.priority_class_mapping[self.customer_class]
        self.prev_priority_class = self.priority_class
        self.previous_class = self.customer_class

InterventionRecord = namedtuple('Record', [
    'id_number',
    'customer_class',
    'original_customer_class',
    'node',
    'arrival_date',
    'waiting_time',
    'service_start_date',
    'service_time',
    'service_end_date',
    'time_blocked',
    'exit_date',
    'destination',
    'queue_size_at_arrival',
    'queue_size_at_departure',
    'server_id',
    'record_type',
    'pre_PRO',
    'post_PRO'
    ])

class Intervention(ciw.Node):
    def write_individual_record(self, individual):
        if math.isinf(self.c):
            server_id = False
        else:
            server_id = individual.server.id_number
                
        record = InterventionRecord(
            individual.id_number,
            individual.previous_class,
            individual.original_class,
            self.id_number,
            individual.arrival_date,
            individual.service_start_date - individual.arrival_date,
            individual.service_start_date,
            individual.service_end_date - individual.service_start_date,
            individual.service_end_date,
            individual.exit_date - individual.service_end_date,
            individual.exit_date,
            individual.destination,
            individual.queue_size_at_arrival,
            individual.queue_size_at_departure,
            server_id,
            'service',
            individual.pre_q,
            individual.post_q)
        individual.data_records.append(record)

def build_and_run_simulation(data, priority_column, referrals_per_day, interventions_per_day, n_years, warmup, cooldown, observation_period, seed=0):
    n_priority_classes = data[priority_column].max() + 1

    arrival_distributions = {
        f'Class {i}': [ciw.dists.NoArrivals(), ciw.dists.NoArrivals()] for i in range(n_priority_classes)}
    arrival_distributions['Class 0'][0] = ciw.dists.Exponential(rate=referrals_per_day)
    
    ciw.seed(seed)
    N = ciw.create_network(
        arrival_distributions=arrival_distributions,
        service_distributions={f'Class {i}': [ciw.dists.Deterministic(value=0), ciw.dists.Deterministic(value=1)] for i in range(n_priority_classes)},
        number_of_servers=[1, interventions_per_day],
        routing={f'Class {i}': [[0.0, 1.0], [0.0, 0.0]] for i in range(n_priority_classes)},
        priority_classes={f'Class {i}': i for i in range(n_priority_classes)}
    )
    Patient.data = data
    Patient.priority_column = priority_column
    Q = ciw.Simulation(N, individual_class=Patient, node_class=[ciw.Node, Intervention])
    Q.simulate_until_max_time((warmup + n_years + cooldown) * 365, progress_bar=True)

    recs = Q.get_all_records()
    recs = pd.DataFrame([r for r in recs if r.node==2 if r.arrival_date > (warmup * 365) if r.arrival_date < (warmup + n_years * 365)])
    recs['hsaly_days'] = recs.apply(lambda row: calculate_hsaly(start=row['arrival_date'], intervention=row['service_start_date'], observation_period=(observation_period*365), pre=row['pre_PRO'], post=row['post_PRO']), axis=1)
    recs['hsaly_years'] =  recs['hsaly_days'] / 365
    return recs


def plot_results(list_of_recs, names, baseline):
    fig, axarr = plt.subplots(3, 4, figsize=(22, 15))
    
    # HSALYs
    max_hsaly = max(r['hsaly_years'].max() for r in list_of_recs)
    for i, recs in enumerate(list_of_recs):
        mean_hsaly = recs['hsaly_years'].mean()
        var_hsaly = recs['hsaly_years'].var()
        sns.kdeplot(recs['hsaly_years'], ax=axarr[(0, i)], color='black', linewidth=2)
        axarr[(0, i)].set_xlim(0, max_hsaly)
        axarr[(0, i)].set_title(names[i], fontsize=24, pad=60)
        axarr[(0, i)].set_yticks([])
        if i == 1:
            axarr[(0, i)].annotate('PDF of HSALY', (-1.62, 0.5), xycoords='axes fraction', fontsize=20, rotation=90, va='center')
        axarr[(0, i)].set_ylabel('')
        axarr[(0, i)].set_xlabel('HSALY', fontsize=16)
        axarr[(0, i)].spines['right'].set_visible(False)
        axarr[(0, i)].spines['top'].set_visible(False)
        axarr[(0, i)].spines['left'].set_visible(True)
        axarr[(0, i)].spines['bottom'].set_visible(True)
        axarr[(0, i)].annotate(f"Mean = {round(mean_hsaly, 2)}\nVariance = {round(var_hsaly, 2)}", (0.1, 0.5), xycoords='axes fraction', fontsize=14)
    
        
    # HSALY Gained
    combined_recs = pd.concat([recs.set_index('id_number')['hsaly_years'] for recs in list_of_recs], axis=1, keys=names)
    for i, name in enumerate(names):
        if i == 0:
            axarr[(1, i)].axis('off')
        else:
            hsaly_diffs = combined_recs[name] - combined_recs[baseline]
            ppos_hslaydiff = (hsaly_diffs > 0).mean()
            sns.kdeplot(hsaly_diffs, ax=axarr[(1, i)], color='grey', linewidth=0, fill=True, clip=(0, None))
            sns.kdeplot(hsaly_diffs, ax=axarr[(1, i)], color='black', linewidth=2)
            axarr[(1, i)].annotate(f"% with Positive\nHSALY = {round(ppos_hslaydiff * 100, 1)}%", (-0.05, 0.5), xycoords='axes fraction', fontsize=14)
            axarr[(1, i)].spines['left'].set_position('zero')
            axarr[(1, i)].set_xlabel("HSALY Gained")
            axarr[(1, i)].spines['right'].set_visible(False)
            axarr[(1, i)].spines['top'].set_visible(False)
            axarr[(1, i)].spines['left'].set_visible(True)
            axarr[(1, i)].spines['bottom'].set_visible(True)
            axarr[(1, i)].set_xlim(-4, 4)
            axarr[(1, i)].set_yticks([])
            if i == 1:
                axarr[(1, i)].annotate('PDF of HSALY Gained', (-1.62, 0.5), xycoords='axes fraction', fontsize=20, rotation=90, va='center')
            axarr[(1, i)].set_ylabel('')
    
    
    # Correlation
    for i, recs in enumerate(list_of_recs):
        axarr[(2, i)].plot(recs.groupby('pre_PRO')['waiting_time'].mean().rolling(4).mean(), c='black', linewidth=2)
        spearman = scipy.stats.spearmanr(recs['pre_PRO'], recs['waiting_time'], nan_policy='omit').correlation
        axarr[(2, i)].text(5, 15, f"Spearman r = {round(spearman, 3)}", fontsize=14)
        axarr[(2, i)].set_xlabel('Pre-intervention PRO')
        axarr[(2, i)].set_ylabel('Waiting Time')
        axarr[(2, i)].set_ylim(0, 24)
        axarr[(2, i)].spines['right'].set_visible(False)
        axarr[(2, i)].spines['top'].set_visible(False)
        axarr[(2, i)].spines['left'].set_visible(True)
        axarr[(2, i)].spines['bottom'].set_visible(True)
        if i == 1:
            axarr[(2, i)].annotate('Correlation', (-1.62, 0.5), xycoords='axes fraction', fontsize=20, rotation=90, va='center')
    
    # Lines
    lineh0 = plt.Line2D((0.06, 0.92),(0.08, 0.08), color="grey", linewidth=2)
    lineh1 = plt.Line2D((0.06, 0.92),(0.36, 0.36), color="grey", linewidth=2)
    lineh2 = plt.Line2D((0.06, 0.92),(0.64, 0.64), color="grey", linewidth=2)
    lineh3 = plt.Line2D((0.06, 0.92),(0.92, 0.92), color="grey", linewidth=2)
    linev0 = plt.Line2D((0.095, 0.095),(0.08, 0.97), color="grey", linewidth=0.5)
    linev1 = plt.Line2D((0.3, 0.3),(0.08, 0.97), color="grey", linewidth=0.5)
    linev2 = plt.Line2D((0.505, 0.505),(0.08, 0.97), color="grey", linewidth=0.5)
    linev3 = plt.Line2D((0.71, 0.71),(0.08, 0.97), color="grey", linewidth=0.5)
    linev4 = plt.Line2D((0.92, 0.92),(0.08, 0.97), color="grey", linewidth=0.5)
    fig.add_artist(lineh0)
    fig.add_artist(lineh1)
    fig.add_artist(lineh2)
    fig.add_artist(lineh3)
    fig.add_artist(linev0)
    fig.add_artist(linev1)
    fig.add_artist(linev2)
    fig.add_artist(linev3)
    fig.add_artist(linev4)
    
    plt.subplots_adjust(wspace=0.3, hspace=0.6)
    return fig
