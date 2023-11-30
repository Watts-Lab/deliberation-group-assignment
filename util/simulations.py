import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations
from multiprocessing import Pool
from participant_stream import ParticipantStream
from datetime import datetime
import time

def generate_theoretical_pdf(rng: np.random.Generator):
    def continuous_uniform_range_pair(lower_bound, upper_bound, num_samples):
        samples = rng.uniform(lower_bound, upper_bound, (2, num_samples))
        for i in range(num_samples):
            while samples[0,i] == samples[1,i]:
                samples[0,i] = rng.uniform(lower_bound, upper_bound)
                samples[1,i] = rng.uniform(lower_bound, upper_bound)
            if samples[0,i] > samples[1, i]:
                samples[0,i], samples[1,i] = samples[1,i], samples[0,i]
        return samples
    
    def generate_hist2d_pdf(hist, x_edges, y_edges):
        def sample_pdf(x, y):
            x_bin = np.argmax(np.where((np.expand_dims(x_edges, axis=1) < x).transpose(), x_edges, 0), axis=1)
            y_bin = np.argmax(np.where((np.expand_dims(y_edges, axis=1) < y).transpose(), y_edges, 0), axis=1)
            return hist[x_bin, y_bin]
        return sample_pdf
    
    theory_num_samples = 50000
    theory_samples = continuous_uniform_range_pair(19, 80, theory_num_samples)
    bins = 60
    t_hist, t_younger_edges, t_older_edges = np.histogram2d(theory_samples[0], theory_samples[1], bins=bins, range=[[19,80],[19,80]], density=True)
    age_pdf = generate_hist2d_pdf(t_hist, t_younger_edges, t_older_edges)
    return age_pdf

def simulate_algo_1(rng, age_pdf):
    stream = ParticipantStream(rng=rng)
    datapoints = None
    data_kde = None
    batches = 0
    scores = {"linear": [], "squared": []}

    def generate_batch(n):
        return stream.generate_participants(n)
    
    def add_batch(samples: np.ndarray):
        # assuming samples come in the form of [[p1, p2], [p1, p2], ...]
        nonlocal datapoints, data_kde, batches
        if datapoints is not None:
            datapoints = np.hstack([datapoints, samples.transpose()])
        else:
            datapoints = np.copy(samples.transpose())
        # Pick kernel that has width = 10% of dimension domain
        data_kde = stats.gaussian_kde(datapoints)
        batches += 1
    
    def global_pdf(samples: np.ndarray):
        nonlocal datapoints, data_kde
        if datapoints is not None:
            return data_kde(samples)
        return np.zeros(samples.shape[1])
    
    def run_batch():
        pool = generate_batch(100)
        # extract age feature from participant metadata
        ages = np.fromiter((p['age'] for p in pool), dtype=int)
        # sample possible pairs to build kde
        sampled_pairs = rng.choice(ages, (2, 10000))
        sample_pdf = stats.gaussian_kde(sampled_pairs)
        # get all possible pairings given pool
        all_pairs = np.array(list(combinations(ages, 2)))
        # compute probability of the pairings occuring, adjusted by what's already in batch
        all_pairs_prob = sample_pdf(all_pairs.transpose()) + global_pdf(all_pairs.transpose())
        # sort all pairs from lowest probability to highest probability
        best_pairs = np.take(all_pairs, np.argsort(all_pairs_prob), axis=0)
        # Greedily pick pairs to sample
        age_map = {}
        for p in pool:
            if p['age'] not in age_map:
                age_map[p['age']] = [p]
            else:
                age_map[p['age']].append(p)
        grouping = []
        age_grouping = []
        for pair in all_pairs:
            p1 = int(pair[0])
            p2 = int(pair[1])
            if p1 > p2:
                p1, p2 = p2, p1
            if p1 not in age_map or p2 not in age_map:
                continue
            if p1 == p2 and len(age_map[p1]) < 2:
                continue
            grouping.append((age_map[p1].pop(), age_map[p2].pop()))
            age_grouping.append([p1, p2])
            if len(age_map[p1]) == 0:
                del age_map[p1]
            if p2 != p1 and len(age_map[p2]) == 0:
                del age_map[p2]
            if not age_map:
                break
        # update sampled dataset
        add_batch(np.array(age_grouping))
    
    def evaluate_dataset():
        x_coords, y_coords = np.mgrid[19:80, 19:80]
        pos = np.vstack([x_coords.ravel(), y_coords.ravel()])
        
        dif = np.reshape((age_pdf(pos[0], pos[1]) - data_kde(pos)).T, x_coords.shape)

        return {"linear": np.sum(np.triu(np.abs(dif))), "squared": np.sum(np.triu(np.abs(dif ** 2)))}
    
    for i in range(100):
        run_batch()
        res = evaluate_dataset()
        scores["linear"].append(res['linear'])
        scores["squared"].append(res['squared'])
    evaluate_dataset()
    return scores

def simulate_baseline(rng, age_pdf):
    b_stream = ParticipantStream(rng=rng)
    b_samples = None
    b_kde = None
    b_scores = {'linear': [], 'squared': []}

    def run_baseline_batch(size):
        nonlocal b_samples, b_kde
        size = size >> 1 << 1
        b_pool = b_stream.generate_participants(size)
        bs = np.zeros((2, size >> 1))
        for i in range(0, len(b_pool), 2):
            if b_pool[i]['age'] < b_pool[i+1]['age']:
                bs[0,i//2] = b_pool[i]['age']
                bs[1,i//2] = b_pool[i+1]['age']
            else:
                bs[0,i//2] = b_pool[i+1]['age']
                bs[1,i//2] = b_pool[i]['age']
        if b_samples is not None:
            b_samples = np.hstack([b_samples, bs])
        else:
            b_samples = bs
        b_kde = stats.gaussian_kde(b_samples)
    
    def evaluate_baseline_batch(plot = False):
        b_x_coords, b_y_coords = np.mgrid[19:80, 19:80]
        b_pos = np.vstack([b_x_coords.ravel(), b_y_coords.ravel()])
        b_dif = np.reshape((age_pdf(b_pos[0], b_pos[1]) - b_kde(b_pos)).T, b_x_coords.shape)

        return {"linear": np.sum(np.triu(np.abs(b_dif))), "squared": np.sum(np.triu(np.abs(b_dif ** 2)))}
    
    # batch 100
    for i in range(100):
        run_baseline_batch(100)
        res = evaluate_baseline_batch()
        b_scores['linear'].append(res['linear'])
        b_scores['squared'].append(res['squared'])
    evaluate_baseline_batch()
    return b_scores

def simulate_once(seed_num):
    print(f"Initialized simulation with seed {seed_num}")
    rng = np.random.default_rng(seed=seed_num)
    age_pdf = generate_theoretical_pdf(rng)
    a1s = simulate_algo_1(rng, age_pdf)
    bs = simulate_baseline(rng, age_pdf)
    print(f"Finished simulation with seed {seed_num}")
    return (a1s, bs)

def main():
    seeds = np.random.randint(1000, 9999, 100)
    baseline_scores = {'linear': np.empty((100, 100)), 'squared': np.empty((100, 100))}
    algo_1_scores = {'linear': np.empty((100, 100)), 'squared': np.empty((100, 100))}
    
    start_time = time.time()
    with Pool(16) as worker_pool:
        for i, (a1s, bs) in enumerate(worker_pool.map(simulate_once, seeds)):
            algo_1_scores['linear'][i] = a1s['linear']
            algo_1_scores['squared'][i] = a1s['squared']
            baseline_scores['linear'][i] = bs['linear']
            baseline_scores['squared'][i] = bs['squared']
    print("Pool execution took %s seconds" % (time.time() - start_time))
    
    np.savez_compressed(f'simulation_scores_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.npz', 
                        baseline_linear=baseline_scores['linear'], 
                        baseline_squared=baseline_scores['squared'], 
                        algo1_linear=algo_1_scores['linear'], 
                        algo1_squared=algo_1_scores['squared'])
    df_index = pd.MultiIndex.from_product([['baseline', 'algo_1'], ['linear', 'squared']], names=['simulation', 'score_type'])
    df_data = np.column_stack((baseline_scores['linear'].flatten(), 
                              baseline_scores['squared'].flatten(), 
                              algo_1_scores['linear'].flatten(),
                             algo_1_scores['squared'].flatten()))
    df = pd.DataFrame(df_data, columns=df_index)
    df.to_csv(f'simulation_scores_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv')

if __name__ == '__main__':
    main()