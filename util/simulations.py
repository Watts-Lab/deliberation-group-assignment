import numpy as np
from sklearn.neighbors import KernelDensity
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

#------------------------------------------------
# Algo 1 Simulation
#------------------------------------------------

def simulate_algo_1(rng, age_pdf):
    stream = ParticipantStream(rng=rng)
    datapoints = None
    data_kde = KernelDensity(bandwidth=7.625, kernel='gaussian')
    batches = 0
    scores = {"linear": [], "squared": []}

    def generate_batch(n):
        return stream.generate_participants(n)
    
    def add_batch(samples: np.ndarray):
        # assuming samples come in the form of [[p1, p2], [p1, p2], ...]
        nonlocal datapoints, data_kde, batches
        if datapoints is not None:
            datapoints = np.vstack([datapoints, samples])
        else:
            datapoints = np.copy(samples)
        # Pick kernel that has width = 10% of dimension domain
        data_kde.fit(datapoints)
        batches += 1
    
    def global_pdf(samples: np.ndarray):
        nonlocal datapoints, data_kde
        if datapoints is not None:
            return np.exp(data_kde.score_samples(samples))
        return np.zeros(samples.shape[0])
    
    def run_batch():
        pool = generate_batch(100)
        # extract age feature from participant metadata
        ages = np.fromiter((p['age'] for p in pool), dtype=int)
        # sample possible pairs to build kde
        sampled_pairs = rng.choice(ages, (10000, 2))
        sample_pdf = KernelDensity(bandwidth=7.625, kernel='gaussian').fit(sampled_pairs)
        # get all possible pairings given pool
        all_pairs = np.array(list(combinations(ages, 2)))
        # compute probability of the pairings occuring, adjusted by what's already in batch
        all_pairs_prob = np.exp(sample_pdf.score_samples(all_pairs)) + global_pdf(all_pairs) 
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
        for pair in best_pairs:
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
        grid_x, grid_y = np.arange(19, 81), np.arange(19, 81)
        x_coords, y_coords = np.meshgrid(grid_x, grid_y)
        pos = np.vstack([x_coords.ravel(), y_coords.ravel()]).T
        # sample every bin of kde
        dif = np.reshape((age_pdf(pos.T[0], pos.T[1]) - global_pdf(pos)), x_coords.shape)

        return {"linear": np.sum(np.tril(np.abs(dif))), "squared": np.sum(np.tril(np.abs(dif ** 2)))}
    
    for i in range(100):
        run_batch()
        res = evaluate_dataset()
        scores["linear"].append(res['linear'])
        scores["squared"].append(res['squared'])
    evaluate_dataset()
    return scores

#------------------------------------------------
# Algo 2 Simulation
#------------------------------------------------

def simulate_algo_2(rng, age_pdf):
    stream = ParticipantStream(rng=rng)
    datapoints = None
    data_kde = KernelDensity(bandwidth=7.625, kernel='gaussian')
    batches = 0
    scores = {"linear": [], "squared": []}

    def generate_batch(n):
        return stream.generate_participants(n)
    
    def add_batch(samples: np.ndarray):
        # assuming samples come in the form of [[p1, p2], [p1, p2], ...]
        nonlocal datapoints, data_kde, batches
        if datapoints is not None:
            datapoints = np.vstack([datapoints, samples])
        else:
            datapoints = np.copy(samples)
        # Pick kernel that has width = 10% of dimension domain
        data_kde.fit(datapoints)
        batches += 1
    
    def global_pdf(samples: np.ndarray):
        nonlocal datapoints, data_kde
        if datapoints is not None:
            return np.exp(data_kde.score_samples(samples))
        return np.zeros(samples.shape[0])
    
    def run_batch():
        nonlocal datapoints
        pool = generate_batch(100)
        # extract age feature from participant metadata
        ages = np.fromiter((p['age'] for p in pool), dtype=int)
        # get map of unique ages to participants with age
        ptcp_cnt = {}
        for i, age in enumerate(ages):
            if age not in ptcp_cnt:
                ptcp_cnt[age] = [pool[i]]
            else:
                ptcp_cnt[age].append(pool[i])
        ptcp_unique = np.array(list(ptcp_cnt.keys()))
        # compute rarity of each person
        ptcp_pdf = KernelDensity(bandwidth=6.2, kernel='gaussian').fit(ages.reshape(-1, 1))
        
        groupings = []
        age_grouping = []
        # if no previous data, then group in order of rareness
        if datapoints is None:
            ptcp_rareness = np.exp(ptcp_pdf.score_samples(ages.reshape(-1,1)))
            ranked_idx = np.argsort(ptcp_rareness)
            for i in range(0, len(pool), 2):
                i1 = ranked_idx[i]
                i2 = ranked_idx[i+1]
                p1 = pool[i1]['age']
                p2 = pool[i2]['age']
                if p1 < p2:
                    groupings.append((pool[i1], pool[i2]))
                    age_grouping.append([p1, p2])
                else:
                    groupings.append((pool[i2], pool[i1]))
                    age_grouping.append([p2, p1])
        # otherwise, fix rarest person, then select most useful group with that person
        else:
            ptcp_rareness = np.exp(ptcp_pdf.score_samples(ptcp_unique.reshape(-1,1)))
            ptcp_unique = np.take(ptcp_unique, np.argsort(ptcp_rareness), axis=0)
            while ptcp_unique.shape[0] > 0:
                # select rarest feature set
                ptcp = ptcp_unique[0]
                # get all possible remaining pairs in order of least frequency in dataset (most undersampled)
                pairs = np.array([[ptcp, y] for y in ptcp_unique])
                pairs_prob = global_pdf(pairs)
                best_pairs = np.take(pairs, np.argsort(pairs_prob), axis=0)
                # greedily pick pairs until ptcp with selected features are exhausted
                bp_idx = 0
                while ptcp_cnt[ptcp]:
                    # pick first person
                    p1 = ptcp_cnt[ptcp].pop()
                    # try current most useful sample
                    pair = best_pairs[bp_idx]
                    # if sample is not available, remove from candidates
                    if not ptcp_cnt[pair[1]]:
                        best_pairs = np.delete(best_pairs, bp_idx, axis=0)
                        if bp_idx >= best_pairs.shape[0]:
                            bp_idx = 0
                        continue
                    # add group and set pointer to next most useful sample
                    groupings.append((p1, ptcp_cnt[pair[1]].pop()))
                    age_grouping.append(pair.tolist())
                    bp_idx = bp_idx + 1 if bp_idx + 1 < best_pairs.shape[0] else 0
                
                if not ptcp_cnt[ptcp]:
                    ptcp_unique = ptcp_unique[1:]

        # update sampled dataset
        add_batch(np.array(age_grouping))
    
    def evaluate_dataset():
        grid_x, grid_y = np.arange(19, 81), np.arange(19, 81)
        x_coords, y_coords = np.meshgrid(grid_x, grid_y)
        pos = np.vstack([x_coords.ravel(), y_coords.ravel()]).T
        # sample every bin of kde
        dif = np.reshape((age_pdf(pos.T[0], pos.T[1]) - global_pdf(pos)), x_coords.shape)

        return {"linear": np.sum(np.tril(np.abs(dif))), "squared": np.sum(np.tril(np.abs(dif ** 2)))}
    
    for i in range(100):
        run_batch()
        res = evaluate_dataset()
        scores["linear"].append(res['linear'])
        scores["squared"].append(res['squared'])
    evaluate_dataset()
    return scores

#------------------------------------------------
# Algo 3 Simulation
#------------------------------------------------

def simulate_algo_3(rng, age_pdf):
    stream = ParticipantStream(rng=rng)
    datapoints = None
    data_kde = KernelDensity(bandwidth=7.625, kernel='gaussian')
    batches = 0
    scores = {"linear": [], "squared": []}

    def generate_batch(n):
        return stream.generate_participants(n)
    
    def add_batch(samples: np.ndarray):
        # assuming samples come in the form of [[p1, p2], [p1, p2], ...]
        nonlocal datapoints, data_kde, batches
        if datapoints is not None:
            datapoints = np.vstack([datapoints, samples])
        else:
            datapoints = np.copy(samples)
        # Pick kernel that has width = 10% of dimension domain
        data_kde.fit(datapoints)
        batches += 1
    
    def global_pdf(samples: np.ndarray):
        nonlocal datapoints, data_kde
        if datapoints is not None:
            return np.exp(data_kde.score_samples(samples))
        return np.zeros(samples.shape[0])
    
    def run_batch():
        nonlocal datapoints
        pool = generate_batch(100)
        # extract age feature from participant metadata
        ages = np.fromiter((p['age'] for p in pool), dtype=int)
        # get map of unique ages to participants with age
        ptcp_cnt = {}
        for i, age in enumerate(ages):
            if age not in ptcp_cnt:
                ptcp_cnt[age] = [pool[i]]
            else:
                ptcp_cnt[age].append(pool[i])
        ptcp_unique = np.array(list(ptcp_cnt.keys()))
        # compute rarity of each person
        ptcp_pdf = KernelDensity(bandwidth=6.2, kernel='gaussian').fit(ages.reshape(-1, 1))
        
        groupings = []
        age_grouping = []
        # group in order of rareness
        ptcp_rareness = np.exp(ptcp_pdf.score_samples(ages.reshape(-1,1)))
        ranked_idx = np.argsort(ptcp_rareness)
        for i in range(0, len(pool), 2):
            i1 = ranked_idx[i]
            i2 = ranked_idx[i+1]
            p1 = pool[i1]['age']
            p2 = pool[i2]['age']
            if p1 < p2:
                groupings.append((pool[i1], pool[i2]))
                age_grouping.append([p1, p2])
            else:
                groupings.append((pool[i2], pool[i1]))
                age_grouping.append([p2, p1])

        # update sampled dataset
        add_batch(np.array(age_grouping))
    
    def evaluate_dataset():
        grid_x, grid_y = np.arange(19, 81), np.arange(19, 81)
        x_coords, y_coords = np.meshgrid(grid_x, grid_y)
        pos = np.vstack([x_coords.ravel(), y_coords.ravel()]).T
        # sample every bin of kde
        dif = np.reshape((age_pdf(pos.T[0], pos.T[1]) - global_pdf(pos)), x_coords.shape)

        return {"linear": np.sum(np.tril(np.abs(dif))), "squared": np.sum(np.tril(np.abs(dif ** 2)))}
    
    for i in range(100):
        run_batch()
        res = evaluate_dataset()
        scores["linear"].append(res['linear'])
        scores["squared"].append(res['squared'])
    evaluate_dataset()
    return scores

#------------------------------------------------
# Baseline Simulation
#------------------------------------------------

def simulate_baseline(rng, age_pdf):
    b_stream = ParticipantStream(rng=rng)
    b_samples = None
    b_kde = KernelDensity(bandwidth=7.625, kernel='gaussian')
    b_scores = {'linear': [], 'squared': []}

    def run_baseline_batch(size):
        nonlocal b_samples, b_kde
        size = size >> 1 << 1
        b_pool = b_stream.generate_participants(size)
        bs = np.zeros((size >> 1, 2))
        for i in range(0, len(b_pool), 2):
            if b_pool[i]['age'] < b_pool[i+1]['age']:
                bs[i//2, 0] = b_pool[i]['age']
                bs[i//2, 1] = b_pool[i+1]['age']
            else:
                bs[i//2, 0] = b_pool[i+1]['age']
                bs[i//2, 1] = b_pool[i]['age']
        if b_samples is not None:
            b_samples = np.vstack([b_samples, bs])
        else:
            b_samples = np.copy(bs)
        b_kde.fit(b_samples)

    def evaluate_baseline_batch(plot = False):
        b_x_coords, b_y_coords = np.meshgrid(np.arange(19, 81), np.arange(19, 81))
        b_pos = np.vstack([b_x_coords.ravel(), b_y_coords.ravel()]).T
        b_dif = np.reshape((age_pdf(b_pos.T[0], b_pos.T[1]) - np.exp(b_kde.score_samples(b_pos))), b_x_coords.shape)

        return {"linear": np.sum(np.tril(np.abs(b_dif))), "squared": np.sum(np.tril(np.abs(b_dif ** 2)))}
    
    # batch 100
    for i in range(100):
        run_baseline_batch(100)
        res = evaluate_baseline_batch()
        b_scores['linear'].append(res['linear'])
        b_scores['squared'].append(res['squared'])
    evaluate_baseline_batch()
    return b_scores

#------------------------------------------------
# Run Full Simulation
#------------------------------------------------

def simulate_once(seed_num):
    print(f"Initialized simulation with seed {seed_num}")
    rng = np.random.default_rng(seed=seed_num)
    age_pdf = generate_theoretical_pdf(rng)
    a1s = simulate_algo_1(rng, age_pdf)
    a2s = simulate_algo_2(rng, age_pdf)
    a3s = simulate_algo_3(rng, age_pdf)
    bs = simulate_baseline(rng, age_pdf)
    print(f"Finished simulation with seed {seed_num}")
    return (a1s, a2s, a3s, bs)

#------------------------------------------------
# Driver Code
#------------------------------------------------

def main():
    seeds = np.random.randint(1000, 9999, 100)
    baseline_scores = {'linear': np.empty((100, 100)), 'squared': np.empty((100, 100))}
    algo_1_scores = {'linear': np.empty((100, 100)), 'squared': np.empty((100, 100))}
    algo_2_scores = {'linear': np.empty((100, 100)), 'squared': np.empty((100, 100))}
    algo_3_scores = {'linear': np.empty((100, 100)), 'squared': np.empty((100, 100))}
    
    start_time = time.time()
    with Pool(16) as worker_pool:
        for i, (a1s, a2s, a3s, bs) in enumerate(worker_pool.map(simulate_once, seeds)):
            algo_1_scores['linear'][i] = a1s['linear']
            algo_1_scores['squared'][i] = a1s['squared']
            algo_2_scores['linear'][i] = a2s['linear']
            algo_2_scores['squared'][i] = a2s['squared']
            algo_3_scores['linear'][i] = a3s['linear']
            algo_3_scores['squared'][i] = a3s['squared']
            baseline_scores['linear'][i] = bs['linear']
            baseline_scores['squared'][i] = bs['squared']
    print("Pool execution took %s seconds" % (time.time() - start_time))
    
    np.savez_compressed(f'simulation_results/simulation_scores_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.npz', 
                        baseline_linear=baseline_scores['linear'], 
                        baseline_squared=baseline_scores['squared'], 
                        algo1_linear=algo_1_scores['linear'], 
                        algo1_squared=algo_1_scores['squared'],
                        algo2_linear=algo_2_scores['linear'], 
                        algo2_squared=algo_2_scores['squared'],
                        algo3_linear=algo_3_scores['linear'], 
                        algo3_squared=algo_3_scores['squared'])
    df_index = pd.MultiIndex.from_product([['baseline', 'algo_1', 'algo_2', 'algo_3'], ['linear', 'squared']], names=['simulation', 'score_type'])
    df_data = np.column_stack((baseline_scores['linear'].flatten(), 
                              baseline_scores['squared'].flatten(), 
                              algo_1_scores['linear'].flatten(),
                              algo_1_scores['squared'].flatten(),
                              algo_2_scores['linear'].flatten(),
                              algo_2_scores['squared'].flatten(),
                              algo_3_scores['linear'].flatten(),
                              algo_3_scores['squared'].flatten()))
    df = pd.DataFrame(df_data, columns=df_index)
    df.to_csv(f'simulation_results/simulation_scores_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv')

if __name__ == '__main__':
    main()