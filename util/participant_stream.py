from scipy import stats
import numpy as np

class ParticipantStream():
    def __init__(self, rng=None, seed=None):
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng(seed)
        self.party_dist = {'elements': ['Republican', 'Democrat', 'Independent'], 'probabilities': [0.35, 0.45, 0.2]}
        party_map = {'Republican': 0, 'Democrat': 1, 'Independent': 0.5}
        self.party_map_func = np.vectorize(party_map.get)
        self.age_dist = stats.truncnorm(a=((19 - 45) / 15), b=((80 - 45) / 15), loc=45, scale=15)
        self.age_map_func = lambda arr: (arr - 19) / 61
        self.gender_dist = {'elements': ['M', 'F'], 'probabilities': [0.55, 0.45]}
        gender_map = {'M': 0, 'F': 1}
        self.gender_map_func = np.vectorize(gender_map.get)
        self.arrival_time_dist = stats.lognorm(s=1, scale=20)
        self.departure_time_dist = stats.lognorm(s=1, scale=20)

    def generate_participants(self, n):
        party_s = self.rng.choice(a=self.party_dist['elements'], size=n, p=self.party_dist['probabilities'])
        age_s = self.age_dist.rvs(size=n, random_state=self.rng)
        gender_s = self.rng.choice(a=self.gender_dist['elements'], size=n, p=self.gender_dist['probabilities'])
        arrival_s = sorted(self.arrival_time_dist.rvs(size=n, random_state=self.rng))
        departure_s = self.departure_time_dist.rvs(size=n, random_state=self.rng)
        return [
            {
                'party': party_s[i],
                'age': int(age_s[i]),
                'gender': gender_s[i],
                'arrival_time': round(arrival_s[i]),
                'departure_time': round(departure_s[i] + arrival_s[i])
            }
            for i in range(n)
        ]
    
    def generate_participants_normalized(self, n):
        party_s = self.rng.choice(a=self.party_dist['elements'], size=n, p=self.party_dist['probabilities'])
        age_s = self.age_dist.rvs(size=n, random_state=self.rng).astype(int)
        gender_s = self.rng.choice(a=self.gender_dist['elements'], size=n, p=self.gender_dist['probabilities'])
        arrival_s = sorted(self.arrival_time_dist.rvs(size=n, random_state=self.rng))
        departure_s = self.departure_time_dist.rvs(size=n, random_state=self.rng)
        party_sn = self.party_map_func(party_s)
        age_sn = self.age_map_func(age_s)
        gender_sn = self.gender_map_func(gender_s)
        return [
            {
                'party': party_s[i],
                'age': age_s[i],
                'gender': gender_s[i],
                'arrival_time': round(arrival_s[i]),
                'departure_time': round(departure_s[i] + arrival_s[i]),
                'n_party': party_sn[i],
                'n_age': age_sn[i],
                'n_gender': gender_sn[i]
            }
            for i in range(n)
        ]