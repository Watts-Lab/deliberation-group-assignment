from scipy import stats
from numpy.random import default_rng

class ParticipantStream():
    def __init__(self, rng=None, seed=None):
        if rng is not None:
            self.rng = rng
        else:
            self.rng = default_rng(seed)
        self.party_dist = {'elements': ['Republican', 'Democrat', 'Independent'], 'probabilities': [0.35, 0.45, 0.2]}
        self.age_dist = stats.truncnorm(a=((19 - 45) / 15), b=((80 - 45) / 15), loc=45, scale=15)
        self.gender_dist = {'elements': ['M', 'F'], 'probabilities': [0.55, 0.45]}
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