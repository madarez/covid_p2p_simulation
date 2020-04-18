import simpy
import math
import copy
import datetime
import itertools
import numpy as np
from collections import defaultdict
from orderedset import OrderedSet
import copy
from event import Event
from config import *
from utils import compute_distance, _get_random_area
from track import Tracker

class Env(simpy.Environment):

    def __init__(self, initial_timestamp):
        super().__init__()
        self.initial_timestamp = initial_timestamp

    def time(self):
        return self.now

    @property
    def timestamp(self):
        return self.initial_timestamp + datetime.timedelta(
            minutes=self.now * TICK_MINUTE)

    def minutes(self):
        return self.timestamp.minute

    def hour_of_day(self):
        return self.timestamp.hour

    def day_of_week(self):
        return self.timestamp.weekday()

    def is_weekend(self):
        return self.day_of_week() in [0, 6]

    def time_of_day(self):
        return self.timestamp.isoformat()


class City(object):

    def __init__(self, env, n_people, rng, x_range, y_range, start_time, init_percent_sick, Human, sim_days):
        self.env = env
        self.rng = rng
        self.x_range = x_range
        self.y_range = y_range
        self.total_area = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0])
        self.n_people = n_people
        self.start_time = start_time
        self.init_percent_sick = init_percent_sick
        self.sim_days=sim_days
        print("Initializing locations ...")
        self.initialize_locations()

        self.humans = []
        self.households = OrderedSet()
        print("Initializing humans ...")
        self.initialize_humans(Human)

        self.log_static_info()

        print("Computing their preferences")
        self._compute_preferences()
        self.tracker = Tracker(env, self)

    def create_location(self, specs, type, name, area=None):
        _cls = Location
        if type in ['household', 'senior_residency']:
            _cls = Household
        if type == 'hospital':
            _cls = Hospital

        return   _cls(
                        env=self.env,
                        rng=self.rng,
                        name=f"{type}:{name}",
                        location_type=type,
                        lat=self.rng.randint(*self.x_range),
                        lon=self.rng.randint(*self.y_range),
                        area=area,
                        social_contact_factor=specs['social_contact_factor'],
                        capacity= None if not specs['rnd_capacity'] else self.rng.randint(*specs['rnd_capacity']),
                        surface_prob = specs['surface_prob']
                        )

    def initialize_locations(self):
        for location, specs in LOCATION_DISTRIBUTION.items():
            if location in ['household']:
                continue

            n = math.ceil(self.n_people/specs["n"])
            area = _get_random_area(n, specs['area'] * self.total_area, self.rng)
            locs = [self.create_location(specs, location, i, area[i]) for i in range(n)]
            setattr(self, f"{location}s", locs)

    def initialize_humans(self, Human):
        # allocate humans to houses such that (unsolved)
        # 1. average number of residents in a house is (approx.) 2.6
        # 2. not all residents are below 15 years of age
        # 3. age occupancy distribution follows HUMAN_DSITRIBUTION.residence_preference.house_size

        # current implementation is an approximate heuristic

        # make humans
        count_humans = 0
        house_allocations = {2:[], 3:[], 4:[], 5:[]}
        n_houses = 0
        for age_bin, specs in HUMAN_DISTRIBUTION.items():
            n = math.ceil(specs['p'] * self.n_people)
            ages = self.rng.randint(*age_bin, size=n)

            senior_residency_preference = specs['residence_preference']['senior_residency']

            professions = ['healthcare', 'school', 'others', 'retired']
            p = [specs['profession_profile'][x] for x in professions]
            profession = self.rng.choice(professions, p=p, size=n)

            for i in range(n):
                count_humans += 1
                age = ages[i]

                # residence
                res = None
                if self.rng.random() < senior_residency_preference:
                    res = self.rng.choice(self.senior_residencys)
                # workplace
                if profession[i] == "healthcare":
                    workplace = self.rng.choice(self.hospitals + self.senior_residencys)
                elif profession[i] == 'school':
                    workplace = self.rng.choice(self.schools)
                elif profession[i] == 'others':
                    type_of_workplace = self.rng.choice([0,1,2], p=OTHERS_WORKPLACE_CHOICE, size=1)[0]
                    type_of_workplace = [self.workplaces, self.stores, self.miscs][type_of_workplace]
                    workplace = self.rng.choice(type_of_workplace)
                else:
                    workplace = res

                self.humans.append(Human(
                        env=self.env,
                        rng=self.rng,
                        name=count_humans,
                        age=age,
                        household=res,
                        workplace=workplace,
                        profession=profession[i],
                        rho=0.1,
                        gamma=0.21,
                        infection_timestamp=self.start_time if self.rng.random() < self.init_percent_sick else None,
                        sim_days=self.sim_days
                        )
                    )

        # assign houses
        # stores tuples - (location, current number of residents, maximum number of residents allowed)
        remaining_houses = []
        for human in self.humans:
            if human.household is not None:
                continue
            if len(remaining_houses) == 0:
                cap = self.rng.choice(range(1,6), p=HOUSE_SIZE_PREFERENCE, size=1)
                x = self.create_location(LOCATION_DISTRIBUTION['household'], 'household', len(self.households))

                remaining_houses.append((x, cap))

            # get_best_match
            res = None
            for  c, (house, n_vacancy) in enumerate(remaining_houses):
                new_avg_age = (human.age + sum(x.age for x in house.residents))/(len(house.residents) + 1)
                if new_avg_age > MIN_AVG_HOUSE_AGE:
                    res = house
                    n_vacancy -= 1
                    if n_vacancy == 0:
                        remaining_houses = remaining_houses[:c] + remaining_houses[c+1:]
                    break

            if res is None:
                for i, (l,u) in enumerate(HUMAN_DISTRIBUTION.keys()):
                    if l <= human.age < u:
                        bin = (l,u)
                        break

                house_size_preference = HUMAN_DISTRIBUTION[(l,u)]['residence_preference']['house_size']
                cap = self.rng.choice(range(1,6), p=house_size_preference, size=1)
                res = self.create_location(LOCATION_DISTRIBUTION['household'], 'household', len(self.households))
                if cap - 1 > 0:
                    remaining_houses.append((res, cap-1))

            # FIXME: there is some circular reference here
            res.residents.append(human)
            human.assign_household(res)
            self.households.add(res)

        # assign area to house
        area = _get_random_area(len(self.households), LOCATION_DISTRIBUTION['household']['area'] * self.total_area, self.rng)
        for i,house in enumerate(self.households):
            house.area = area[i]

    def log_static_info(self):
        for h in self.humans:
            Event.log_static_info(self, h, self.env.timestamp)

    @property
    def events(self):
        return list(itertools.chain(*[h.events for h in self.humans]))

    def pull_events(self):
        return list(itertools.chain(*[h.pull_events() for h in self.humans]))

    def _compute_preferences(self):
        """ compute preferred distribution of each human for park, stores, etc."""
        for h in self.humans:
            h.stores_preferences = [(compute_distance(h.household, s) + 1e-1) ** -1 for s in self.stores]
            h.parks_preferences = [(compute_distance(h.household, s) + 1e-1) ** -1 for s in self.parks]


class Location(simpy.Resource):

    def __init__(self, env, rng, area, name, location_type, lat, lon,
            social_contact_factor, capacity, surface_prob):

        if capacity is None:
            capacity = simpy.core.Infinity

        super().__init__(env, capacity)
        self.humans = OrderedSet() #OrderedSet instead of set for determinism when iterating
        self.name = name
        self.rng = rng
        self.lat = lat
        self.lon = lon
        self.area = area
        self.location_type = location_type
        self.social_contact_factor = social_contact_factor
        self.env = env
        self.contamination_timestamp = datetime.datetime.min
        self.contaminated_surface_probability = surface_prob
        self.max_day_contamination = 0

    def infectious_human(self):
        return any([h.is_infectious for h in self.humans])

    def __repr__(self):
        return f"{self.name} - occ:{len(self.humans)}/{self.capacity} - I:{self.infectious_human()}"

    def add_human(self, human):
        self.humans.add(human)
        if human.is_infectious:
            self.contamination_timestamp = self.env.timestamp
            rnd_surface = float(self.rng.choice(a=MAX_DAYS_CONTAMINATION, size=1, p=self.contaminated_surface_probability))
            self.max_day_contamination = max(self.max_day_contamination, rnd_surface)

    def remove_human(self, human):
        self.humans.remove(human)

    @property
    def is_contaminated(self):
        return self.env.timestamp - self.contamination_timestamp <= datetime.timedelta(days=self.max_day_contamination)

    @property
    def contamination_probability(self):
        if self.is_contaminated:
            lag = (self.env.timestamp - self.contamination_timestamp)
            lag /= datetime.timedelta(days=1)
            p_infection = 1 - lag / self.max_day_contamination # linear decay; &envrionmental_contamination
            return self.social_contact_factor * p_infection
        return 0.0

    def __hash__(self):
        return hash(self.name)

    def serialize(self):
        """ This function serializes the location object"""
        s = self.__dict__
        if s.get('env'):
            del s['env']
        if s.get('rng'):
            del s['rng']
        if s.get('_env'):
            del s['_env']
        if s.get('contamination_timestamp'):
            del s['contamination_timestamp']
        if s.get('residents'):
            del s['residents']
        if s.get('humans'):
            del s['humans']
        return s

class Household(Location):
    def __init__(self, **kwargs):
        super(Household, self).__init__(**kwargs)
        self.residents = []


class Hospital(Location):
    ICU_AREA = 0.10
    ICU_CAPACITY = 0.10
    def __init__(self, **kwargs):
        env = kwargs.get('env')
        rng = kwargs.get('rng')
        capacity = kwargs.get('capacity')
        name = kwargs.get("name")
        lat = kwargs.get('lat')
        lon = kwargs.get('lon')
        area = kwargs.get('area')
        surface_prob = kwargs.get('surface_prob')
        social_contact_factor = kwargs.get('social_contact_factor')

        super(Hospital, self).__init__( env=env,
                                        rng=rng,
                                        area=area * (1-self.ICU_AREA),
                                        name=name,
                                        location_type="hospital",
                                        lat=lat,
                                        lon=lon,
                                        social_contact_factor=social_contact_factor,
                                        capacity=int(capacity* (1- self.ICU_CAPACITY)),
                                        surface_prob=surface_prob,
                                        )
        self.location_contamination = 1
        self.icu = ICU( env=env,
                        rng=rng,
                        area=area * (self.ICU_AREA),
                        name=f"{name}-icu",
                        location_type="hospital-icu",
                        lat=lat,
                        lon=lon,
                        social_contact_factor=social_contact_factor,
                        capacity=int(capacity* (self.ICU_CAPACITY)),
                        surface_prob=surface_prob,
                        )

    def add_human(self, human):
        human.obs_hospitalized = True
        super().add_human(human)

    def remove_human(self, human):
        human.obs_hospitalized = False
        super().remove_human(human)


class ICU(Location):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def add_human(self, human):
        human.obs_hospitalized = True
        human.obs_in_icu = True
        super().add_human(human)

    def remove_human(self, human):
        human.obs_hospitalized = False
        human.obs_in_icu = False
        super().remove_human(human)

