# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

class Sink:

    def write(self, key, obj):
        raise Exception('Virtual function is not overriden')

    def flush(self):
        raise Exception('Virtual function is not overriden')


class CompositeSink(Sink):

    def __init__(self, sinks):
        self._sinks = sinks

    def write(self, key, obj):
        for sink in self._sinks:
            sink.write(key, obj)

    def flush(self):
        for sink in self._sinks:
            sink.flush()


class HDFSink(Sink):

    def __init__(self, ctx, file_path):
        self._file_path = file_path
        self._data = {}
        self._logger = ctx.logger

    def write(self, name, df, reset_index=True):
        self._data[name] = df.reset_index() if reset_index else df

    def flush(self):

        store = pd.HDFStore(self._file_path, complib='blosc', complevel=9)

        for name, data in self._data.iteritems():
            self._logger.info("saving dataset to hdf store '%s'", name)
            store[name] = data
        store.close()

        self._data = {}


class CommonSink(Sink):

    """
    The most general sink.
    """

    def __init__(self, ctx, store):
        self._ctx = ctx #model context
        self._store = store

        n_tick = int(ctx.ctrl.ticks)

        self.n_agent = np.zeros(n_tick, int)
        self.mean_age = np.zeros(n_tick)
        self.n_female = np.zeros(n_tick, int)
        self.skill_dist = np.zeros((5, n_tick), int)
        self.mean_wealth = np.zeros(n_tick)
        self.n_partnered = np.zeros(n_tick, int)
        self.n_coupling = np.zeros(n_tick, int)
        self.n_uncoupling = np.zeros(n_tick, int)
        self.skill_dist_new = np.zeros((5, n_tick), int)
        self.mean_savings = np.zeros(n_tick)
        self.mean_interest = np.zeros(n_tick)
        self.n_dying = np.zeros(n_tick, int)
        self.mean_age_dying = np.zeros(n_tick)
        self.n_baby = np.zeros(n_tick, int)
        self.mean_age_mother = np.zeros(n_tick)
        self.gini = np.zeros(n_tick)
        self.gini_adult = np.zeros(n_tick)
        self.gini_20_39 = np.zeros(n_tick)
        self.gini_40_64 = np.zeros(n_tick)
        self.gini_65p = np.zeros(n_tick)
        self.direct_amount = np.zeros(n_tick)
        self.n_direct = np.zeros(n_tick, int)
        self.trust_amount = np.zeros(n_tick)
        self.n_from_trust = np.zeros(n_tick, int)

        self.max_age = 150
        self.n_agent_by_age = np.zeros((n_tick, self.max_age), int)
        self.n_female_by_age = np.zeros((n_tick, self.max_age), int)
        self.mean_wealth_by_age = np.zeros((n_tick, self.max_age))
        self.n_partnered_by_age = np.zeros((n_tick, self.max_age), int)
        self.n_coupling_by_age = np.zeros((n_tick, self.max_age), int)
        self.n_uncoupling_by_age = np.zeros((n_tick, self.max_age), int)
        self.n_dying_by_age = np.zeros((n_tick, self.max_age), int)
        self.n_baby_by_age = np.zeros((n_tick, self.max_age), int)
        self.mean_savings_by_age = np.zeros((n_tick, self.max_age))
        self.mean_interest_by_age = np.zeros((n_tick, self.max_age))
        self.mean_children_by_age = np.zeros((n_tick, self.max_age))
        self.n_female_0_children = np.zeros((n_tick, self.max_age), int)
        self.n_female_1_children = np.zeros((n_tick, self.max_age), int)
        self.n_female_2_children = np.zeros((n_tick, self.max_age), int)
        self.n_female_3p_children = np.zeros((n_tick, self.max_age), int)
        self.direct_amount_by_age = np.zeros((n_tick, self.max_age))
        self.n_direct_by_age = np.zeros((n_tick, self.max_age), int)
        self.trust_amount_by_age = np.zeros((n_tick, self.max_age))
        self.n_from_trust_by_age = np.zeros((n_tick, self.max_age), int)

        self.tree = None

    def write(self, tick, obj):
        (age, female, skill, wealth, partner, coupling, uncoupling, parents,
         savings, interest, dying, have_baby, boost, alive, direct_amount,
         trust_amount, tree, trusts) = obj

        self.n_agent[tick] = alive.sum()
        self.mean_age[tick] = age[alive].mean()
        self.n_female[tick] = female[alive].sum()
        self.skill_dist[:, tick] = np.histogram(skill[alive], bins=range(6))[0]
        self.mean_wealth[tick] = wealth[alive].mean()
        self.n_partnered[tick] = (partner >= 0).sum()
        self.n_coupling[tick] = coupling.sum()
        self.n_uncoupling[tick] = uncoupling.sum()
        self.mean_savings[tick] = savings[alive].mean()
        self.mean_interest[tick] = interest[alive].mean()
        self.n_dying[tick] = dying.sum()
        self.mean_age_dying[tick] = age[dying].mean()
        self.n_baby[tick] = have_baby.sum()
        self.mean_age_mother[tick] = age[have_baby].mean()
        self.gini[tick] = gini(wealth[alive])
        self.gini_adult[tick] = gini(wealth[alive & (age >= 20)])
        self.gini_20_39[tick] = gini(wealth[alive & (age >= 20) & (age < 40)])
        self.gini_40_64[tick] = gini(wealth[alive & (age >= 40) & (age < 65)])
        self.gini_65p[tick] = gini(wealth[alive & age >= 65])
        self.direct_amount[tick] = direct_amount.sum()
        self.n_direct[tick] = (direct_amount > 0).sum()
        self.trust_amount[tick] = trust_amount.sum()
        self.n_from_trust[tick] = (trust_amount > 0).sum()

        bins = range(self.max_age+1)
        self.n_agent_by_age[tick, :] = np.histogram(age[alive], bins=bins)[0]
        self.n_female_by_age[tick, :] = np.histogram(age[alive & female], bins=bins)[0]
        self.mean_wealth_by_age[tick, :] = mean_by(wealth[alive], age[alive], 0, self.max_age)
        self.n_partnered_by_age[tick, :] = np.histogram(age[partner >= 0], bins=bins)[0]
        self.n_coupling_by_age[tick, :] = np.histogram(age[coupling], bins=bins)[0]
        self.n_uncoupling_by_age[tick, :] = np.histogram(age[uncoupling], bins=bins)[0]
        self.n_dying_by_age[tick, :] = np.histogram(age[dying], bins=bins)[0]
        self.n_baby_by_age[tick, :] = np.histogram(age[have_baby], bins=bins)[0]
        self.mean_savings_by_age[tick, :] = mean_by(savings[alive], age[alive], 0, self.max_age)
        self.mean_interest_by_age[tick, :] = mean_by(interest[alive], age[alive], 0, self.max_age)
        n_children = np.histogram(parents.ravel(), bins=range(len(age)+1))[0]
        self.mean_children_by_age[tick, :] = mean_by(n_children[alive], age[alive], 0, self.max_age)
        self.n_female_0_children[tick, :] = np.histogram(age[alive & female & (n_children == 0)], bins=bins)[0]
        self.n_female_1_children[tick, :] = np.histogram(age[alive & female & (n_children == 1)], bins=bins)[0]
        self.n_female_2_children[tick, :] = np.histogram(age[alive & female & (n_children == 2)], bins=bins)[0]
        self.n_female_3p_children[tick, :] = np.histogram(age[alive & female & (n_children >= 3)], bins=bins)[0]
        self.direct_amount_by_age[tick, :] = sum_by(direct_amount, age, 0, self.max_age)
        self.n_direct_by_age[tick, :] = np.histogram(age[direct_amount > 0], bins=bins)[0]
        self.trust_amount_by_age[tick, :] = sum_by(trust_amount, age, 0, self.max_age)
        self.n_from_trust_by_age[tick, :] = np.histogram(age[trust_amount > 0], bins=bins)[0]

        self.tree = tree

    def flush(self):
        sink = self._store
        ticks = pd.Series(range(len(self.n_agent)), name='tick')

        data = pd.DataFrame({
            'n_agent': self.n_agent,
            'mean_age': self.mean_age,
            'n_female': self.n_female,
            'mean_wealth': self.mean_wealth,
            'n_partnered': self.n_partnered,
            'n_coupling': self.n_coupling,
            'n_uncoupling': self.n_uncoupling,
            'mean_savings': self.mean_savings,
            'mean_interest': self.mean_interest,
            'n_dying': self.n_dying,
            'mean_age_dying': self.mean_age_dying,
            'n_baby': self.n_baby,
            'mean_age_mother': self.mean_age_mother,
            'gini': self.gini,
            'gini_adult': self.gini_adult,
            'gini_20_39': self.gini_20_39,
            'gini_40_64': self.gini_40_64,
            'gini_65p': self.gini_65p,
            'direct_amount': self.direct_amount,
            'n_direct': self.n_direct,
            'trust_amount': self.trust_amount,
            'n_from_trust': self.n_from_trust,
            }, index=ticks)
        data = data[['n_agent', 'mean_age', 'n_female', 'mean_wealth',
                     'n_partnered', 'n_coupling', 'n_uncoupling', 'mean_savings',
                     'mean_interest', 'n_dying', 'mean_age_dying', 'n_baby',
                     'mean_age_mother', 'gini', 'gini_adult', 'gini_20_39',
                     'gini_40_64', 'gini_65p', 'direct_amount', 'n_direct',
                     'trust_amount', 'n_from_trust']]
        sink.write('data', data)

        skill_label = ['skill_{}'.format(i) for i in xrange(5)]
        skill_dist = pd.DataFrame(self.skill_dist.T, columns=skill_label, index=ticks)
        sink.write('skill_dist', skill_dist)

        ticks_by_age = np.outer(ticks.values, np.ones(self.max_age, int))
        age_by_age = np.outer(np.ones(len(ticks), int), np.arange(self.max_age, dtype=int))
        by_age = pd.DataFrame({
            'tick': ticks_by_age.ravel(),
            'age': age_by_age.ravel(),
            'n_agent': self.n_agent_by_age.ravel(),
            'n_female': self.n_female_by_age.ravel(),
            'mean_wealth': self.mean_wealth_by_age.ravel(),
            'n_partnered': self.n_partnered_by_age.ravel(),
            'n_coupling': self.n_coupling_by_age.ravel(),
            'n_uncoupling': self.n_uncoupling_by_age.ravel(),
            'n_dying': self.n_dying_by_age.ravel(),
            'n_baby': self.n_baby_by_age.ravel(),
            'mean_savings': self.mean_savings_by_age.ravel(),
            'mean_interest': self.mean_interest_by_age.ravel(),
            'mean_children': self.mean_children_by_age.ravel(),
            'n_female_0_children': self.n_female_0_children.ravel(),
            'n_female_1_children': self.n_female_1_children.ravel(),
            'n_female_2_children': self.n_female_2_children.ravel(),
            'n_female_3p_children': self.n_female_3p_children.ravel(),
            'direct_amount': self.direct_amount_by_age.ravel(),
            'n_direct': self.n_direct_by_age.ravel(),
            'trust_amount': self.trust_amount_by_age.ravel(),
            'n_from_trust': self.n_from_trust_by_age.ravel(),
            })
        by_age = by_age[['tick', 'age', 'n_agent', 'n_female', 'mean_wealth',
                         'n_partnered', 'n_coupling', 'n_uncoupling',
                         'n_dying', 'n_baby', 'mean_savings', 'mean_interest',
                         'mean_children', 'n_female_0_children',
                         'n_female_1_children', 'n_female_2_children',
                         'n_female_3p_children', 'direct_amount', 'n_direct',
                         'trust_amount', 'n_from_trust']]
        sink.write('by_age', by_age, reset_index=False)

        df_tree = pd.DataFrame({
            'ancestor': np.hstack([t.row for t in self.tree]),
            'descendant': np.hstack([t.col for t in self.tree]),
            'generations': np.hstack([(i+1)*np.ones(len(t.row), int)
                                      for i, t in enumerate(self.tree)])
            })
        df_tree = df_tree[['descendant', 'ancestor', 'generations']]
        sink.write('tree', df_tree, reset_index=False)


def mean_by(values, group, min_val, max_val):
    result = np.zeros(max_val - min_val) + np.nan
    series = pd.Series(values).groupby(group).mean()
    series = series[(series.index.values >= min_val) &
                    (series.index.values < max_val)]
    result[series.index.values - min_val] = series.values
    return result

def sum_by(values, group, min_val, max_val):
    dtype = int if values.dtype == bool else values.dtype
    result = np.zeros(max_val - min_val, dtype=dtype)
    series = pd.Series(values).groupby(group).sum()
    series = series[(series.index.values >= min_val) &
                    (series.index.values < max_val)]
    result[series.index.values - min_val] = series.values
    return result

def gini(wealth):
    return 1.0 - (np.sort(wealth).cumsum() / wealth.sum()).sum() / (0.5 * len(wealth))




class IndividualSink(Sink):

    def __init__(self, ctx, store):
        self._ctx = ctx #model context
        self._store = store

        self.n_tick = int(ctx.ctrl.ticks)
        self.n_agent_0 = int(ctx.ctrl.agents)
        self.n_store = self.n_agent_0 * 2   # Allow for population growth

        self.age = np.zeros((self.n_tick, self.n_store), int)
        self.female = np.zeros((self.n_tick, self.n_store), bool)
        self.skill = np.zeros((self.n_tick, self.n_store), int)
        self.wealth = np.zeros((self.n_tick, self.n_store), float)
        self.partner = np.zeros((self.n_tick, self.n_store), int)
        self.coupling = np.zeros((self.n_tick, self.n_store), bool)
        self.uncoupling = np.zeros((self.n_tick, self.n_store), bool)
        self.mother = np.zeros((self.n_tick, self.n_store), int)
        self.father = np.zeros((self.n_tick, self.n_store), int)
        self.savings = np.zeros((self.n_tick, self.n_store), float)
        self.interest = np.zeros((self.n_tick, self.n_store), float)
        self.dying = np.zeros((self.n_tick, self.n_store), bool)
        self.have_baby = np.zeros((self.n_tick, self.n_store), bool)
        self.boost = np.zeros((self.n_tick, self.n_store), float)
        self.alive = np.zeros((self.n_tick, self.n_store), bool)
        self.direct_amount = np.zeros((self.n_tick, self.n_store), float)
        self.trust_amount = np.zeros((self.n_tick, self.n_store), float)

    def write(self, tick, obj):
        (age, female, skill, wealth, partner, coupling, uncoupling, parents,
         savings, interest, dying, have_baby, boost, alive, direct_amount,
         trust_amount, tree, trusts) = obj

        n_write = len(age)
        if n_write > self.n_store:
            # Need more memory
            n_new = int(0.5 * self.n_agent_0)
            for attr in ('age', 'female', 'skill', 'wealth', 'partner',
                         'coupling', 'uncoupling', 'mother', 'father',
                         'savings', 'interest', 'dying', 'have_baby',
                         'boost', 'alive', 'direct_amount', 'trust_amount'):
                old = getattr(self, attr)
                extra = np.zeros((self.n_tick, n_new), dtype=old.dtype)
                new = np.hstack((old, extra))
                setattr(self, attr, new)
            self.n_store += n_new

        self.age[tick, :n_write] = age
        self.female[tick, :n_write] = female
        self.skill[tick, :n_write] = skill
        self.wealth[tick, :n_write] = wealth
        self.partner[tick, :n_write] = partner
        self.coupling[tick, :n_write] = coupling
        self.uncoupling[tick, :n_write] = uncoupling
        self.mother[tick, :n_write] = parents[:, 0]
        self.father[tick, :n_write] = parents[:, 1]
        self.savings[tick, :n_write] = savings
        self.interest[tick, :n_write] = interest
        self.dying[tick, :n_write] = dying
        self.have_baby[tick, :n_write] = have_baby
        self.boost[tick, :n_write] = boost
        self.alive[tick, :n_write] = alive
        self.direct_amount[tick, :n_write] = direct_amount
        self.trust_amount[tick, :n_write] = trust_amount

    def flush(self):
        sink = self._store

        # Trim to correct size
        n_agent = np.where(self.alive.sum(axis=0) > 0)[0][-1]
        for attr in ('age', 'female', 'skill', 'wealth', 'partner',
                     'coupling', 'uncoupling', 'mother', 'father',
                     'savings', 'interest', 'dying', 'have_baby',
                     'boost', 'alive', 'direct_amount', 'trust_amount'):
            old = getattr(self, attr)
            new = old[:, :n_agent]
            setattr(self, attr, new)

        ticks = pd.Series(range(self.n_tick), name='tick')

        ticks_by_agents = np.outer(ticks.values, np.ones(n_agent, int))
        agents_by_agents = np.outer(np.ones(len(ticks), int), np.arange(n_agent, dtype=int))
        history = pd.DataFrame({
            'tick': ticks_by_agents.ravel(),
            'agent': agents_by_agents.ravel(),
            'age': self.age.ravel(),
            'female': self.female.ravel(),
            'skill': self.skill.ravel(),
            'wealth': self.wealth.ravel(),
            'partner': self.partner.ravel(),
            'coupling': self.coupling.ravel(),
            'uncoupling': self.uncoupling.ravel(),
            'mother': self.mother.ravel(),
            'father': self.father.ravel(),
            'savings': self.savings.ravel(),
            'interest': self.interest.ravel(),
            'dying': self.dying.ravel(),
            'have_baby': self.have_baby.ravel(),
            'boost': self.boost.ravel(),
            'alive': self.alive.ravel(),
            'direct_amount': self.direct_amount.ravel(),
            'trust_amount': self.trust_amount.ravel(),
        })
        history = history[['tick', 'agent', 'age', 'female', 'skill',
                           'wealth', 'partner', 'coupling', 'uncoupling',
                           'mother', 'father', 'savings',
                           'interest', 'dying', 'have_baby', 'boost',
                           'alive', 'direct_amount', 'trust_amount']]
        sink.write('history', history, reset_index=False)


class TrustsSink(Sink):

    def __init__(self, ctx, store):
        self._ctx = ctx #model context
        self._store = store

        self.n_tick = int(ctx.ctrl.ticks)
        self.n_agent_0 = int(ctx.ctrl.agents)
        self.n_store = 2 * self.n_agent_0   # Allow two trusts per initial agent

        self.amount = np.zeros((self.n_tick, self.n_store), float)
        self.initial = np.zeros((self.n_tick, self.n_store), float)
        self.generations = np.zeros((self.n_tick, self.n_store), int)
        self.ancestor = np.zeros((self.n_tick, self.n_store), int)
        self.active = np.zeros((self.n_tick, self.n_store), bool)

        self.n_write = 0

    def write(self, tick, obj):
        (age, female, skill, wealth, partner, coupling, uncoupling, parents,
         savings, interest, dying, have_baby, boost, alive, direct_amount,
         trust_amount, tree, trusts) = obj

        n_write = len(trusts['amount'])
        if n_write > self.n_store:
            # Need more memory
            n_new = int(0.5 * self.n_agent_0)
            for attr in ('amount', 'initial', 'generations', 'ancestor', 'active'):
                old = getattr(self, attr)
                extra = np.zeros((self.n_tick, n_new), dtype=old.dtype)
                new = np.hstack((old, extra))
                setattr(self, attr, new)
            self.n_store += n_new

        self.amount[tick, :n_write] = trusts['amount']
        self.initial[tick, :n_write] = trusts['initial']
        self.generations[tick, :n_write] = trusts['generations']
        self.ancestor[tick, :n_write] = trusts['ancestor']
        self.active[tick, :n_write] = trusts['active']

        self.n_write = n_write

    def flush(self):
        sink = self._store

        # Trim to correct size
        for attr in ('amount', 'initial', 'generations', 'ancestor', 'active'):
            old = getattr(self, attr)
            new = old[:, :self.n_write]
            setattr(self, attr, new)

        ticks = pd.Series(range(self.n_tick), name='tick')

        ticks_by_trusts = np.outer(ticks.values, np.ones(self.n_write, int))
        trusts_by_trusts = np.outer(np.ones(len(ticks), int), np.arange(self.n_write, dtype=int))
        trusts = pd.DataFrame({
            'tick': ticks_by_trusts.ravel(),
            'trust': trusts_by_trusts.ravel(),
            'amount': self.amount.ravel(),
            'initial': self.initial.ravel(),
            'generations': self.generations.ravel(),
            'ancestor': self.ancestor.ravel(),
            'active': self.active.ravel(),
        })

        trusts = trusts[['tick', 'trust', 'amount', 'initial', 'generations',
                           'ancestor', 'active']]
        sink.write('trusts', trusts, reset_index=False)

def plot_history(data, agent, agent_numbers=True):
    if isinstance(data, basestring):
        data = pd.HDFStore(os.path.join(data, 'output', 'output.h5'))
    history = data.history
    life = history[history.agent == agent].set_index('tick')
    life.index = 2016 + life.index
    plt.figure(figsize=(10, 5))
    plt.axes(position=[0.08, 0.1, 0.90, 0.87])
    ymax = 1.05*life.wealth.max() / 1000.0
    ymax = ymax if ymax > 0 else 1.0
    plt.fill_between(life.index, 0, ymax, color='black', alpha=0.2, where=(life.dying | ~life.alive))
    partner_list = np.unique(life.partner[(life.partner >= 0) & life.alive])
    # Babies will be added to with partner(s)'s babies
    baby_series = life.have_baby.copy()
    for partner in partner_list:
        in_relationship = (life.partner == partner).values
        partner_life = history[history.agent == partner].set_index('tick')
        partner_life.index = 2016 + partner_life.index
        baby_series = baby_series | (in_relationship & partner_life.have_baby)        
        # Extend out a year to include the final months of the relationship
        in_relationship[1:] = (~partner_life.dying[:-1] & in_relationship[:-1]) | in_relationship[1:]
        in_relationship = in_relationship & (life.alive | life.dying)
        plt.fill_between(life.index, 0, ymax, color='magenta', alpha=0.1, where=in_relationship)
    for coupling in life.index[life.coupling]:
        plt.plot([coupling, coupling], [0, ymax], color='red', linestyle=':', linewidth=1.5)
        label = 'partner'
        if agent_numbers:
            number = life.partner.loc[coupling]
            number = number if number >= 0 else '?'
            label = '{} ({})'.format(label, number)
        plt.text(coupling+0.3, 0.85*ymax, label, rotation=90, color='red')
    for uncoupling in life.index[life.uncoupling]:
        plt.plot([uncoupling, uncoupling], [0, ymax], color='red', linestyle=':', linewidth=1.5)
        label = 'separated'
        if agent_numbers:
            number = life.partner.loc[uncoupling] if uncoupling > 0 else -1
            number = number if number >= 0 else '?'
            label = '{} ({})'.format(label, number)
        plt.text(uncoupling+0.3, 0.85*ymax, label, rotation=90, color='red')
    for have_baby in life.index[baby_series]:
        plt.plot([have_baby, have_baby], [0, ymax], color='green', linestyle='--', linewidth=1.5)
        label = 'baby'
        if agent_numbers:
            baby = history.agent[
                ((history.mother == agent) | (history.father == agent)) &
                (history.age == 0) & history.alive &
                (history.tick == (have_baby-life.index[0]))].iloc[0]
            label = '{} ({})'.format(label, baby)
        plt.text(have_baby+0.3, 0.4*ymax, label, rotation=90, color='green')
    plt.plot(life.wealth[life.alive & ~life.dying] / 1000.0, linewidth=2.0, alpha=0.7)
    plt.xlim(life.index[0], life.index[-1])
    plt.ylim(0, ymax)
    plt.text(1+life.index[life.alive][0], 0.03*ymax, 'Age:')
    for age in range(10, 200, 10):
        this_age = (life.age == age)
        if np.any(this_age):
            year = life.index[this_age][0]
            tick = year - life.index[0]
            if tick >= 8:
                plt.text(year, 0.03*ymax, age, ha='center')
                plt.plot([year, year], [0, 0.022*ymax], color='black')
    label = 'Female' if np.any(life.female) else 'Male'
    if agent_numbers:
        label = '{}: {}'.format(agent, label)
    plt.text(2+life.index[0], 0.95*ymax, label)
    plt.xlabel('Year')
    plt.ylabel(u'Total wealth (£1000s)')
    plt.show()

def get_history(data, agent):
    if isinstance(data, basestring):
        data = pd.HDFStore(os.path.join(data, 'output', 'output.h5'))
    events = []
    history = data.history
    life = history[history.agent == agent].set_index('tick')
    mother = life.mother[life.alive].max()
    father = life.father[life.alive].max()
    born = (life.age == 0) & life.alive
    if np.any(born):
        tick = np.where(born)[0][0]
        events.append([tick, 'born', None])
    else:
        tick = 0
        events.append([tick, 'initial age', life.age.loc[0]])
    events.append([tick, 'mother', mother])
    events.append([tick, 'father', father])
    if np.any(life.dying):
        events.append([np.where(life.dying)[0][0], 'died', None])
    if np.any(life.coupling):
        for t in np.where(life.coupling)[0]:
            if life.uncoupling.loc[t]:
                # Damn, that was too fast to record
                events.append([t, 'coupled', None])
            else:
                events.append([t, 'coupled', life.partner.loc[t]])
    if life.alive.loc[0] and (life.partner.loc[0] >= 0) and not life.coupling.loc[0]:
        events.append([0, 'initial partner', life.partner.loc[0]])
    if np.any(life.uncoupling):
        for t in np.where(life.uncoupling)[0]:
            if life.coupling.loc[t]:
                # Too fast
                events.append([t, 'uncoupled', None])
            else:
                events.append([t, 'uncoupled', life.partner.loc[t-1]])
    babies = life.have_baby
    widows = pd.Series(np.zeros(len(life)))
    for partner in life.partner[(life.partner >= 0) & life.alive].unique():
        partner_life = history[(history.agent == partner) & (history.partner == agent)].set_index('tick')
        babies[partner_life.index] += partner_life.have_baby.values
        widows[partner_life.index] += partner_life.dying.values
    for tick in np.where(babies.values)[0]:
        baby = history.agent[((history.mother == agent) | (history.father == agent)) &
                             (history.age == 0) & history.alive & (history.tick == tick)].iloc[0]
        events.append([tick, 'baby', baby])
    for tick in np.where(widows.values)[0]:
        events.append([tick, 'widowed', None])
    for child in history.agent[((history.mother == agent) | (history.father == agent)) &
                               (history.age > 0) & (history.tick == 0)].values:
        events.append([0, 'existing child', child])
    return sorted(events, key=lambda x: x[0])

def generation_correlation(result, age, measure):
    agent = result.history[(result.history.age == age) & (result.history.tick > (result.history.tick.max()-10)) & result.history.alive & ~result.history.dying].agent.values
    measure = result.history[(result.history.age == age) & result.history.alive & ~result.history.dying].set_index('agent')[measure]
    tree = result.tree[result.tree.descendant.isin(agent) & result.tree.ancestor.isin(measure.index)]
    gencor = []
    for generations in xrange(1, tree.generations.max()+1):
        tree_cut = tree[tree.generations == generations]
        measure_ancestor = measure.loc[tree_cut.ancestor].groupby(tree_cut.descendant.values).mean()
        measure_descendant = measure.loc[measure_ancestor.index]
        gencor.append(np.corrcoef(measure_ancestor, measure_descendant)[0, 1])
    return gencor


    
