from collections import namedtuple
import numpy as np
from scipy import sparse as sps
# import pandas as pd


def choose_age(n_agent, f_as, rng):
    # Return age as int
    f_a = f_as.sum(axis=1)
    age = rng.choice(len(f_a), size=n_agent, p=f_a)
    return age

def choose_sex(age, f_as, rng):
    # Return True for women, False for men
    f_female = f_as[:, 1] / f_as.sum(axis=1)
    return rng.rand(len(age)) < f_female[age]

def choose_skill(age, female, f_ea, f_ea_age, f_es, rng):
    # Return skill on 0-4 int range
    # First set probabilities based on age
    prob = np.array([
        np.interp(age, f_ea_age, f_ea[i, :]) for i in xrange(len(f_ea))]).T
    # Ensure probabilities sum to 1
    prob /= prob.sum(axis=1, keepdims=True)
    # Then adjust based on sex
    prob_female = prob[female, :].mean(axis=0)
    prob_male = prob[~female, :].mean(axis=0)
    prob[female, :] *= f_es[:, 1] / prob_female
    prob[~female, :] *= f_es[:, 0] / prob_male
    # Ensure probabilities sum to 1
    prob /= prob.sum(axis=1, keepdims=True)
    return multinomial(prob, unique=False, rng=rng)

def choose_boost(n_agent, rng):
    """Return boost to agent's wealth/income potential."""
    return 0.7 * rng.randn(n_agent)

def choose_wealth(age, skill, female, partner, boost, w_a, w_a_age, w_e, w_e_edu,
                  mean_w, cw_f, cw_f_frac, rng):
    # Return assets in GBP
    # TODO: Male/female difference
    potential = np.interp(age, w_a_age, w_a) * np.interp(skill, w_e_edu, w_e)
    potential[partner >= 0] *= 2.0
    potential /= np.mean(potential)
    potential += boost
    breadwinner = (
        (age >= 20) &
        ((partner == -1) |
         (potential > potential[partner]) |
         ((potential == potential[partner]) & female)))
    n_breadwinner = breadwinner.sum()
    breadwinner = np.where(breadwinner)[0]
    breadwinner = breadwinner[np.argsort(potential[breadwinner])]
    total_wealth = mean_w * n_breadwinner
    cumulative_wealth = np.interp(np.linspace(0, 1, n_breadwinner, endpoint=True),
                                  cw_f_frac, cw_f * total_wealth)
    wealth_values = np.hstack((cumulative_wealth[0], cumulative_wealth[1:] - cumulative_wealth[:-1]))
    wealth = np.zeros(len(age))
    wealth[breadwinner] = wealth_values
    wealth[partner >= 0] = 0.5 * (wealth[partner >= 0] + wealth[partner[partner >= 0]])
    return wealth

def choose_partner(age, female, skill, f_cohabiting, assortative_mating, rng):
    # Return indices of partners (-1 for single) and of exes
    n_agent = len(age)
    partner_rate = f_cohabiting[:, 1]
    if len(partner_rate) < (age.max()+1):
        partner_rate = np.hstack((partner_rate, np.zeros(age.max()+1-len(partner_rate))))
    # coupling = female & (rng.rand(n_agent) < partner_rate[age])
    coupling = female & (age >= 20)
    eligible = ~female & (age >= 20)
    sigma_skill = 0.5 / assortative_mating
    difference = np.zeros((coupling.sum(), n_agent)) + np.inf
    difference[:, eligible] = (
        ((age[coupling, None] - age[None, eligible]) / 3.0)**2 +
        ((skill[coupling, None] - skill[None, eligible]) / sigma_skill)**2)
    compatibility = np.exp(-0.5 * difference)
    match = multinomial(compatibility, unique=True, rng=rng)
    # Take out those people who failed to find a partner
    coupling[np.where(coupling)[0][match == -1]] = False
    match = match[match >= 0]
    still_together = (rng.rand(coupling.sum()) < partner_rate[age[coupling]])
    coupling_together = np.where(coupling)[0][still_together]
    coupling_apart = np.where(coupling)[0][~still_together]
    partner = np.zeros(n_agent, int) - 1
    ex = np.zeros(n_agent, int) - 1
    partner[coupling_together] = match[still_together]
    partner[match[still_together]] = coupling_together
    ex[coupling_apart] = match[~still_together]
    ex[match[~still_together]] = coupling_apart
    # No duplicate values
    assert len(np.unique(partner[partner >= 0])) == len(partner[partner >= 0]), 'Duplicate values in partner'
    assert len(np.unique(ex[ex >= 0])) == len(ex[ex >= 0]), 'Duplicate values in ex'
    # Reciprocal arrangement
    for idx in range(n_agent):
        if partner[idx] >= 0:
            reciprocal = (partner[partner[idx]] == idx)
            message = 'Non-reciprocal partnership'
            if not reciprocal:
                message = message + ': {}, {}, {}'.format(idx, partner[idx], partner[partner[idx]])
            assert reciprocal, message
    for idx in range(n_agent):
        if ex[idx] >= 0:
            reciprocal = (ex[ex[idx]] == idx)
            message = 'Non-reciprocal exship'
            if not reciprocal:
                message = message + ': {}, {}, {}'.format(idx, ex[idx], ex[ex[idx]])
            assert reciprocal, message
    # Partnered or not, but not both
    in_partner = np.zeros(n_agent, bool)
    in_partner[partner[partner >= 0]] = True
    in_ex = np.zeros(n_agent, bool)
    in_ex[ex[ex >= 0]] = True
    assert not np.any(in_partner & in_ex), 'Scoundrel'
    return partner, ex

def multinomial(prob, unique=False, rng=np.random):
    """Return multinomial selection with different probabilities."""
    rechoose = np.where(prob.sum(axis=1) > 0.0)[0]
    choice = np.zeros(prob.shape[0], int) - 1
    while len(rechoose) > 0:
        prob = prob / prob.sum(axis=1, keepdims=True)
        rnd = rng.rand(len(rechoose))[:, None]
        choice[rechoose] = (rnd > prob[rechoose, :].cumsum(axis=1)).sum(axis=1)
        if unique:
            # Would be good to find a way to vectorise this
            prob[:, choice] = 0.0
            rechoose = np.array([idx for idx in xrange(len(choice))
                                 if np.any(choice[:idx] == choice[idx]) and
                                 (prob[idx, :].sum() > 0)])
            hamstrung = np.array([idx for idx in xrange(len(choice))
                                  if np.any(choice[:idx] == choice[idx]) and
                                  (prob[idx, :].sum() == 0)]).astype(int)
            choice[hamstrung] = -1
        else:
            rechoose = np.array([])
    return choice

def choose_anniversaries(age, female, partner, rng):
    """Return lengths of the relationships."""
    n_agent = len(age)
    anniversaries = np.zeros(n_agent, int) - 1
    female_couple = female & (partner >= 0)
    max_anniv = np.minimum(age[female_couple], age[partner[female_couple]]) - 16
    anniv = (rng.rand(female_couple.sum()) * max_anniv).astype(int)
    anniversaries[female_couple] = anniv
    anniversaries[partner[female_couple]] = anniv
    return anniversaries

def choose_parents(age, female, partner, ex, skill, wealth, ch_ac, rng):
    # Return indices of parents
    n_agent = len(age)
    parents = np.zeros((n_agent, 2), int) - 1
    partner_or_ex = np.maximum(partner, ex)
    n_children = multinomial(ch_ac[age[female]], rng=rng)
    # Standard deviation between father and child should be about 0.92
    # Given the distribution of skills, require a sigma in the probability of
    # ~1.0 to get the right output standard deviation
    # Values found by running a simulation with the correct assignment of
    # children's skill
    skill_parent = skill[female]
    father_alive = partner_or_ex[female] >= 0
    skill_parent[father_alive] = skill[partner_or_ex[female][father_alive]]
    prob = np.exp(-0.5 * ((skill_parent[:, None] - skill[None, :]) / 1.0)**2)
    # Need to make sure each parent's age is suitable
    age_difference_mother = age[female, None] - age[None, :]
    age_difference_father = np.zeros_like(age_difference_mother) + 30
    age_difference_father[father_alive, :] = (
        age[partner_or_ex[female][father_alive], None] - age[None, :])
    # Aiming for an average age of mother of ~28. This can be improved
    compatible = ((age_difference_mother >= 20) & (age_difference_mother < 36) &
                  (age_difference_father >= 20)).astype(float)
    prob *= compatible
    for i_children in range(1, n_children.max()+1):
        looking = (n_children >= i_children)
        match = multinomial(prob[looking, :], unique=True, rng=rng)
        parents[match, 0] = np.where(female)[0][looking]
        parents[match, 1] = partner_or_ex[parents[match, 0]]
        prob[:, match] = 0.0
    assert ~np.any(parents[:, 0] == np.arange(n_agent)), 'Someone is their own mother'
    assert ~np.any(parents[:, 1] == np.arange(n_agent)), 'Someone is their own father'
    return parents

def extend_divorce_rate(D, dims, start_year):
    """Extend (or cut) divorce rate to go up to year before start year."""
    latest_year = dims[0][-1]
    if latest_year >= start_year:
        D = D[dims[0] < start_year, :]
    else:
        D = np.vstack((D, np.outer(np.ones(int(start_year - latest_year - 1)), D[-1, :])))
    return D

def trim_death_rate(X, dims, start_year, ticks):
    """Trim death rate to correct range of years."""
    X = X[:, :, dims[2] >= start_year]
    if X.shape[2] > ticks:
        X = X[:, :, :ticks]
    elif X.shape[2] < ticks:
        X_old = X.copy()
        X = np.zeros((X.shape[0], X.shape[1], ticks))
        X[:, :, :X_old.shape[2]] = X_old
        for idx in xrange(X_old.shape[2], ticks):
            X[:, :, idx] = X_old[:, :, -1]
    return X

def make_family_tree(parents):
    """Return list of sparse matrices showing ancestors/descendants."""
    n_agent = len(parents)
    tree = []
    idx = np.outer(np.arange(len(parents)), np.ones(2, int))
    descendant = idx.ravel()
    ancestor = parents.ravel()
    keep = ancestor >= 0
    while keep.sum() > 0:
        descendant = descendant[keep]
        ancestor = ancestor[keep]
        data = np.ones(len(descendant), bool)
        tree.append(sps.coo_matrix((data, (ancestor, descendant)),
                                   shape=(n_agent, n_agent)))
        ancestor = parents[ancestor, :].ravel()
        descendant = idx[descendant, :].ravel()
        keep = ancestor >= 0
    return tree

def base(ctx):

    rng = ctx.ctrl.random_state
    logger = ctx.logger

    n_agent = int(ctx.ctrl.agents)
    ticks = int(ctx.ctrl.ticks)

    logger.info('initialising ages')
    age = choose_age(n_agent, ctx.data.f_as, rng)
    logger.info('initialising sexes')
    female = choose_sex(age, ctx.data.f_as, rng)
    logger.info('initialising education levels')
    skill = choose_skill(age, female, ctx.data.f_ea, ctx.meta.f_ea.dims[1], ctx.data.f_es, rng)
    logger.info('initialising partners')
    partner, ex = choose_partner(age, female, skill, ctx.data.C_as, ctx.parm.assortative_mating, rng)
    logger.info('initialising salary levels')
    boost = choose_boost(len(age), rng)
    logger.info('initialising wealth')
    wealth = choose_wealth(age, skill, female, partner, boost,
                           ctx.data.w_a, ctx.meta.w_a.dims,
                           ctx.data.w_e, ctx.meta.w_e.dims, ctx.data.w[0],
                           ctx.data.cw_f, ctx.meta.cw_f.dims, rng)
    logger.info('initialising relationship lengths')
    anniversaries = choose_anniversaries(age, female, partner, rng)
    logger.info('initialising parents')
    parents = choose_parents(age, female, partner, ex, skill, wealth, ctx.data.ch_ac, rng)
    logger.info('making family tree')
    tree = make_family_tree(parents)

    logger.info('reshaping data')
    D_yl = extend_divorce_rate(ctx.data.D_yl, ctx.meta.D_yl.dims, ctx.ctrl.start_year)
    X_say = trim_death_rate(ctx.data.X_say, ctx.meta.X_say.dims, ctx.ctrl.start_year, ticks)

    alive = np.ones(n_agent, dtype=bool)

    data = namedtuple('data', 'age female skill wealth partner anniversaries '
                      'parents boost D_yl X_say alive tree')

    return data(age, female, skill, wealth, partner, anniversaries, parents,
                boost, D_yl, X_say, alive, tree)

