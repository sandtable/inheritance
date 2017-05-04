import numpy as np
import pandas as pd
import scipy.sparse as sps

def assign_skill(skill_mother, skill_father, f_ea, rng):
    """Return education level on 0-4 int scale."""
    # TODO: Incorporate effect of mother's education
    # TODO: Check for sex differences
    # According to
    #     http://webarchive.nationalarchives.gov.uk/20160105160709/http://www.ons.gov.uk/ons/rel/household-income/intergenerational-transmission-of-poverty-in-the-uk---eu/2014/sty-causes-of-poverty-uk.html
    # the father's educational attainment is the strongest factor in deciding
    # the child's educational attainment. Children of fathers with low
    # attainment (defined here as skill <= 1, i.e. poor GCSEs or nothing) are
    # 7.5 times as likely to have low attainment as other children. The
    # parameter sigma=0.2 here has been calibrated to reproduce that
    sigma = 0.2
    skill_father_rank = np.zeros(len(skill_father))
    skill_father_rank[np.argsort(skill_father)] = np.linspace(0, 1, len(skill_father), endpoint=True)
    skill_child_rank = skill_father_rank + sigma * rng.randn(len(skill_father))
    skill_child_rank[np.argsort(skill_child_rank)] = np.linspace(0, 1, len(skill_child_rank))
    skill_child = np.zeros(len(skill_child_rank), int)-1
    limits = np.hstack((0, np.cumsum(f_ea[:, 0])))
    for i in xrange(5):
        skill_child[(skill_child_rank >= limits[i]) & (skill_child_rank <= limits[i+1])] = i
    return skill_child

def calculate_income(age, skill, female, boost, alive, i_as, i_as_age, i_e, i_e_edu, i_f):
    """Return income for each agent."""
    # Meritocracy?
    potential = np.interp(skill, i_e_edu, i_e)
    # No, patriarchy
    potential[~female] *= np.interp(age[~female], i_as_age, i_as[:, 0])
    potential[female] *= np.interp(age[female], i_as_age, i_as[:, 1])
    potential /= np.mean(potential)
    potential += boost
    earning = alive & (age >= 20) & (age < 65)
    n_earning = earning.sum()
    earning = np.where(earning)[0]
    earning = earning[np.argsort(potential[earning])]
    income_values = np.interp(np.linspace(0, 1, n_earning, endpoint=True),
                              np.linspace(0, 1, len(i_f), endpoint=True),
                              i_f)
    income = np.zeros(len(age))
    income[earning] = income_values
    return income

def assign_death(age, female, alive, X_tick, rng):
    """Return boolean of who is off to join the choir invisibule."""
    idx_sex = female.astype(int)
    idx_age = np.minimum(age, X_tick.shape[1]-1)
    mortality = X_tick[idx_sex, idx_age]
    return alive & (rng.rand(len(age)) < mortality)

def play_cupid(age, female, skill, wealth, partner, parents, alive, anniversaries,
               c_a, assortative_mating, rng):
    """Return updated partner array with new couplings."""
    # TODO: Prevent incest
    n_agent = len(age)
    partner_rate = c_a
    if len(partner_rate) < (age.max()+1):
        partner_rate = np.hstack((partner_rate, np.zeros(age.max()+1-len(partner_rate))))
    coupling = (alive & female & (partner == -1) & (rng.rand(n_agent) < partner_rate[age]))
    eligible = (alive & (~female) & (partner == -1) & (age >= 20))
    sigma_skill = 0.5 / assortative_mating
    sigma_wealth = 2.5e4 / assortative_mating
    difference = np.zeros((coupling.sum(), n_agent)) + np.inf
    difference[:, eligible] = (
        ((age[coupling, None] - age[None, eligible]) / 3.0)**2 +
        ((skill[coupling, None] - skill[None, eligible]) / sigma_skill)**2 +
        ((wealth[coupling, None] - wealth[None, eligible]) / sigma_wealth)**2)
    compatibility = np.exp(-0.5 * difference)
    match = multinomial(compatibility, unique=True, rng=rng)
    good_match = match >= 0
    coupling = np.where(coupling)[0]
    coupling = coupling[good_match]
    match = match[good_match]
    partner[coupling] = match
    partner[match] = coupling
    anniversaries[coupling] = 0
    anniversaries[match] = 0

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

def play_anticupid(partner, female, anniversaries, tick, D, rng):
    """Break up couples."""
    couples = np.where((partner >= 0) & female)[0]
    year_index = tick - anniversaries[couples]
    year_index[year_index >= 0] = -1
    year_index[year_index < -1*D.shape[0]] = 0
    anniv_index = anniversaries[couples].copy()
    anniv_index[anniv_index >= D.shape[1]] = -1
    rate = D[year_index, anniv_index]
    split_up = rng.rand(len(couples)) < rate
    anniversaries[couples[split_up]] = -1
    anniversaries[partner[couples[split_up]]] = -1
    partner[partner[couples[split_up]]] = -1
    partner[couples[split_up]] = -1

def assign_babies(age, female, partner, alive, fr_a, rng):
    """Return bool array of women having babies."""
    n_agent = len(age)
    if age.max() >= len(fr_a):
        # Extend fertility array, assuming fertility rate of 0
        fr_a = np.hstack((fr_a, np.zeros(age.max() + 1 - len(fr_a))))
    have_baby = alive & female & (partner >= 0) & (rng.rand(n_agent) < fr_a[age])
    return have_baby

def insert_babies(age, female, skill, wealth, parents, partner,
                  anniversaries, coupling, uncoupling, have_baby,
                  savings, interest, boost, alive, trust_amount,
                  tree, f_female_baby, f_ea, rng):
    n_agent = len(age)
    n_baby = have_baby.sum()

    new_female = (rng.rand(n_baby) < f_female_baby)
    new_skill = assign_skill(skill[have_baby], skill[partner[have_baby]], f_ea, rng)
    new_parents = np.array([np.arange(n_agent)[have_baby], partner[have_baby]], int).T

    age = np.hstack((age, np.zeros(n_baby, int)))
    female = np.hstack((female, new_female))
    skill = np.hstack((skill, new_skill))
    wealth = np.hstack((wealth, np.zeros(n_baby)))
    parents = np.vstack((parents, new_parents))
    partner = np.hstack((partner, np.zeros(n_baby, int)-1))
    anniversaries = np.hstack((anniversaries, np.zeros(n_baby, int)-1))

    coupling = np.hstack((coupling, np.zeros(n_baby, bool)))
    uncoupling = np.hstack((uncoupling, np.zeros(n_baby, bool)))
    have_baby = np.hstack((have_baby, np.zeros(n_baby, bool)))
    savings = np.hstack((savings, np.zeros(n_baby)))
    interest = np.hstack((interest, np.zeros(n_baby)))
    boost = np.hstack((boost, 0.7 * rng.randn(n_baby)))
    alive = np.hstack((alive, np.ones(n_baby, bool)))
    trust_amount = np.hstack((trust_amount, np.zeros(n_baby)))

    # Update the family tree
    idx = np.outer(np.arange(n_agent+n_baby), np.ones(2, int))
    descendant = idx[-n_baby:, :].ravel()
    ancestor = parents[-n_baby:, :].ravel()
    keep = ancestor >= 0
    tree_idx = 0
    while keep.sum() > 0:
        new_shape = (n_agent+n_baby, n_agent+n_baby)
        if tree_idx >= len(tree):
            # This number of generations has not been reached before
            # Append an empty matrix
            tree.append(sps.coo_matrix(
                ([], ([], [])), shape=new_shape))
        old = tree[tree_idx]
        descendant = descendant[keep]
        ancestor = ancestor[keep]
        data = np.ones(len(descendant), bool)
        tree[tree_idx] = sps.coo_matrix(
            (np.hstack((old.data, data)), 
             (np.hstack((old.row, ancestor)), np.hstack((old.col, descendant)))),
            shape=new_shape)
        ancestor = parents[ancestor, :].ravel()
        descendant = idx[descendant, :].ravel()
        keep = ancestor >= 0
        tree_idx += 1

    return (age, female, skill, wealth, parents, partner, anniversaries,
            coupling, uncoupling, have_baby, savings, interest, boost, alive,
            trust_amount, tree)

def execute_will(wealth, dying, partner, parents, alive, age, tree, trusts,
                 inheritance='direct', trust_levels=(2, 3)):
    """Update wealth based on inheritance."""
    n_agent = len(wealth)
    can_inherit = alive & ~dying
    n_children = np.histogram(parents[can_inherit, :].ravel(), bins=range(n_agent+1))[0]
    if inheritance == 'direct':
        direct_inheritance(wealth, dying, can_inherit, partner, parents, n_children)
    elif inheritance == 'trust':
        trust_inheritance(wealth, dying, can_inherit, partner, parents,
                          n_children, age, tree, trusts, levels=trust_levels)

def direct_inheritance(wealth, dying, can_inherit, partner, parents, n_children):
    # If you have children from previous relationship, split money between
    # current partner and previous children
    # Need to know number of children from current relationship, number from
    # previous relationship, and number of current partners (0 or 1)
    n_agent = len(wealth)
    parents_together = (can_inherit & (partner[parents[:, 1]] == parents[:, 0]) &
                        (can_inherit[parents[:, 0] | can_inherit[parents[:, 1]]]))
    n_children_current = np.histogram(parents[parents_together, :], bins=range(n_agent+1))[0]
    n_partner = ((partner >= 0) & can_inherit[partner]).astype(int)
    n_dependents = (n_children + n_partner).astype(float)
    # Fraction of wealth to each person in each category
    f_to_previous = dying * 1.0 / n_dependents
    f_to_current = dying * (1 - n_partner) / n_dependents
    f_to_partner = dying * (1 + n_children_current) / n_dependents
    for idx in (0, 1):
        from_previous = ((f_to_previous[parents[:, idx]] > 0) & (parents[:, idx] >= 0) &
                         (~parents_together) & can_inherit)
        wealth[from_previous] += (f_to_previous * wealth)[parents[from_previous, idx]]
        from_current = ((f_to_current[parents[:, idx]] > 0) & (parents[:, idx] >= 0) &
                        parents_together & can_inherit)
        wealth[from_current] += (f_to_current * wealth)[parents[from_current, idx]]
    from_partner = (f_to_partner[partner] > 0) & (partner >= 0) & can_inherit
    wealth[from_partner] += (f_to_partner * wealth)[partner[from_partner]]
    return

def trust_inheritance(wealth, dying, can_inherit, partner, parents, n_children, age, tree, trusts, levels=(2, )):
    # First give the partner 1 / (1 + n_children) of the wealth
    n_partner = ((partner >= 0) & can_inherit[partner]).astype(int)
    f_to_partner = dying * n_partner / (n_children + 1.0)
    from_partner = (f_to_partner[partner] > 0) & (partner >= 0) & can_inherit
    amount = (f_to_partner * wealth)[partner[from_partner]]
    wealth[from_partner] += amount
    wealth[partner[from_partner]] -= amount
    # Now set up trusts
    new_trusts = {
        'ancestor': np.hstack([np.where(dying)[0] for _ in levels]),
        'generations': np.hstack([np.tile(l, dying.sum()) for l in levels]),
        'amount': np.hstack([wealth[dying]/len(levels) for _ in levels]),
        'active': np.hstack([wealth[dying] > 0 for _ in levels]),
        'initial': np.hstack([wealth[dying]/len(levels) for _ in levels]),
    }
    # Immediate payout to existing adults
    beneficiaries = (age >= 20) & can_inherit
    future_beneficiaries = (age < 20) & can_inherit
    pay_from_trusts(new_trusts, wealth, tree, beneficiaries, future_beneficiaries)
    # Add them to the lists
    for key in trusts:
        trusts[key] = np.hstack((trusts[key], new_trusts[key]))
    return

def pay_from_trusts(trusts, wealth, tree, beneficiaries, future_beneficiaries):
    n_agent = len(wealth)
    for level in np.unique(trusts['generations']):
        if level > len(tree):
            # There are no relationships this deep
            continue
        this_level = (trusts['generations'] == level) & trusts['active']
        amount = np.zeros(n_agent)
        amount[trusts['ancestor'][this_level]] = trusts['amount'][this_level]
        n_beneficiaries = np.squeeze(np.asarray(tree[level-1].astype(int).dot(beneficiaries[:, None])))
        n_future = np.squeeze(np.asarray(tree[level-1].astype(int).dot(future_beneficiaries[:, None])))
        payout = amount / (n_beneficiaries + n_future)
        payout[(n_beneficiaries + n_future) == 0] = 0.0
        wealth[beneficiaries] += np.squeeze(np.asarray(tree[level-1].astype(int).T.dot(payout[:, None])))[beneficiaries]
        trusts['amount'][this_level] -= (n_beneficiaries * payout)[trusts['ancestor'][this_level]]
        trusts['active'] = (trusts['amount'] > 0)



def apply_to_descendants(base_pop, population, parents, level, func):
    for idx in (0, 1):
        print 'Matching at level', level
        print len(population), 'in population'
        parent = parents[population, idx]
        matched = (parent >= 0)
        print matched.sum(), 'matched'
        subset = population[matched]
        if level == 1:
            print 'Calling function'
            func(base_pop[subset], parent[subset])
        if level > 1:
            print 'Next level'
            apply_to_descendants(base_pop[subset], subset, parents, level-1, func)
    return

def count_descendants(population, parents, level):
    n_agent = len(parents)
    n_descendant = np.zeros(n_agent)
    population = np.where(population)[0]
    def increment(desc, ansc):
        n_descendant[:] += np.histogram(ansc, bins=range(n_agent+1))
    apply_to_descendants(
        population, population, parents, level, increment)
    return n_descendant

def payout(trusts, wealth, parents, beneficiaries, n_beneficiaries, n_future, level):
    n_total = n_beneficiaries + n_future
    this_level = (trusts['generations'] == level)
    # Amount left by each dead agent at this generational level
    to_pay = np.histogram(trusts['ancestor'][this_level], bins=range(len(wealth)+1),
                          weights=trusts['amount'][this_level])
    def pay(desc, ansc):
        wealth[desc] += (to_pay / n_total)[ansc]
    apply_to_descendants(
        beneficiaries, beneficiaries, parents, level, pay)


def update_partner_deaths(partner, dying):
    """Make widow(er)s single and update indices of partners."""
    leaving_widow = dying & (partner >= 0)
    partner[partner[leaving_widow]] = -1
    partner[leaving_widow] = -1
    return

def bury_dead(dying, partner, alive):
    """Update relationship status and vitality status."""
    update_partner_deaths(partner, dying)
    alive[dying] = False
    return

def check_partners(partner):
    """Assert basic properties of monogamous relationships."""
    # No duplicate values
    unique = len(np.unique(partner[partner >= 0])) == len(partner[partner >= 0])
    message = 'Duplicate values in partner'
    if not unique:
        n_partner = np.histogram(partner, bins=range(len(partner)+1))[0]
        duplicate = np.where(n_partner > 1)[0]
        for d in duplicate:
            message = message + ' [{} ({}): {}]'.format(
                d, partner[d], ','.join(['{}'.format(i) for i in np.where(partner == d)[0]]))
    assert unique, message
    # Reciprocal arrangement
    for idx in range(len(partner)):
        if partner[idx] >= 0:
            assert partner[partner[idx]] == idx, 'Non-reciprocal partnership'


def simulate(ctx, sink):

    rng = ctx.ctrl.random_state

    inheritance = ctx.parm.inheritance
    trust_levels = range(int(ctx.parm.trust_first_generation),
                         int(ctx.parm.trust_last_generation)+1)
    assortative_mating = ctx.parm.assortative_mating
    savings_rate = ctx.parm.savings_rate
    interest_rate = ctx.parm.interest_rate

    age = ctx.data.age
    female = ctx.data.female.astype(bool)
    skill = ctx.data.skill
    wealth = ctx.data.wealth
    partner = ctx.data.partner
    anniversaries = ctx.data.anniversaries
    parents = ctx.data.parents
    boost = ctx.data.boost
    alive = ctx.data.alive

    wealth_20 = np.sort(wealth[age == 20])

    f_as = ctx.data.f_as
    fr_a = ctx.data.fr_a
    D_yl = ctx.data.D_yl
    X_say = ctx.data.X_say
    c_a = ctx.data.c_a
    i_as = ctx.data.i_as
    i_as_age = ctx.meta.i_as.dims[0]
    i_e = ctx.data.i_e
    i_e_edu = ctx.meta.i_e.dims
    i_f = ctx.data.i_f
    f_ea = ctx.data.f_ea
    tree = ctx.data.tree
    trusts = {
        'ancestor': np.array([], dtype=int),
        'generations': np.array([], dtype=int),
        'amount': np.array([], dtype=float),
        'active': np.array([], dtype=bool),
        'initial': np.array([], dtype=float),
    }

    f_female_baby = f_as[0, 1] / f_as[0, :].sum()

    ctx.logger.info('number of agents: %s', len(age))
    ctx.logger.info('mean age: %s', age.mean())
    ctx.logger.info('%s women, %s men', female.sum(), len(female) - female.sum())
    ctx.logger.info('mean skill: %s', skill.mean())
    ctx.logger.info('mean wealth: %s', wealth.mean())
    ctx.logger.info('number partnered: %s', (partner >= 0).sum())
    ctx.logger.info('mean age of partnered: %s (female), %s (male)', age[(partner >= 0) & female].mean(), age[(partner >= 0) & ~female].mean())
    ctx.logger.info('number with parents: %s, %s', (parents[:, 0] >= 0).sum(), (parents[:, 1] >= 0).sum())

    n_tick = ctx.ctrl.ticks

    for tick in range(n_tick):

        # The inescapable passage of time
        age[alive] += 1
        anniversaries[alive & (anniversaries >= 0)] += 1

        # If your parents are rich, you'll already have some money by age 20
        coming_of_age = alive & (age == 20)
        wealth_parents = (
            (parents[coming_of_age, 0] >= 0) * wealth[parents[coming_of_age, 0]] +
            (parents[coming_of_age, 1] >= 0) * wealth[parents[coming_of_age, 1]])
        wealth_rank = np.zeros(len(wealth_parents))
        wealth_rank[np.argsort(wealth_parents)] = np.linspace(
            0, 1, len(wealth_rank), endpoint=True)
        wealth[coming_of_age] += np.interp(
            wealth_rank,
            np.linspace(0, 1, len(wealth_20), endpoint=True),
            wealth_20)

        # Calculate everyone's income
        income = calculate_income(age, skill, female, boost, alive, i_as, i_as_age, i_e, i_e_edu, i_f)

        # The rich get richer
        savings = savings_rate * income
        interest = interest_rate * wealth
        interest[age >= 65] = -0.01 * wealth[age >= 65]
        wealth = wealth + savings + interest
        trusts['amount'] *= (1.0 + interest_rate)

        # Pay out from trusts
        beneficiaries = (age == 20) & alive
        future_beneficiaries = (age < 20) & alive
        wealth_before = wealth.copy()
        pay_from_trusts(trusts, wealth, tree, beneficiaries, future_beneficiaries)
        trust_amount = wealth - wealth_before

        # Get hitched
        before_coupling = partner.copy()
        play_cupid(age, female, skill, wealth, partner, parents, alive,
                   anniversaries, c_a, assortative_mating, rng)
        coupling = (partner != before_coupling)

        # Get unhitched
        before_uncoupling = partner.copy()
        play_anticupid(partner, female, anniversaries, tick, D_yl, rng)
        uncoupling = (partner != before_uncoupling)

        # Sprogs
        have_baby = assign_babies(age, female, partner, alive, fr_a, rng)
        (age, female, skill, wealth, parents, partner, anniversaries,
         coupling, uncoupling, have_baby, savings, interest, boost, alive,
         trust_amount, tree) = insert_babies(
            age, female, skill, wealth, parents, partner, anniversaries,
            coupling, uncoupling, have_baby, savings, interest, boost, alive,
            trust_amount, tree, f_female_baby, f_ea, rng)

        # Retire
        wealth[(age == 65) & alive] *= 0.5

        # Pop clogs
        dying = assign_death(age, female, alive, X_say[:, :, tick], rng)

        # Perpetuate the inequalities of society
        wealth_before = wealth.copy()
        execute_will(wealth, dying, partner, parents, alive, age, tree, trusts,
                     inheritance=inheritance, trust_levels=trust_levels)
        direct_amount = wealth - wealth_before
        direct_amount[dying] = 0.0

        sink.write(tick, (age, female, skill, wealth, partner, coupling,
                          uncoupling, parents, savings, interest,
                          dying, have_baby, boost, alive, direct_amount,
                          trust_amount, tree, trusts))

        # Bury the dead
        bury_dead(dying, partner, alive)

        check_partners(partner)

        ctx.logger.info('tick %s', tick)
