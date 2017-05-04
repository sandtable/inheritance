# -*- encoding: utf8 -*-

# import sandgit2.api as sg
import pandas as pd
import numpy as np
import os

DATA_DIR = 'data'

INPUT_DIR = 'raw_data'

def save_tensor(name, description, tensor, dims, data_dir):
    """Save tensor and metadata to a numpy file."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    np.savez('{}/{}.npz'.format(data_dir, name),
             name=str(name),
             desc=str(description),
             array=tensor,
             dims=np.asarray(dims))

def generate_data(name, description, fn, export=True, data_dir=DATA_DIR):
    """Generate a tensor and its metadata, save to file if requested."""
    tnsr, dims = fn()
    print "{0}\t@{1:16}\t{2}".format(name, tnsr.shape, description)
    if tnsr.shape != tuple(map(len, dims)):
        print "tensor={}, dims={}".format(tnsr.shape, tuple(map(len, dims)))
    assert np.all(tnsr.shape == tuple(map(len, dims)))
    if export:
        save_tensor(name, description, tnsr, dims, data_dir)
    return tnsr, dims

def n_age_sex():
    """Return DataFrame of population split by age and sex."""
    path = os.path.join(INPUT_DIR, 'census/sex_by_age.xlsx')
    data = pd.read_excel(path, 'Sheet 1', header=10, skipfooter=3, index_col=0)
    data = data[2:][['Males', 'Females']].reset_index(drop=True)
    return data

def f_age_sex():
    """Return DataFrame of fraction of population at each age and sex."""
    data = n_age_sex()
    data /= data.values.sum()
    return data.values, (data.index.values, ('male', 'female'))

def fertility_rate():
    """Return female fertility rate as a function of age."""
    path = os.path.join(INPUT_DIR, 'ONS/birthsbyparentscharacteristics2014final.xls')
    data = pd.read_excel(path, 'Table 6')
    raw_rate = data.loc[34, 'Unnamed: 11':'Unnamed: 17'].values # Live births within marriage/civil partnership per 1,000 women, 2014
    raw_rate /= 1000.0
    rate = np.zeros(100)
    rate[15:20] = raw_rate[0]
    rate[20:25] = raw_rate[1]
    rate[25:30] = raw_rate[2]
    rate[30:35] = raw_rate[3]
    rate[35:40] = raw_rate[4]
    rate[40:45] = raw_rate[5]
    rate[45:50] = raw_rate[6] # 45 and over.
    return rate, (np.arange(len(rate)), )

def f_married(year=2015):
    """Return married fraction (age X sex)."""
    path = os.path.join(INPUT_DIR, 'ONS/2015releasetablesewonlyinccidatav1.0.xls')
    age_key = np.zeros(100, dtype='S10')
    age_key[:16] = '0-15'
    age_key[16:20] = '16-19'
    for start in range(20, 85, 5):
        age_key[start:start+5] = '{}-{}'.format(start, start+4)
    age_key[85:] = '85+'
    result = []
    for header in (91, 179): # Males, Females
        data = pd.read_excel(path, 'Table 1 Marital Status', header=header, index_col=[0, 1])[:85]
        data = data[year]
        data = data.replace(['u', 'z', '.'], 0.0)
        n_dict = {status: data.loc[[(status, k) for k in age_key]] for status in
                  ('Single', 'Married', 'Civil Partnered', 'Divorced', 'Widowed')}
        married = n_dict['Married'].values + n_dict['Civil Partnered'].values
        unmarried = n_dict['Single'].values + n_dict['Divorced'].values + n_dict['Widowed'].values
        frac = (married / (married + unmarried))
        result.append(frac)
    return np.array(result).T, (np.arange(100), ('male', 'female'))

def f_cohabiting(year=2015):
    """Return cohabiting fraction (age X sex)."""
    path = os.path.join(INPUT_DIR, 'ONS/2015releasetablesewonlyinccidatav1.0.xls')
    age_key = np.zeros(100, dtype='S10')
    age_key[:16] = '0-15'
    age_key[16:30] = '16-29'
    for start in range(30, 70, 5):
        age_key[start:start+5] = '{}-{}'.format(start, start+4)
    age_key[70:] = '70+'
    result = []
    for header in (66, 129): # Males, Females
        data = pd.read_excel(path, 'Table 2 Living Arrangements', header=header, index_col=[0, 1])[:60]
        data = data[year]
        data = data.replace(['u', 'z', '.'], 0.0)
        n_dict = {status_key: data.loc[[(status, k) for k in age_key]] for status_key, status in
                  {'married': 'Living in a couple: Married or civil partnered',
                   'cohabiting': 'Living in a couple: Cohabiting - never married or civil partnered',
                   'cohabiting_previous': 'Living in a couple: Cohabiting- previously married or civil partnered',
                   'single': 'Not living in a couple: Never married or civil partnered',
                   'single_previous': 'Not living in a couple: Previously married or civil partnered'}.items()}
        coupled = n_dict['married'].values + n_dict['cohabiting'].values + n_dict['cohabiting_previous'].values
        single = n_dict['single'].values + n_dict['single_previous'].values
        frac = (coupled / (coupled + single))
        result.append(frac)
    return np.array(result).T, (np.arange(100), ('male', 'female'))

def coupling_rate():
    """Return fraction of women by age who will form a couple that year."""
    path_marriage = os.path.join(INPUT_DIR, 'ONS/cohabitationandcohortanalyses11.xls')
    path_births = os.path.join(INPUT_DIR, 'ONS/birthsbyparentscharacteristics2014final.xls')
    n_female = n_age_sex().values[:, 1]
    n_female *= (1.0 - f_married(2011)[0][:len(n_female), 1]) # Comparing to number of unmarried women
    n_age = len(n_female)
    n_marriage = pd.read_excel(path_marriage, 'Table 1', header=None)
    n_marriage = n_marriage.values[30:43, 2]
    edges = [16] + range(20, 80, 5) + [85]
    marriage_rate = np.zeros(n_age)
    for idx in xrange(len(edges)-1):
        marriage_rate[edges[idx]:edges[idx+1]] = (
            n_marriage[idx] / n_female[edges[idx]:edges[idx+1]].sum())
    # 47.7% of births are outside of marriage, so boost the coupling rate accordingly
    # Strong age dependence
    births = pd.read_excel(path_births, 'Table 2', header=7, index_col=1)[:10]
    births = births.loc[['Within Marriage/Civil Partnership1', 'Outside Marriage/Civil Partnership1']]
    births.index = ['within', 'outside']
    births = births.drop('Unnamed: 0', axis=1)
    married_fraction = np.zeros_like(marriage_rate)
    edges = [16] + range(20, 45, 5) + [85]
    for idx in xrange(len(edges)-1):
        married_fraction[edges[idx]:edges[idx+1]] = (
            births.loc['within'].values[idx] / births.sum().values[idx])
    return marriage_rate / married_fraction, (np.arange(n_age), )

def divorce_rate():
    """Return array of divorce rate as function of marriage year and duration."""
    path = os.path.join(INPUT_DIR, 'ONS/ageatmarriagedurationofmarriageandcohortanalysestcm77424213.xls')
    cumulative = pd.read_excel(path, 'Table 2', header=6)
    cumulative.index = [int(i[:4]) if isinstance(i, basestring) else i for i in cumulative.index.get_level_values(0)]
    cumulative = cumulative.loc[1963:2012]
    # For future predictions, assume divorce rate at anniversary X is the same
    # as for the most recent cohort to have data for that anniversary
    rate = cumulative.T.diff().T.fillna(method='pad')
    # Rate in first year is equal to total in first year
    rate[1] = cumulative[1]
    # Nobody has got divorced the instant they got married
    rate[0] = 0.0
    rate = rate[sorted(rate.columns)]
    # Convert to fractional rate
    rate /= 100.0
    return rate.values, (rate.index.values, rate.columns)

def death_rate():
    """Return array of death rates as function of year and age."""
    path = os.path.join(INPUT_DIR, 'ONS/wukprincipal14qx.xls')
    rate = []
    for sheet in ('Males period qx', 'Females period qx'):
        raw_rate = pd.read_excel(path, sheet, header=6, index_col=0)[:101]
        rate.append(raw_rate.values / 100000.0)
    return np.array(rate), (('male', 'female'), raw_rate.index.values, raw_rate.columns)

def n_children():
    """Return array of fraction of women with 0, 1, 2, 3, 4+ children."""
    path = os.path.join(INPUT_DIR, 'ONS/cohortfertility2014v2_tcm77-422295.xls')
    data = pd.read_excel(path, 'Table 3', header=7)
    data = data[np.isfinite(data.index.get_level_values(0))]
    # Fill in blank "age" values
    index = pd.DataFrame(zip(*data.index.values)).T
    index = index.replace('45 3', 45).fillna(method='pad')
    index = pd.MultiIndex.from_tuples([tuple(x) for x in index.values])
    year_idx = index.get_level_values(0)
    age_idx = index.get_level_values(1)
    # Fraction of women with each number of children as a function of age
    result = np.zeros((100, len(data.columns)-1))
    max_age = age_idx.max()
    current_year = year_idx.max() + age_idx.min()
    # Do each number of children separately
    for idx, key in enumerate(data.columns[:-1]):
        # For women <45, take the fraction from the most recent cohort to reach that age
        fraction = []
        for age in np.unique(age_idx):
            this_age = (age_idx == age)
            fraction.append(data[this_age][key].values[year_idx[this_age].argmax()])
        left = 100.0 if idx == 0 else 0.0 # Assume nobody gives birth at age <20
        result[:int(max_age), idx] = np.interp(np.arange(max_age), np.unique(age_idx), fraction, left=left)
        # For women >= 45, take the fractions at age 45 for their birth years
        fraction = []
        for year in np.unique(year_idx):
            this_year = (year_idx == year)
            fraction.append(data[this_year][key].values[age_idx[this_year].argmax()])
        result[int(max_age):, idx] = np.interp(np.arange(max_age, 100), current_year - np.unique(year_idx)[::-1], fraction[::-1])
    result /= 100.0 # percentages to fractions
    return result, (np.arange(result.shape[0]), np.arange(result.shape[1]))

def median_wealth_by_age():
    """Return median wealth as function of age."""
    # These numbers read from the bar chart on p12 of "Main Results from the
    # Wealth and Assets Survey 2006/08" (Wealth_in_GB_2006_2008_tcm77-169961.pdf)
    age = np.array([20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0])
    wealth = np.array([15000.0, 65000.0, 170000.0, 290000.0, 420000.0, 310000.0, 230000.0, 170000.0])
    return wealth, (age, )

def median_wealth_by_education():
    """Return median wealth as function of education."""
    # These numbers read from the bar chart on p13 of "Main Results from the
    # Wealth and Assets Survey 2006/08" (Wealth_in_GB_2006_2008_tcm77-169961.pdf)
    education = np.array([0, 2, 4])
    wealth = np.array([100000.0, 200000.0, 400000.0])
    return wealth, (education, )

def gini_data():
    """Return cumulative wealth of households."""
    path = os.path.join(INPUT_DIR, 'ONS/figure7_tcm77-272248.xls')
    data = pd.read_excel(path, 'Background figures', header=2, parse_cols=[3, 4])
    data = data[np.isfinite(data.values)]
    data /= 100.0
    return data['Cumulative % wealth'], (data['Cumulative % households'], )

def mean_wealth():
    """Return mean wealth of UK households."""
    path = os.path.join(INPUT_DIR, 'ONS/table9_tcm77-271482.xls')
    wealth = pd.read_excel(path, 'Table 9', header=5, parse_cols=[1, 5, 6, 7], skip_footer=5)
    mean = (0.01 * wealth['Percentage'] * wealth[u'Mean £']).sum()
    return np.array([mean]), (np.array([0.0]), )

def median_income_by_age_sex():
    """Return median income as function of age and sex."""
    path = os.path.join(INPUT_DIR, 'ONS/Table_3_2_14.xlsx')
    income = []
    for header in (35, 61): # Male, Female
        data = pd.read_excel(path, '3.2', header=header)[4:17]
        data = data['Median income                  before tax']
        income.append(data.values)
    age = np.array([
        20.0 if a == 'Under 20' else
        75.0 if a == '75 and over' else
        0.5 * (float(a[:2]) + float(a[-2:]) + 1)
        for a in data.index.get_level_values(0)
    ])
    return np.array(income).astype(float).T, (age, ('male', 'female'))

def relative_income_by_education():
    """Return relative incomes by education level."""
    # https://www.theguardian.com/news/datablog/2011/aug/24/earnings-by-qualification-degree-level
    # No quals, some quals, GCSE A*-C, A-levels, degree
    diff = np.array([0.8, 0.93, 1.0, 1.15, 1.85])
    skill = np.arange(5)
    return diff, (skill, )

def income_distribution():
    """Return income by percentile of the population."""
    path = os.path.join(INPUT_DIR, 'ONS/Table_3_1a_14.xlsx')
    income = pd.read_excel(path, '3.1a', header=6, index_col=0)[4:103]['2013-14']
    return income.values, (income.index.values.astype(float), )

def education_by_x(path, values, labels):
    """Return fraction at each education level as function of x."""
    # No quals, 1-4 GCSEs, 5+ GCSEs or apprenticeship, 2+ A-levels, degree
    data = pd.read_excel(path, 'Sheet 1', header=10, index_col=0)[2:7]
    skill = range(5)
    skill_labels = [
        ['No qualifications'],
        ['Level 1 qualifications'],
        ['Level 2 qualifications', 'Apprenticeship'],
        ['Level 3 qualifications'],
        ['Level 4 qualifications and above'],
    ]
    count = np.zeros((len(skill_labels), len(labels)))
    for i_skill, skill_label in enumerate(skill_labels):
        for i_label, label in enumerate(labels):
            count[i_skill, i_label] = data[skill_label].sum(axis=1).loc[label]
    fraction = count / count.sum(axis=0, keepdims=True)
    return fraction, (skill, values)

def education_by_age():
    """Return fraction at each education level as function of age."""
    path = os.path.join(INPUT_DIR, 'census/education_by_age.xlsx')
    # Ignore the 16-24 group, as many of them have not finished education yet
    age = [30.0, 42.5, 57.5, 75.0]
    age_labels = [
        'Age 25 to 34',
        'Age 35 to 49',
        'Age 50 to 64',
        'Age 65 and over',
    ]
    return education_by_x(path, age, age_labels)

def education_by_sex():
    """Return fraction at each education level as function of sex."""
    path = os.path.join(INPUT_DIR, 'census/education_by_sex.xlsx')
    sex = ('male', 'female')
    sex_labels = ('Males', 'Females')
    return education_by_x(path, sex, sex_labels)

if __name__ == '__main__':
    generate_data('f_as'  , 'fraction by age and sex (A, S)', f_age_sex)
    generate_data('fr_a'  , 'fertility rate by age (A)', fertility_rate)
    generate_data('D_yl'  , 'divorce rate by year of marriage and duration (Ym, Lm)', divorce_rate)
    generate_data('X_say' , 'death rate by sex, age and year (S, A, Y)', death_rate)
    generate_data('c_a'   , 'coupling rate by age (A)', coupling_rate)
    generate_data('M_as'  , 'married fraction by age and sex (A, S)', f_married)
    generate_data('C_as'  , 'cohabiting fraction by age and sex (A, S)', f_cohabiting)
    generate_data('ch_ac' , 'number of children by age (A, C)', n_children)
    generate_data('w_a'   , 'median wealth by age (A)', median_wealth_by_age)
    generate_data('w_e'   , 'median wealth by education (E)', median_wealth_by_education)
    generate_data('cw_f'  , 'cumulative wealth by household fraction (f)', gini_data)
    generate_data('w'     , 'mean household wealth', mean_wealth)
    generate_data('i_as'  , 'median income by age and sex (A, S)', median_income_by_age_sex)
    generate_data('i_e'   , 'relative income by education (E)', relative_income_by_education)
    generate_data('i_f'   , 'income distribution (f)', income_distribution)
    generate_data('f_ea'  , 'fraction by education and age (E, A)', education_by_age)
    generate_data('f_es'  , 'fraction by education and sex (E, S)', education_by_sex)

