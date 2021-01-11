import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
df['overweight'] = [
    1 if x > 25 else 0 for x in df['weight'] / ((df['height']/100) ** 2)]

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholestorol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.

df['cholesterol'] = [0 if x == 1 else 1 for x in df['cholesterol']]
df['gluc'] = [0 if x == 1 else 1 for x in df['gluc']]

# Draw Categorical Plot


def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    values = ['active', 'alco', 'cholesterol',
              'gluc', 'overweight', 'smoke', 'cardio']

    df_val = df[values]
    df_cat = df_val.melt(
        id_vars='cardio', value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'], var_name='variable', value_name='values')

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the collumns for the catplot to work correctly.
    graph = sns.catplot(x='variable', col='cardio',
                        hue='values', data=df_cat, kind='count')
    graph.set_axis_labels('variable', 'total')
    fig = graph.fig

    #c0 = df[df['cardio'] == 0][values]
    #c1 = df[df['cardio'] == 1][values]

    # c0_melt = c0.melt(value_vars=values, var_name='variable',
    #                  value_name='values')
    # c1_melt = c1.melt(value_vars=values, var_name='variable',
    #                  value_name='values')

    ##df_cat = None

    # Draw the catplot with 'sns.catplot()'
    #fig, ax = plt.subplots(1, 2)
    #ax[0] = sns.catplot(x='variable', hue='values', data=c0_melt, kind='count')
    #ax[1] = sns.catplot(x='variable', hue='values', data=c1_melt, kind='count')

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data

    dfh = df[(df['ap_lo'] <= df['ap_hi'])
             & (df['height'] >= df['height'].quantile(0.025))
             & (df['height'] <= df['height'].quantile(0.975))
             & (df['weight'] >= df['weight'].quantile(0.025))
             & (df['weight'] <= df['weight'].quantile(0.975))]

    # df_heat = None

    # Calculate the correlation matrix
    corr = dfh.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(16, 12))

    # Draw the heatmap with 'sns.heatmap()'
    ax = sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', center=0, linewidths=.5,
                     vmax=0.25, vmin=-0.1, square=True, cbar_kws={'format': '%.2f', 'shrink': .45})

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
