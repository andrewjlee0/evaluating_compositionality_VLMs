import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_analysis_exp1 import mean_confidence_interval


if __name__ == '__main__':
    BOX_WIDTH = 0.4
    color = "#8199EE"
    long_format_file = './data/data_dalle_prompt_agreement_exp3_long_format.csv'
    results = pd.read_csv(long_format_file)

    # Group by Relation1 and Relation2, then calculate the mean of Agree
    grouped = results.groupby(['Relation1', 'Relation2', 'Image'])['Agree'].mean().reset_index()

    # Sort the grouped data by 'Agree'
    grouped = grouped.sort_values(by='Agree', ascending=False)

    # Create a combined column for x-ticks
    grouped['Relation_Combined'] = grouped['Relation1'].replace('convering', 'covering') + ' & ' + grouped['Relation2'].replace('convering', 'covering')
    results['Relation_Combined'] = results['Relation1'].replace('convering', 'covering') + ' & ' + results['Relation2'].replace('convering', 'covering')

    # # Calculate the mean ratings and sort them in descending order
    mean_ratings = results.groupby('Relation_Combined')['Agree'].mean().sort_values(ascending=False).reset_index()

    # Reorder grouped_df based on sorted mean ratings
    grouped['Relation_Combined'] = pd.Categorical(grouped['Relation_Combined'], categories=mean_ratings['Relation_Combined'], ordered=True)
    grouped = grouped.sort_values('Relation_Combined')
    relation_order = grouped['Relation_Combined'].cat.categories.tolist()

    ############ Reversed plot ##################
    sns.set(style="whitegrid")
    plt.figure(figsize=(20, 12))

    # Plot confidence intervals as rectangles
    for i, relation in enumerate(relation_order):
        relation_data = grouped[grouped['Relation_Combined'] == relation]
        mean_value, ci_lower, ci_upper = mean_confidence_interval(relation_data, column_name='Agree', confidence=0.95)
        plt.gca().add_patch(
            plt.Rectangle(
                (i - BOX_WIDTH, ci_lower),
                2 * BOX_WIDTH,
                ci_upper - ci_lower,
                edgecolor='black',
                facecolor=color,
                alpha=0.6
            )
        )
        plt.plot([i - BOX_WIDTH, i + BOX_WIDTH], [mean_value, mean_value], color='black', linewidth=2)

    strip = sns.stripplot(x='Relation_Combined', y='Agree', data=grouped, jitter=False, marker='o', alpha=0.4,
                          color='black')

    for i, artist in enumerate(strip.collections):
        artist.set_facecolor(color)

    # Calculate and plot the means
    means = results.groupby('Relation_Combined')['Agree'].mean().reindex(mean_ratings['Relation_Combined'])
    means = means.reset_index()

    sns.despine(left=True, bottom=True, right=True, top=True)
    plt.grid(True, axis='both', linestyle='-', color='gray', alpha=0.2)
    plt.ylabel('Agreement', fontsize=28)
    plt.xlabel('')
    plt.yticks(fontsize=24)
    plt.xticks(fontsize=20, rotation=45, ha='right', va='top')
    plt.ylim([-0.03, 1.03])

    plt.tight_layout()
    plt.savefig('data/exp3_boxplot.pdf', format='pdf')
    plt.show()
