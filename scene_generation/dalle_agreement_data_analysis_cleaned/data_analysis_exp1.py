import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def mean_confidence_interval(df, column_name, confidence=0.95):
    """
    Calculate the mean and 95% confidence interval for a specified column in a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    column_name (str): The name of the column for which to calculate the mean and confidence interval.
    confidence (float): The confidence level for the interval (default is 0.95).

    Returns:
    tuple: A tuple containing the mean, lower bound, and upper bound of the confidence interval.
    """
    data = df[column_name]
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of the mean
    margin_of_error = sem * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error

    return mean, lower_bound.item(), upper_bound.item()


if __name__ == '__main__':
    EXP = 'exp1'
    excluded_participant = [3, 11, 16, 23]
    BOX_WIDTH=0.4
    long_format_file = f'./data/data_dalle_prompt_agreement_{EXP}_long_format.csv'

    results = pd.read_csv(long_format_file)

    # Calculate the mean ratings and sort them in descending order
    mean_ratings = results.groupby('Relation')['Agree'].mean().sort_values(ascending=False).reset_index()

    ################## Plot Conwell & Ullman in the same plot ############################
    conwell_df = pd.read_csv('./data/choice_data_conwell_ullman.csv')
    conwell_df['Image'] = conwell_df['Subject'] + '_' + conwell_df['Relation'] + '_' + conwell_df['Predicate'] + '_' + conwell_df['ImageID']

    conwell_df = conwell_df.rename(columns={'JSONID': 'ID', 'Selected': 'Agree', 'RelationType': 'Type'})
    conwell_df['Type'] = conwell_df['Type'].str.capitalize()
    # Change 'cover' to 'covering' in the Relation column
    conwell_df['Relation'] = conwell_df['Relation'].replace('cover', 'covering')

    # Add dataset
    conwell_df['Model'] = ['DALL-E 2'] * len(conwell_df)
    results['Model'] = ['DALL-E 3'] * len(results)

    # Combine the dataframes, ignoring columns that don't match
    combined_df = pd.concat([conwell_df, results], ignore_index=True)

    # Set the order for the Relation column
    relation_order = mean_ratings['Relation']
    combined_df['Relation'] = pd.Categorical(combined_df['Relation'], categories=relation_order, ordered=True)
    combined_df['Model'] = pd.Categorical(combined_df['Model'], categories=['DALL-E 2', 'DALL-E 3'], ordered=True)

    # Group by Relation, Image, Type, and Dataset, then calculate the mean 'Agree'
    grouped_combined_df = combined_df.groupby(['Relation', 'Image', 'Type', 'Model']).agg({
        'Agree': 'mean'
    }).dropna().reset_index()

    palette = {'DALL-E 2': '#E3738B', 'DALL-E 3': '#8CA5EA'}
    colors = [palette['DALL-E 2'], palette['DALL-E 3']] * len(results['Relation'].unique())

    # Map the color palette to the combined data
    def get_color(row):
        return palette[row['Model']]

    grouped_combined_df['Color'] = grouped_combined_df.apply(get_color, axis=1)

    plt.figure(figsize=(15, 12))
    sns.set(style="whitegrid")

    strip = sns.stripplot(x='Relation', y='Agree', hue='Model', data=grouped_combined_df, jitter=False, marker='o',
                          alpha=0.3, dodge=True, legend=False)
    for i, artist in enumerate(strip.collections):
        artist.set_facecolor(colors[i])

    # Plot confidence intervals as rectangles
    for i, relation in enumerate(relation_order):
        for j, model in enumerate(['DALL-E 2', 'DALL-E 3']):
            offset = 0.03 if j == 1 else -0.03 - BOX_WIDTH
            relation_data = grouped_combined_df[
                (grouped_combined_df['Relation'] == relation) & (grouped_combined_df['Model'] == model)]
            mean_value, ci_lower, ci_upper = mean_confidence_interval(relation_data, column_name='Agree',
                                                                      confidence=0.95)
            plt.gca().add_patch(
                plt.Rectangle(
                    (i + offset, ci_lower),
                    BOX_WIDTH,
                    ci_upper - ci_lower,
                    edgecolor='black',
                    facecolor=palette[model],
                    alpha=0.6
                )
            )
            plt.plot([i + offset, i + offset + BOX_WIDTH], [mean_value, mean_value], color='black', linewidth=2)

    # Adding labels and title
    legend_handles = [plt.Line2D([0], [0], color=color, lw=4) for color in palette.values()]
    legend_labels = palette.keys()
    plt.legend(legend_handles, legend_labels, title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=22,
               title_fontsize=24)

    sns.despine(left=True, bottom=True, right=True, top=True)
    plt.grid(True, axis='both', linestyle='-', color='gray', alpha=0.2)

    # Adjust axis labels and limits
    plt.xlabel('Relation', fontsize=28)
    plt.ylabel('Agreement', fontsize=28)
    plt.xticks(fontsize=24, rotation=45, ha='right', va='top')
    plt.yticks(fontsize=24)
    plt.ylim([-0.03, 1.03])
    plt.xlim([14.5, -0.5])

    plt.tight_layout()
    plt.savefig('data/exp1B_boxplot.pdf', format='pdf')
    plt.show()

    ###################### Paired t-test between DALL-E 3 and DALL-E 2 data ######################
    conwell_prompt_df = conwell_df.groupby(['Prompt']).agg({
        'Agree': 'mean'
    }).reset_index()
    prompt_df = results.groupby(['Prompt']).agg({
        'Agree': 'mean'
    }).reset_index()

    # Merge the dataframes on 'Prompt' to find the overlapping prompts
    merged_df = pd.merge(conwell_prompt_df, prompt_df, on='Prompt', how='inner', suffixes=('_conwell', '_prompt'))

    # Conduct a paired t-test on the 'Agree' columns
    t_stat, p_value = stats.ttest_rel(merged_df['Agree_prompt'], merged_df['Agree_conwell'])

    print(f"Paired t-test for 'Agree' values over matched prompts:")
    print(f"Mean of 'Agree' in conwell_prompt_df: {merged_df['Agree_conwell'].mean():.3f}")
    print(f"Mean of 'Agree' in prompt_df: {merged_df['Agree_prompt'].mean():.3f}")
    print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")

    # Additional info for the paper
    n_prompts = merged_df.shape[0]
    degrees_of_freedom = n_prompts - 1

    print(f"Number of matched prompts: {n_prompts}")
    print(f"Degrees of freedom: {degrees_of_freedom}")
    print("----------------------------------------------------------------------")
