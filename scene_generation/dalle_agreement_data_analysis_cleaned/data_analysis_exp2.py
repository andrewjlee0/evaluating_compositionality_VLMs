import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from data_analysis_exp1 import mean_confidence_interval


def get_prompt_likelihood(likelihood_csv, tag='prompt_davinci_mult_prob'):
    likelihood_df = pd.read_csv(likelihood_csv)

    result_mapping = {}
    for prefix in ['unreversed_', 'reversed_']:
        prompts = likelihood_df[prefix + 'prompt'].tolist()
        likelihood = np.array(likelihood_df[prefix + tag].tolist(), dtype=float)
        for p, l in zip(prompts, likelihood):
            result_mapping[p.strip()] = l
    return result_mapping


if __name__ == '__main__':
    BOX_WIDTH = 0.35
    EXP = 'exp2'
    excluded_participant = [9, 21, 38, 55]
    long_format_file = f'./data/data_dalle_prompt_agreement_{EXP}_long_format.csv'
    prompt_list = pd.read_excel('./data/Exp2_prompt_list.xlsx')
    normal_prompts = list(prompt_list['final_prompts'].dropna())
    normal_prompts = [p.strip() for p in normal_prompts]
    reversed_prompts = list(prompt_list['final_reversed_prompts'].dropna())
    reversed_prompts = [p.strip() for p in reversed_prompts]

    reverse_prompt_mapping = {r_p: p for r_p, p in zip(reversed_prompts, normal_prompts)}

    prompt_likelihood = get_prompt_likelihood('./data/unreversed_and_reversed_prompts_scores.csv', tag='prompt_davinci_mult_prob')

    results = pd.read_csv(long_format_file)

    # Replace all occurrences of 'Original' with 'Basic' in the 'Prompt Type' column
    results['Prompt Type'] = results['Prompt Type'].replace('Original', 'Basic')

    # Group by Relation, Image, and Type, then calculate the mean 'Agree'
    grouped_df = results.groupby(['Relation', 'Image', 'Type', 'Prompt Type']).agg({
        'Agree': 'mean'
    }).reset_index()

    # Set the order for 'Prompt Type'
    grouped_df['Prompt Type'] = pd.Categorical(grouped_df['Prompt Type'], categories=['Basic', 'Reversed'],
                                               ordered=True)

    # Calculate the mean ratings and sort them in descending order
    mean_ratings = results.groupby('Relation')['Agree'].mean().sort_values(ascending=False).reset_index()

    # Reorder grouped_df based on sorted mean ratings
    grouped_df['Relation'] = pd.Categorical(grouped_df['Relation'], categories=mean_ratings['Relation'], ordered=True)
    grouped_df = grouped_df.sort_values('Relation')
    relation_order = grouped_df['Relation'].cat.categories.tolist()

    ###################### Correlation between image and likelihood ##############
    image_df = results.groupby(['Prompt', 'Image', 'Relation']).agg({
        'Agree': 'mean'
    }).reset_index()

    likelihoods = []
    # Iterate through each row
    for index, row in image_df.iterrows():
        likelihood = prompt_likelihood[row['Prompt']]
        likelihoods.append(likelihood)

    image_df['Likelihood'] = likelihoods

    # Calculate the Pearson correlation coefficient and the p-value
    correlation, p_value = pearsonr(image_df['Agree'], image_df['Likelihood'])
    # Calculate degrees of freedom
    degrees_of_freedom = len(image_df) - 2

    print(f"Corr image v.s. likelihood: r({degrees_of_freedom}) = {correlation:.04f} (p = {p_value:.04f})")

    # Calculate the Spearman correlation coefficient and the p-value
    correlation, p_value = spearmanr(image_df['Agree'], image_df['Likelihood'])
    # Calculate degrees of freedom
    degrees_of_freedom = len(image_df) - 2

    print(f"Spearman Corr image v.s. likelihood: r({degrees_of_freedom}) = {correlation:.04f} (p = {p_value:.04f})")

    ############## Flip axes ##########
    # Define the custom color palette
    custom_palette = {'Basic': '#AFD2E9', 'Reversed': '#9D96B8'}
    colors = [custom_palette['Basic'], custom_palette['Reversed']] * len(grouped_df['Relation'].unique())

    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))

    strip = sns.stripplot(x='Relation', y='Agree', hue='Prompt Type', data=grouped_df, jitter=False, marker='o',
                          dodge=True,
                          alpha=0.3, palette='dark:black', legend=False)

    # Set custom colors for each strip point
    for i, artist in enumerate(strip.collections):
        artist.set_facecolor(colors[i])

    # Plot confidence intervals as rectangles
    for i, relation in enumerate(relation_order):
        for j, prompt_type in enumerate(['Basic', 'Reversed']):
            offset = 0.03 if j == 1 else - 0.03 - BOX_WIDTH
            relation_data = grouped_df[
                (grouped_df['Relation'] == relation) & (grouped_df['Prompt Type'] == prompt_type)]
            mean_value, ci_lower, ci_upper = mean_confidence_interval(relation_data, column_name='Agree',
                                                                      confidence=0.95)
            plt.gca().add_patch(
                plt.Rectangle(
                    (i + offset, ci_lower),
                    BOX_WIDTH,
                    ci_upper - ci_lower,
                    edgecolor='black',
                    facecolor=custom_palette[prompt_type],
                    alpha=0.6
                )
            )
            plt.plot([i + offset, i + offset + BOX_WIDTH], [mean_value, mean_value], color='black', linewidth=2)

    # Adding labels and title
    legend_handles = [plt.Line2D([0], [0], color=color, lw=4) for color in custom_palette.values()]
    legend_labels = custom_palette.keys()
    plt.legend(legend_handles, legend_labels, title='Prompt Type', bbox_to_anchor=(1.05, 1), loc='upper left',
               fontsize=22, title_fontsize=24)

    sns.despine(left=True, bottom=True, right=True, top=True)
    plt.grid(True, axis='both', linestyle='-', color='gray', alpha=0.2)

    # Adjust axis labels and limits
    plt.xlabel('Relation', fontsize=28)
    plt.ylabel('Agreement', fontsize=28)
    plt.xticks(fontsize=24, rotation=45)
    plt.yticks(fontsize=24)
    plt.ylim([-0.03, 1.03])

    plt.tight_layout()

    plt.savefig('data/exp2_boxplot.pdf', format='pdf')
    plt.show()
