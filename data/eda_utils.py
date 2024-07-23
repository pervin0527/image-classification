import seaborn as sns
import matplotlib.pyplot as plt

def visualize_class_distribution(train_df, meta_df, save_plot=False, plot_path="class_distribution.png"):
    class_dict = dict(zip(meta_df['target'], meta_df['class_name']))
    train_df['class_name'] = train_df['target'].map(class_dict)

    class_counts = train_df['class_name'].value_counts().reset_index()
    class_counts.columns = ['class_name', 'file_count']

    plt.figure(figsize=(12, 6))
    sns.barplot(x='class_name', y='file_count', hue='class_name', data=class_counts, palette='viridis', legend=False)
    plt.xticks(rotation=90)
    plt.title('Number of Files per Class')
    plt.xlabel('Class Name')
    plt.ylabel('File Count')
    plt.tight_layout()

    if save_plot:
        plt.savefig(plot_path)
        print(f"Plot saved as {plot_path}")
    else:
        plt.show()

    return class_counts