import numpy as np
import matplotlib.pyplot as plt

def plot_attention_histogram(attn_mat, title='Attention Values Distribution', figsize=(10, 6)):
    """
    Plot a histogram of attention matrix values with predefined bins
    
    Parameters:
        attn_mat (np.ndarray): 2D attention matrix (after softmax)
        title (str): Plot title
        figsize (tuple): Figure size
    """
    # Flatten the attention matrix to 1D array
    values = attn_mat.flatten()
    
    # Define bins [0-0.1, 0.1-0.2, ..., 0.9-1.0]
    #bins = [i/10 for i in range(11)]  # [0.0, 0.1, 0.2, ..., 1.0]
    fine_bins = np.linspace(0, 0.1, 101)
    bins = np.append(fine_bins, [1.0])

    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    main_hist = values[values <= 0.1]
    n, bins, patches = ax.hist(main_hist, bins=fine_bins, edgecolor='black', alpha=0.7)

    # Add the >0.1 count as a separate bar
    large_count = np.sum(values > 0.1)
    if large_count > 0:
        # Add a bar for >0.1 values at position x=0.15 (right of the 0.1 mark)
        bar = ax.bar(0.15, large_count, width=0.05, edgecolor='black', alpha=0.7)
        # Add text label
        ax.text(0.15, large_count*1.05, f'{int(large_count):,}', 
                ha='center', va='bottom')
    
    # Customize plot
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Attention Value Ranges', fontsize=12)
    ax.set_ylabel('Frequency (Log Scale)', fontsize=12)
    
    # Set x-axis limits and ticks
    ax.set_xlim(0, 0.2)
    xticks = [0, 0.025, 0.05, 0.075, 0.1, 0.15]
    ax.set_xticks(xticks)
    ax.set_xticklabels(['0', '0.025', '0.05', '0.075', '0.1', '>0.1'])
    
    # Use log scale on y-axis
    ax.set_yscale('log')
    
    # For the 0-0.1 bins, add labels every 10 bins to avoid clutter
    for i in range(0, 100, 10):
        if i < len(n) and n[i] > 0:
            ax.text(patches[i].get_x() + patches[i].get_width()/2,
                    n[i]*1.05,
                    f'{int(n[i]):,}',
                    ha='center', va='bottom', fontsize=8)
    
    # Customize plot
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Attention Value Ranges', fontsize=12)
    ax.set_ylabel('Frequency (Log Scale)', fontsize=12)
    #ax.set_xticklabels([f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(len(bins)-1)], 
    #                  rotation=45, ha='right')
    
    # Use log scale on y-axis (common for attention distributions)
    ax.set_yscale('log')
    
    # For the 0-0.1 bins, we'll add labels only every 10 bins to avoid clutter
    for i in range(0, 100, 10):
        if n[i] > 0:
            ax.text(patches[i].get_x() + patches[i].get_width()/2,
                    n[i]*1.05,
                    f'{int(n[i]):,}',
                    ha='center', va='bottom', fontsize=8) 
    plt.tight_layout()
    return fig

# Example usage:
if __name__ == '__main__':
    # Load your attention matrix (replace with actual loading code)
    # attn_mat = np.load('attention_matrix.npy') 
    
    # Generate example data if no real data available
    attn_mat = np.load("./attention_matrix/attn_reg_fwd5_head3.txt.npz")['data']
    print(f"Size of attention matrix is {attn_mat.shape}")
    
    fig = plot_attention_histogram(attn_mat, 
                                 title='Distribution of Attention Values')
    plt.show()
