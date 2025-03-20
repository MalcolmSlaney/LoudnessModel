import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import transfer_functions

def plot_bode(tf_data, sample_rate=44100, title="Bode Plot", figsize=(10, 6), 
              n_fft=4096, save_path=None):
    """
    Create a Bode magnitude plot for a single transfer function
    
    Parameters:
    -----------
    tf_data : numpy.ndarray
        The transfer function data
    sample_rate : int, optional
        Sample rate in Hz
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size (width, height)
    n_fft : int, optional
        FFT size
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    fig : matplotlib figure object
    """
    # Compute frequency response
    freqs, response = signal.freqz(tf_data, worN=n_fft)
    
    # Convert to Hz
    freqs = freqs * sample_rate / (2 * np.pi)
    
    # Calculate magnitude in dB
    magnitude = 20 * np.log10(np.abs(response) + 1e-10)  # Avoid log(0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot magnitude
    ax.semilogx(freqs, magnitude)
    ax.set_title(title)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.grid(True, which="both", linestyle='-', alpha=0.7)
    
    # Add frequency markers
    freq_ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    ax.set_xticks(freq_ticks)
    ax.set_xticklabels([str(f) for f in freq_ticks])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def compare_bode(tf_list, labels=None, sample_rate=44100, title="Comparison of Transfer Functions", 
                figsize=(10, 6), n_fft=4096, save_path=None):
    """
    Compare multiple transfer functions on the same Bode magnitude plot
    
    Parameters:
    -----------
    tf_list : list of numpy.ndarray
        List of transfer function data to compare
    labels : list of str, optional
        Labels for each transfer function
    sample_rate : int, optional
        Sample rate in Hz
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size (width, height)
    n_fft : int, optional
        FFT size
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    fig : matplotlib figure object
    """
    if labels is None:
        labels = [f'TF {i+1}' for i in range(len(tf_list))]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color cycle
    colors = plt.cm.tab10.colors
    
    # Frequency markers
    freq_ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    
    # Plot each transfer function
    for i, (tf_data, label) in enumerate(zip(tf_list, labels)):
        # Compute frequency response
        freqs, response = signal.freqz(tf_data, worN=n_fft)
        
        # Convert to Hz
        freqs = freqs * sample_rate / (2 * np.pi)
        
        # Calculate magnitude in dB
        magnitude = 20 * np.log10(np.abs(response) + 1e-10)  # Avoid log(0)
        
        color = colors[i % len(colors)]
        
        # Plot magnitude
        ax.semilogx(freqs, magnitude, label=label, color=color)
    
    # Configure plot
    ax.set_title(title)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.grid(True, which="both", linestyle='-', alpha=0.7)
    ax.set_xticks(freq_ticks)
    ax.set_xticklabels([str(f) for f in freq_ticks])
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

# Example usage
if __name__ == "__main__":
    # This won't run when imported, but shows how to use the functions
    # Example: Plot a simple Bode plot
    # Example: Compare multiple transfer functions
    compare_bode([transfer_functions.df_32000, transfer_functions.ed_32000, transfer_functions.ff_32000], 
                labels=["DF", "ED", "FF"],
                title="Comparison of Transfer Functions")
    
    plt.show()