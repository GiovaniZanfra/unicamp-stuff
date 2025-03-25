import librosa
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
import numpy as np 
from scipy.signal import tf2zpk
import soundfile as sf


def plot_weights_evolution(weights_evolution, max_weights=2, title=""):
    plt.figure(figsize=(10, 6))
    
    # Get the number of weights in the first element of weights_evolution
    num_weights = weights_evolution[0].shape[0]
    
    # Limit the number of weights to plot (max_weights argument or total weights)
    weights_to_plot = min(num_weights, max_weights)
    
    # Loop through the weights and plot the evolution
    for i in range(weights_to_plot):  # Plot up to max_weights or all weights
        plt.plot([w[i] for w in weights_evolution], label=f'Weight {i+1}')
    
    plt.xlabel('Iterations')
    plt.ylabel('Weight Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def normalize(signal):
    """
    Normalizes a signal to have zero mean and unit variance.

    Parameters:
        signal (np.array): Input signal.

    Returns:
        np.array: Normalized signal.
    """
    return signal / np.max(np.abs(signal))

def rls(desired_signal, input_signal, forgetting_factor, filtering_order, delta=1.0):
    """
    Implements the RLS algorithm for adaptive filtering.

    Parameters:
        desired_signal (np.array): The desired signal.
        input_signal (np.array): The input signal.
        forgetting_factor (float): Forgetting factor (0 < forgetting_factor <= 1).
        filtering_order (int): The number of filter coefficients.
        delta (float): Regularization parameter for initializing the inverse correlation matrix.

    Returns:
        tuple: (output_signal, final_weights, weights_evolution, error_signal)
    """
    if filtering_order >= len(desired_signal):
        raise ValueError("Filtering order must be smaller than the length of the signals.")

    # Normalize signals
    input_signal = normalize(input_signal)
    desired_signal = normalize(desired_signal)

    signal_length = len(desired_signal)
    weights = np.zeros(filtering_order)
    output_signal = np.zeros(signal_length)
    error_signal = np.zeros(signal_length)

    # Initialize inverse correlation matrix
    P = np.eye(filtering_order) / delta

    # Track weights evolution
    weights_evolution = []

    for n in range(filtering_order, signal_length):
        desired_ = desired_signal[n]
        input_ = input_signal[n-filtering_order:n]  # Slice of length `filtering_order`

        # Compute filter output
        output_ = np.dot(weights, input_)
        output_signal[n] = output_

        # Compute error
        error_ = desired_ - output_
        error_signal[n] = error_

        # Compute gain vector
        input_reshaped = input_.reshape(-1, 1)
        gain = P @ input_reshaped / (forgetting_factor + input_.T @ P @ input_reshaped)

        # Update weights
        weights += (gain.flatten() * error_)
        weights_evolution.append(weights.copy())

        # Update inverse correlation matrix
        P = (P - gain @ input_.reshape(1, -1) @ P) / forgetting_factor

    return output_signal, weights, weights_evolution, error_signal


def lms(desired_signal, input_signal, step_size, filtering_order, power_normalization=False):
    """
    Implements the LMS algorithm for adaptive filtering.

    Parameters:
        desired_signal (np.array): The desired signal.
        input_signal (np.array): The input signal.
        step_size (float): Initial step size for weight updates.
        filtering_order (int): The number of filter coefficients.
        power_normalization (bool): Whether to normalize step size by input signal power.

    Returns:
        tuple: (output_signal, final_weights, weights_evolution, error_signal)
    """
    if filtering_order >= len(desired_signal):
        raise ValueError("Filtering order must be smaller than the length of the signals.")
    
    # Normalize signals
    input_signal = normalize(input_signal)
    desired_signal = normalize(desired_signal)

    # Compute power of the input signal for optional step size normalization
    input_power = np.mean(input_signal**2)

    if power_normalization:
        step_size /= input_power

    weights = np.zeros(filtering_order)
    signal_length = len(desired_signal)
    output_signal = np.zeros(signal_length)
    error_signal = np.zeros(signal_length)

    # Track weights evolution
    weights_evolution = []

    for n in range(filtering_order, signal_length):
        desired_ = desired_signal[n]
        
        # Ensure input slice is of the correct size, handling the end of the signal
        input_ = input_signal[n-filtering_order:n]  # Slice of length `filtering_order`
        
        if len(input_) < filtering_order:
            # If there are fewer samples left, pad the slice with zeros or any appropriate value
            input_ = np.pad(input_, (filtering_order - len(input_), 0), mode='constant', constant_values=0)
        
        output_ = np.dot(input_, weights)
        output_signal[n] = output_
        error_ = desired_ - output_
        error_signal[n] = error_
        weights += step_size * error_ * input_
        weights_evolution.append(weights.copy())  # Save current weights

    return output_signal, weights, weights_evolution, error_signal


def find_max_step_size(input_signal, filtering_order):
    """
    Finds the maximum step size for LMS algorithm convergence using power normalization.

    Parameters:
        input_signal (np.array): The input signal.
        filtering_order (int): The order of the adaptive filter.

    Returns:
        float: Maximum step size for convergence.
    """
    # Normalize input signal
    input_signal = normalize(input_signal)

    # Calculate average power of the input signal
    input_power = np.mean(input_signal**2)

    # Use trace approximation: A_max < tr[R], where tr[R] ≈ filtering_order * input_power
    trace_R = filtering_order * input_power

    # Maximum step size
    max_step_size = 2 / trace_R
    return max_step_size


def apply_fep(sinal, coeficientes):
    """
    Aplica o Filtro de Estimação de Parâmetros (FEP) ao sinal usando coeficientes ótimos.
    
    :param sinal: O sinal de entrada (numpy array).
    :param coeficientes: Coeficientes do filtro (numpy array).
    :return: Sinal filtrado e erro de predição.
    """
    num_coeficientes = len(coeficientes)
    preditor = np.zeros(len(sinal))
    erro = np.zeros(len(sinal))
    
    # Aplicando o filtro
    for n in range(num_coeficientes, len(sinal)):
        # Certifique-se de pegar exatamente num_coeficientes amostras anteriores
        preditor[n] = np.dot(coeficientes, sinal[n-num_coeficientes:n][::-1])
        erro[n] = sinal[n] - preditor[n]
    
    return preditor, erro

def wiener(desired_signal, input_signal, filtering_order):
    # Compute autocorrelation of the input signal
    r = np.correlate(input_signal, input_signal, mode="full")
    r = r[len(r)//2:]  # Only keep the positive lags

    # Create the autocorrelation matrix R
    R = np.zeros((filtering_order, filtering_order))
    for i in range(filtering_order):
        for j in range(filtering_order):
            R[i, j] = r[abs(i-j)]  # Lagged correlation

    # Compute cross-correlation of the desired signal and input signal
    p = np.correlate(input_signal, desired_signal, mode="full")
    p = p[len(p)//2:]  # Only keep the positive lags
    p = p[:filtering_order]  # Match the size to filtering_order

    # Solve for the Wiener filter coefficients (h = R^-1 * p)
    R_inv = np.linalg.inv(R)  # Inverse of autocorrelation matrix
    h = np.dot(R_inv, p)  # Wiener filter coefficients

    # # Apply the Wiener filter to the input signal
    filtered_signal, error = apply_fep(input_signal, h)

    return filtered_signal, error, h

def is_minimum_phase(impulse_response, sample_rate):
    """
    Check if an impulse response is minimum phase.

    Parameters:
        impulse_response (np.array): The impulse response of the system.
        sample_rate (float): The sample rate of the system (not used for the check).

    Returns:
        bool: True if the system is minimum phase, False otherwise.
    """
    # Normalize the impulse response
    impulse_response = impulse_response / np.max(np.abs(impulse_response))

    # Get the zeros, poles, and gain of the system
    b = impulse_response  # Numerator coefficients
    a = [1]  # Denominator coefficients for FIR systems (no poles)
    zeros, poles, _ = tf2zpk(b, a)

    # Check if all zeros are inside the unit circle
    all_zeros_inside_unit_circle = np.all(np.abs(zeros) < 1)
    
    return all_zeros_inside_unit_circle

def analyze_frequencies(signal, sampling_rate):
    """
    Analyzes the frequency content of a signal using FFT.

    Parameters:
        signal (np.array): Input signal in the time domain.
        sampling_rate (float): Sampling rate of the signal in Hz.

    Returns:
        freqs (np.array): Array of frequencies.
        magnitudes (np.array): Magnitude of the FFT.
    """
    # Length of the signal
    N = len(signal)
    
    # Compute FFT
    fft_result = np.fft.fft(signal)
    
    # Only take the positive frequencies
    fft_result = fft_result[:N // 2]
    
    # Frequency array
    freqs = np.fft.fftfreq(N, d=1/sampling_rate)[:N // 2]
    
    # Magnitude of the FFT
    magnitudes = np.abs(fft_result) / N
    
    return freqs, magnitudes

def plot_signals(signal_data, figsize=(10, 9)):
    """
    Plots multiple signals in a single figure with subplots.

    Parameters:
        signal_data (list): A list of tuples, where each tuple contains:
            - A signal (list or array of values to plot).
            - A label (string) describing the signal.
        figsize (tuple): The size of the figure (width, height).

    Example:
        plot_signals([
            ([0.1, 0.5, 0.9], 'Signal 1'),
            ([0.2, 0.6, 0.8], 'Signal 2'),
            ([0.3, 0.7, 0.6], 'Signal 3')
        ])
    """
    num_signals = len(signal_data)
    fig, ax = plt.subplots(num_signals, 1, figsize=figsize)
    
    # Ensure ax is an iterable for single subplot
    if num_signals == 1:
        ax = [ax]

    for i, (signal, label) in enumerate(signal_data):
        ax[i].plot(signal, label=label)
        ax[i].legend()
        ax[i].grid(True)

    plt.tight_layout()
    plt.show()
