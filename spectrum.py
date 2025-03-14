from rtlsdr import RtlSdr
import numpy as np
import matplotlib.pyplot as plt
import scienceplots; plt.style.use('science')
from scipy.signal import welch
import time
from tqdm import tqdm
import argparse
import datetime 

def get_args(): 
    parser = argparse.ArgumentParser(description='SDR Spectrum Analyzer')
    parser.add_argument('-f1', '--start_freq', type=float, default=1200, help='Start frequency in MHz')
    parser.add_argument('-f2', '--stop_freq', type=float, default=1500, help='Stop frequency in MHz')
    parser.add_argument('-s', '--sample_rate', type=float, default=2.56, help='Sample rate in MHz')
    parser.add_argument('-g', '--gain', type=str, default='auto', help='Gain setting')
    parser.add_argument('-n', '--nperseg', type=int, default=1024, help='Number of samples per segment')
    parser.add_argument('-save', '--savedata', action="store_true", default=False, help='Save plot and data to file')
    return parser.parse_args()

def main(): 
    
    args = get_args()
    

    # Initialize SDR
    sdr = RtlSdr()
    sdr.sample_rate = args.sample_rate*1e6  # Max stable sample rate for NooElec R820T2 is 2.56 MHz
    sdr.gain = 'auto'  # Adjust as needed

    # Define Frequency Sweep
    start_freq = args.start_freq * 1e6 
    stop_freq = args.stop_freq * 1e6
    step_size = sdr.sample_rate  # Step by ~2.56 MHz
    freqs = np.arange(start_freq, stop_freq, step_size)  # Frequency steps

    spectrum_data = []
    # Sweep across frequencies
    for freq in tqdm(freqs):
        sdr.center_freq = freq
        time.sleep(0.1)  # Settle time
        npers = args.nperseg
        samples = sdr.read_samples(npers*10)  # Samples 

        # Compute Power Spectral Density (PSD)
        f, Pxx = welch(samples, fs=sdr.sample_rate, nperseg=npers, return_onesided=False)
        
        spectrum_data.append((freq + f - sdr.sample_rate / 2, 10 * np.log10(Pxx)))  # Convert to dB

    sdr.close()
    timenow = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    plottime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Combine Data
    full_freq = np.concatenate([entry[0] for entry in spectrum_data])
    full_power = np.concatenate([entry[1] for entry in spectrum_data])

    # Plot Spectrum
    plt.figure(figsize=(12, 6))
    plt.plot(full_freq / 1e6, full_power, lw=0.5, color='b')
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power (dB)")
    plt.title(f"Spectrum from {args.start_freq} MHz to {args.stop_freq} GHz at {plottime}")
    plt.grid()
    plt.tight_layout()
    
    if args.savedata:
        output_name= f"spectrum-{args.start_freq}-{args.stop_freq}-{timenow}"
        plt.savefig(f"output/plots/{output_name}.png")
        np.savetxt(f"output/data/{output_name}.dat", np.vstack((full_freq/1e6, full_power)).T, fmt='%f', delimiter=' ')
    
    plt.show()
if __name__ == '__main__':
    main()
