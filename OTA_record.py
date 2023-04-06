#!/usr/bin/env python3

# Import Packages
import numpy as np
import os
import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CS16
from scipy.signal import resample_poly, firwin, bilinear, lfilter
import matplotlib.pyplot as plt
import time
import argparse

########################################################################################
# Settings
########################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--nfiles", type=int, default=1, help="number of files to capture")
args = parser.parse_args()

# Determine how much data to record
nfiles = 1              # Number of files to record fix this to allow args input
N = 16384 * 38          # Number of complex samples per file - approximately 20ms

# Data transfer settings
rx_chan = 0                         # RX1 = 0, RX2 = 1
cplx_samples_per_file = N * 25/16   # Increase number of samples to account fo resampling
N = nfiles * cplx_samples_per_file  # total number of samples to record
fs = 31.25e6                        # Radio sample Rate
freq = 2.417e9                      # LO tuning frequency in Hz
use_agc = True                      # Use or don't use the AGC
timeout_us = int(5e6)

# Recording Settings
rec_dir = '/home/deepwave/Research/DSTL/OTA_dataset'  # Location of drive for recording
timestr = time.strftime("%Y%m%d-%H%M%S")
file_prefix = 'OTA' + timestr                         # File prefix for each file

########################################################################################
# Receive Signal
########################################################################################
# File calculations and checks
files_per_buffer = int(nfiles)
real_samples_per_file = int(2 * cplx_samples_per_file)


#  Initialize the AIR-T receiver using SoapyAIRT
sdr = SoapySDR.Device(dict(driver="SoapyAIRT"))  # Create AIR-T instance
sdr.setSampleRate(SOAPY_SDR_RX, 0, fs)           # Set sample rate
sdr.setGainMode(SOAPY_SDR_RX, 0, use_agc)        # Set the gain mode
sdr.setFrequency(SOAPY_SDR_RX, 0, freq)          # Tune the LO

N = int(N)
#print(N)
# Create data buffer and start streaming samples to it
rx_buff = np.empty(2 * N, np.int16)  # Create memory buffer for data stream
rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16, [rx_chan]) # Setup data stream
sdr.activateStream(rx_stream)  # this turns the radio on

# file_names = []  # Save the file names for plotting later. Remove if not plotting.
file_ctr = 0
while file_ctr < nfiles:
    # Read the samples from the data buffer
    sr = sdr.readStream(rx_stream, [rx_buff], N, timeoutUs=timeout_us)

    # Make sure that the proper number of samples was read
    rc = sr.ret
    assert rc == N, 'Error Reading Samples from Device (error code = %d)!' % rc

    # Write buffer to multiple files. Reshaping the rx_buffer allows for iteration
    for file_data in rx_buff.reshape(files_per_buffer, real_samples_per_file):
        # Define current file name
        file_name = os.path.join(rec_dir, '{}_{}.bin'.format(file_prefix, file_ctr))

        # Write signal to disk
        s0 = file_data.astype(float)
        samples = s0[::2] + 1j*s0[1::2] # convert to IQIQIQ...
        # print(samples)
        
        # Low-Pass Filter
        taps = firwin(numtaps=101, cutoff=10e6, fs=fs)
        lpf_samples = np.convolve(samples, taps, 'valid')
        
        # rational resample
        # Resample to 20e6
        resampled_samples = resample_poly(lpf_samples, 16, 25) # 16*31.25=500,20*25=500(need LCM because input needs to be an int). 
        # So we go up by factor of 16, then down by factor of 25 to reach final samp_rate of 20e6
        
        resampled_samples.tofile(file_name)

        # Save file name for plotting later. Remove this if you are not going to plot.
        # file_names.append(file_name)

        # Increment file write counter
        file_ctr += 1


# Stop streaming and close the connection to the radio
sdr.deactivateStream(rx_stream)
sdr.closeStream(rx_stream)

########################################################################################
# Plot Recorded Data
########################################################################################

# nrow = 2
# ncol = np.ceil(float(nfiles) / float(nrow)).astype(int)
# fig, axs = plt.subplots(nrow, ncol, figsize=(11, 11), sharex=True, sharey=True)
# for ax, file_name in zip(axs.flatten(), file_names):
#     # Read data from current file
#     s_interleaved = np.fromfile(file_name, dtype=np.int16)
#
#     # Convert interleaved shorts (received signal) to numpy.float32
#     s_real = s_interleaved[::2].astype(np.float32)
#     s_imag = s_interleaved[1::2].astype(np.float32)
#
#     # Plot time domain signals
#     ax.plot(s_real, 'k', label='I')
#     ax.plot(s_imag, 'r', label='Q')
#     ax.set_xlim([0, len(s_real)])
#     ax.set_title(os.path.basename(file_name))
#
# plt.show()


