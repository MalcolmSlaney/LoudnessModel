# Python Code for Calculation of the Loudness of Time-Varying Sounds
**Authors:** Jeremy JX Hsiao, Malcolm Slaney </br>
*Based on MATLAB Loudness Model provided by Brian C.J. Moore, Brian R. Glasberg and Josef Schlittenlacher*

**Contact Emails:**
jeremyjxhsiao@gmail.com, malcolm@ieee.org </br>
bcjm@cam.ac.uk, bg12@cam.ac.uk, js2251@cam.ac.uk

## TABLE OF CONTENTS
I. INTRODUCTION </br>
II. GETTING STARTED </br>
III. RUNNING THE PROGRAM </br>
IV. OUTPUTS OF THE PROGRAM </br>
V. TESTING SUITE </br>
VI. SUBROUTINES </br>
VII. JAX </br>
VIII. REFERENCES </br>

## I. INTRODUCTION 
This code implements a model of time-varying auditory loudness in Python.
The Python code is based on an original implementation in MATLAB, 
translated using two different computational libraries: 
[Numpy](https://numpy.org/) and [JAX](https://jax.readthedocs.io/en/latest/).

The code calculates loudness according to the model described by Moore et 
al. (2016), but with the modified time constants described by Moore et al. (2018). It was
developed from C code for the same model, and Matlab code written for ANSI S3.4-2007,
based on Moore et al. (1997) and Glasberg and Moore (2006) and ISO 532-2 (2017), 
based on Moore and Glasberg (2007). The source code is provided free for any research purposes.

A link to the original MATLAB code is the first download on this page: 
https://www.psychol.cam.ac.uk/hearing#programs

### Background Information

The model calculates both short-term and long-term loudness, reflecting the complexity of human loudness perception. Short-term loudness captures rapid changes and moment-to-moment variations, eg. syllables or single notes of music, while long-term loudness provides an overall impression of sound over extended periods. These two metrics together help in comprehending how sound intensity and perception change over time, addressing both immediate and sustained auditory experiences.

Loudness is influenced by multiple factors, including frequency, duration, and the phase relationships of sounds. Higher frequencies are often perceived as louder at the same sound pressure level (SPL) compared to lower frequencies, and longer sounds tend to be perceived as louder due to temporal integration. The model uses units such as sone and phon to quantify loudness, with 
*   sones representing perceived loudness on a linear scale (e.g., doubling in sones means doubling in perceived loudness) and 
*   phons representing equal loudness contours across different frequencies. 

Additionally, binaural loudness is considered—sounds perceived with both ears are often louder than those heard with just one ear.


### Why Use This Model?

* **Understand Sound Perception:** This model helps you understand how changes in sound characteristics affect perceived loudness.

* **Research & Analysis:** Useful for academic research in auditory perception and psychoacoustics.

* **Practical Applications:** From product testing in audio hardware to studying environmental noise levels.

#### Time Constants in the TVL2018 Model
According to Moore et al. (2018), the model includes "three stages with different degrees of temporal smoothing, corresponding to instantaneous, short-term, and long-term loudness." The paper states "Short-term loudness is meant to represent the loudness of a short segment of sound, such as a word or a single musical note, whereas long-term loudness is meant to represent the overall loudness of a longer segment of sound, such as a sentence or a musical phrase."

*    Attack time (Ta/Tal) = how quickly the system responds to increases in level
*    Release time (Tr/Trl) = how quickly it responds to decreases in level
*    aa/aal = attack coefficient 
*    ar/arl = release coefficient

#### Original Constants

Moore's 2016 model used:

Short-term: Ta = 22 ms (aa = 0.045), Tr = 50 ms (ar = 0.02)

Long-term: Tal = 99 ms (aal = 0.01), Trl = 2000 ms (arl = 0.0005)

These were chosen to "give reasonable predictions of the way that loudness varies with duration" and "give reasonably accurate predictions of the overall loudness of sounds that are AM at low rates." (Moore, 2018)

#### Modified Constants

After optimization to better fit experimental data:

Short-term: Ta = 22 ms (aa = 0.045), Tr = 30 ms (ar = 0.033)

Long-term: Tal = 99 ms (aal = 0.01), Trl = 751 ms (arl = 0.00133)

For more information on how and why these constants were modified, read section IV from the Moore et al. 2018 paper. 

Short-term is described by the paper as for individual words/notes and long-term for sentences/phrases. The time constants appear to be empirically determined rather than derived from fundamental auditory principles. Moore et al. (2018) focused on refining these values through experimental data fitting rather than explaining their theoretical basis.

## II. GETTING STARTED

### Prerequisites
Ensure you have a version of Python 3.x installed. 

### Installation
1. Clone the repository:

```python
git clone https://github.com/MalcolmSlaney/LoudnessModel.git
```

2. Navigate to project directory.

```python
cd loudnessmodel
```

3. Install the required packages:

```python 
pip install -r requirements.txt
```

**Optional:** </br>
If you wish to use the JAX-accelerated version, install JAX:

```python
pip install jax jaxlib
```

## III. RUNNING THE PROGRAM

The main function for loudness calculation is compute_loudness, located in the tvl2018 module. 


The function compute_loudness takes four parameters

**FUNCTION SIGNATURE:**
```python
def compute_loudness(
    sound: Union[np.ndarray, np.ndarray],
    db_max: float,
    filter: Union[np.ndarray, np.ndarray],
    rate: int = None,
):
```

**`sound`**: Input sound data as a 2D-array

**`db_max`**: The root-mean-square sound pressure level (SPL) of a full-scale sinusoid (i.e., a sinusoid whose peak amplitude is 1). This allows calibration of absolute level. 
Typical values:

* **0–40 dB SPL**: Quiet to very quiet environments.
* **60–80 dB SPL**: Noisy environments.
* **Default**: 50 dB SPL.

**`filter`**: The array specifies the transfer function through the outer and middle ear. 
* `ff_32000` for free-field presentation, 

* `df_32000` for diffuse-field presentation,

* `ed_32000` for middle-ear only (when the signal is picked up at the eardrum, or headphones with a “flat” frequency response at the eardrum are used). 

**`rate`**: The sampling rate of the signal, can be specified. If providing your own array data for the signal, be sure to specify rate. If reading from a file or synthesizing a signal, the rate is determined automatically


## IV. OUTPUTS OF THE PROGRAM
The function returns three main results:
* Instantaneous loudness
* Short-term loudness
* Long-term loudness 

Each is provided as an array with 1 ms intervals starting from t = 0 ms.

**EXAMPLE INPUT** 

```python
from tvl2018 import compute_loudness

frequency = 1000  # Hz - frequency of the tone
duration = 0.1    # seconds - length of the tone
rate = 32000      # Hz - sample rate 
db_max = 50       # dB SPL - reference level

# Synthesize the sound
sound = tvl.synthesize_sound(frequency, duration, rate)

# Calculate loudness 
loudness, short_term, long_term = tvl.compute_loudness(
    sound,
    db_max,
    transfer_functions.ff_32000,
    rate
)
```
## INTERACTIVE DEMOS
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JQcklNVzuwJVy3fBco64IlO87RQ1WeH5?usp=sharing)

These demos provide interactive demonstrations of loudness perception using the TVL2018 model. Each demo highlights different aspects of the program. It provides plots and statistics on analyzed audio data. 

* Demo 1: Basic Loudness Analysis - Understanding fundamental loudness measurements

* Demo 2: Real-World Sound Loudness Analysis - Analyzing loudness in real-world audio files

* Demo 3: Parameter Effects - The impact of frequency, duration, and level


## V. TESTING SUITE

This test suite validates the implementation of the TVL2018 loudness model by covering a general overall test, precision tests, and individual utility functions.

### Basic Tests

- **`test_basic_example`**: Tests the `compute_loudness` function with a 100ms synthesized 1 kHz tone at 50 dB SPL and 32k sample rate with free field transform, checking short-term and long-term loudness calculations. You can change inputs here to get different plots and summary files.

- **`test_peak_constrained_power_optimization`**: Validates and demonstrates that phase adjustments can increase power/loudness while maintaining peak amplitude constraints. Tests different phase configurations (cosine phase baseline, all-pass filter, random phases) to verify improvements in RMS and loudness while keeping peak amplitude constant.


## Baseline Comparison Tests

These tests ensure that the core components of the loudness model produce accurate and consistent results by comparing to a set of predefined inputs. Specifically: 1 kHz tone at 50 db SPL and 32k sample rate with free-field transform. This is most useful when users make changes to tvl2018 but want to maintain the same overall functionality by verifying with a known accurate output. The tests also include commented code for producing debug files and plots if needed. 

- **`test_overall_loudness`**: Verifies that the overall loudness matches the expected maximum long-term loudness value.

- **`test_short_term_loudness`**: Validates the first five values of the short-term loudness against expected results.

- **`test_long_term_loudness`**: Validates the first five values of the long-term loudness against expected results.

- **`test_signal_segment_to_spectrum`**: Confirms that the conversion of a signal segment to its spectrum matches expected frequency and level values.

- **`test_spectrum_to_excitation_pattern_025_selected`**: Tests specific points of the excitation pattern generated from a given spectrum against expected values.

- **`test_excitation_to_specific_loudness_binaural_025_selected`**: Validates the conversion from excitation patterns to specific loudness at selected indices.

- **`test_filtered_signal_to_monaural_instantaneous_specific_loudness_selected`**: Checks instantaneous specific loudness calculations for both ears at selected segments and ERB indices.

## Simple Utility Function Tests

These tests focus on the correctness of some smaller functions that support the main loudness calculations.

- **`test_interpolation`**: Verifies the `interpolation` function using 'pchip' and 'linear' methods, asserting that the standard error remains within acceptable limits.

- **`test_agc_functions`**: Validates the automatic gain control functions (`agc_next_frame_of_vector` and `agc_next_frame`) using known inputs and expected outputs.

- **`test_synthesize_sound`**: Checks the sound synthesis function for correct output shape and amplitude scaling.

- **`test_excitation_threshold_tvl`**: Validates excitation threshold calculations against expected values.


## Running the Tests

To run the test suite, execute the following command:

```python
python tvl2018_test.py
```

Ensure all dependencies are installed and the `tvl2018` module is accessible. The tests can output results to the `results` directory for further inspection if needed.



## VI. SUBROUTINES

You will find many useful subroutines in the main directory and subdirectory ‘functions’. They may be used to calculate excitation patterns, perform a Fast Fourier Transform (FFT), convert sone to phon or Hz to Cam (the units of the ERBN-number scale), calculate the
equivalent rectangular bandwidth of the auditory filter, calculate binaural inhibition, and implement automatic gain circuits, among other things.

## VII. JAX

The Numpy code was translated to JAX and runs. Unfortunately it does not compile as there are several portions of the implementation that are [not pure](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html). We want to fix these details, but also welcome contributions from the community.

To use this code in JAX, import 
```python
import tvl2018_jax as tvl
```
To run the JAX test suite, execute the following command:

```python
python tvl2018_jax_test.py
```




## VIII. REFERENCES

Glasberg, B. R., and Moore, B. C. J. (2006). "Prediction of absolute thresholds 
and equal-loudness contours using a modified loudness model," J. Acoust. Soc. Am. 120, 585-588 
[[link](https://pubmed.ncbi.nlm.nih.gov/16938942/)].

ISO 532-2 (2017). Acoustics - Methods for calculating loudness - Part 2: Moore-Glasberg method (International Organization for Standardization, Geneva) [[link](https://www.iso.org/standard/63078.html)].

Moore, B. C. J., and Glasberg, B. R. (2007). "Modeling binaural loudness," 
J. Acoust. Soc. Am. 121, 1604-1612 
[[link](https://pubs.aip.org/asa/jasa/article-abstract/121/3/1604/952165/Modeling-binaural-loudness)].

Moore, B. C. J., Glasberg, B. R., and Baer, T. (1997). "A model for the prediction of thresholds, loudness and partial loudness," J. Audio Eng. Soc. 45, 224-240 [[link](https://aes2.org/publications/elibrary-page/?id=10272)].

Moore, B. C. J., Glasberg, B. R., Varathanathan, A., and Schlittenlacher, J. (2016). 
"A loudness model for time-varying sounds incorporating binaural inhibition," 
Trends Hear. 20, 1-16 [[link](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5318944/)].

Moore, B. C. J., Jervis, M., Harries, L., and Schlittenlacher, J. (2018). "Testing and refining a loudness model for time-varying sounds incorporating binaural inhibition," J. Acoust. Soc. Am. 143, 1504-1513
[[link](https://pubmed.ncbi.nlm.nih.gov/29604698/)].
