# Python Code for Calculation of the Loudness of Time-Varying Sounds
Jeremy JX Hsiao, Malcolm Slaney
Based on MATLAB Loudness Model Provided by Brian C.J. Moore, Brian R. Glasberg and Josef Schlittenlacher

jeremyjxhsiao@gmail.com, malcolmslaney@gmail.com
bcjm@cam.ac.uk, bg12@cam.ac.uk, js2251@cam.ac.uk

## I. INTRODUCTION
The Python code is based on an original implementation in MATLAB, translated using various Python libraries (NumPy, SciPy, JAX etc.)
It provided calculates loudness according to the model described by Moore et 
al. (2016), but with the modified time constants described by Moore et al. (2018). It was
developed from C code for the same model, and Matlab code written for ANSI S3.4-2007,
based on Moore et al. (1997) and Glasberg and Moore (2006) and ISO 532-2 (2017), 
based on Moore and Glasberg (2007). The source code is provided free for any research purposes.

Link to the original MATLAB code is the first download on this page: https://www.psychol.cam.ac.uk/hearing#programs

## II. RUNNING THE PROGRAM
the function main_tv2018 takes in five parameters: filename_or_sound: Union[str, np.ndarray], db_max: float, filename_filter: str, output_path: str = None, rate: int = None)

filename_or_sound: This can either be a path to a file name, a NumPy array of audio data with rate specified, or you can create your own synthesized wave files by inputting "synthesize_{}khz_{}ms" into filename_or_sound

db_max: The root-mean-square sound pressure level of a full-scale sinusoid, i.e. a sinusoid whose peak amplitude is 1 in Matlab. This allows calibration of absolute level.

filename_filter: The filename of the wav file for which the loudness is calculated. The signal s and sampling rate Fs are specified, the filename is only used as a name for the output files. Use ‘ff_32000.mat’ for free-field presentation, ‘df_32000.mat’ for diffuse-field presentation or ‘ed_32000.mat’ for middle-ear only (when the signal is picked up at the eardrum, or headphones with a “flat” frequency response at the eardrum are used).

rate: sampling rate of the signal, can be specified. If providing your own array data for the signal, be sure to specify rate. 

## III. OUTPUTS OF THE PROGRAM
The function returns three variables, each of them being vectors starting at t = 0 ms and having a step size of 1 ms. The first vector is the instantaneous loudness, the second is the short-term loudness, and the third is long-term loudness, all in sone. In addition, the program creates a text file in the subdirectory out, having the same filename as specified in filenameSound and the extension ‘.txt’. It contains seven columns, specifying the time in ms, instantaneous loudness, short-term loudness and long-term loudness in both sone and loudness level in phon. Finally, the program creates a Matplotlib figure with a black line representing instantaneous loudness, a blue line representing short-term loudness and a red line representing long-term loudness. 

## IV. EXAMPLES
filename_or_sound = 'synthesize_1khz_100ms' 
db_max = 50
filename_filter = 'transfer functions/ff_32000.mat'
loudness, short_term_loudness, long_term_loudness = main_tv2018(filename_or_sound, db_max, filename_filter)

[to do, add tests and descriptions of tests, and more comprehensive examples]
