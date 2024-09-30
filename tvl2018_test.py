import numpy as np
import os
import matplotlib.pyplot as plt

from absl.testing import absltest

import tvl2018 as tvl


class LoudnessTest(absltest.TestCase):
    def test_basic_example(self):
        """Test the main_tv2018 function with a synthesized 1 kHz tone."""
        filename_or_sound = 'synthesize_1khz_100ms'
        db_max = 50
        filter_filename = 'transfer functions/ff_32000.mat'
        debug_plot_filename = os.path.join(
            'results', 'synthesize_1khz_100ms_50dB_loudness_plot.png')
        debug_summary_filename = os.path.join(
            'results', 'synthesize_1khz_100ms_50dB_calibration_level_TVL_2018.txt')

        # Ensure results directory exists
        os.makedirs('results', exist_ok=True)
        (loudness,
         short_term_loudness,
         long_term_loudness) = tvl.main_tv2018(
            filename_or_sound, db_max, filter_filename,
            debug_plot=True,
            debug_plot_filename=debug_plot_filename,
            debug_summary_filename=debug_summary_filename)

        # Weak sanity tests
        np.testing.assert_array_less(long_term_loudness, short_term_loudness + 1e-6)
        np.testing.assert_array_less(0.0, short_term_loudness)
        np.testing.assert_array_less(0.0, long_term_loudness)

        # Make sure debug files are created.
        self.assertTrue(os.path.exists(debug_plot_filename))
        self.assertTrue(os.path.exists(debug_summary_filename))

        # Plotting the short-term and long-term loudness for visualization
        plt.figure(figsize=(10, 5))
        plt.plot(short_term_loudness, label='Short-term Loudness')
        plt.plot(long_term_loudness, label='Long-term Loudness')
        plt.xlabel('Time (ms)')
        plt.ylabel('Loudness (Sone)')
        plt.title('Loudness over Time for 1 kHz Tone at 50 dB')
        plt.legend()
        plt.grid(True)
        plot_filename = os.path.join('results', 'test_basic_example_loudness_plot.png')
        plt.savefig(plot_filename)
        plt.close()
        self.assertTrue(os.path.exists(plot_filename))


    def test_interpolation(self):
        """Test the interpolation function with pchip and linear and visualize the results."""
        # Test data
        np.random.seed(42)
        x = np.arange(0, 1, 0.1)
        y = np.sin(x * 2 * np.pi)
        x_probe = np.random.rand(1000)
        y_true = np.sin(x_probe * 2 * np.pi)

        y_predicted = tvl.interpolation(x, y, x_probe)
        y_error = y_true - y_predicted
        std_error = np.std(y_error)
        print(f'Standard error: {std_error}')

        # Plotting
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, y, 'o', label='Original Data')
        plt.plot(x_probe, y_predicted, '.', label='Interpolated')
        plt.legend()
        plt.title('Interpolation with pchip')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.hist(y_error, bins=50)
        plt.xlabel('Error')
        plt.title('Error Distribution (pchip)')
        plt.grid(True)

        plot_filename = os.path.join('results', 'test_interpolation_pchip.png')
        plt.savefig(plot_filename)
        plt.close()
        self.assertTrue(os.path.exists(plot_filename))

        # Assert that the standard error is acceptable
        self.assertLess(std_error, 0.04, "Standard error too high for pchip interpolation")

        # Test interpolation function with 'linear' interpolator
        print("Testing interpolation function with linear interpolator...")
        np.random.seed(42)
        x = np.arange(0, 1, 0.1)
        y = np.sin(x * 2 * np.pi)
        x_probe = np.random.rand(1000)
        y_true = np.sin(x_probe * 2 * np.pi)

        y_predicted = tvl.interpolation(x, y, x_probe, 'linear')
        y_error = y_true - y_predicted
        std_error = np.std(y_error)
        print(f'Standard error: {std_error}')

        # Plotting
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, y, 'o', label='Original Data')
        plt.plot(x_probe, y_predicted, '.', label='Interpolated')
        plt.legend()
        plt.title('Interpolation with linear')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.hist(y_error, bins=50)
        plt.xlabel('Error')
        plt.title('Error Distribution (linear)')
        plt.grid(True)

        plot_filename = os.path.join('results', 'test_interpolation_linear.png')
        plt.savefig(plot_filename)
        plt.close()
        self.assertTrue(os.path.exists(plot_filename))

        # Assert that the standard error is acceptable
        self.assertLess(std_error, 0.05, "Standard error too high for linear interpolation")


    def test_agc_next_frame_of_vector(self):
        """Test the AGC function with known inputs."""
        v_last_frame = np.array([1.0, 2.0, 3.0])
        v_this_input = np.array([2.0, 1.5, 4.0])
        aA = 0.5  # Attack parameter
        aR = 0.1  # Release parameter

        expected_output = np.array([
            aA * v_this_input[0] + (1 - aA) * v_last_frame[0],  # Attack
            aR * v_this_input[1] + (1 - aR) * v_last_frame[1],  # Release
            aA * v_this_input[2] + (1 - aA) * v_last_frame[2],  # Attack
        ])

        output = tvl.agc_next_frame_of_vector(v_last_frame, v_this_input, aA, aR)
        np.testing.assert_allclose(output, expected_output, rtol=1e-5)
    
    
    def test_agc_next_frame(self):
        d_last_frame = 1.0
        d_this_input = 2.0
        attack = 0.5
        release = 0.1

        # Attack condition
        output = tvl.agc_next_frame(d_last_frame, d_this_input, attack, release)
        expected_output = attack * d_this_input + (1 - attack) * d_last_frame
        self.assertAlmostEqual(output, expected_output, places=5)

        # Release condition
        d_last_frame = 2.0
        d_this_input = 1.0
        output = tvl.agc_next_frame(d_last_frame, d_this_input, attack, release)
        expected_output = release * d_this_input + (1 - release) * d_last_frame
        self.assertAlmostEqual(output, expected_output, places=5)

        # Equal inputs
        d_last_frame = 1.0
        d_this_input = 1.0
        output = tvl.agc_next_frame(d_last_frame, d_this_input, attack, release)
        self.assertAlmostEqual(output, d_this_input, places=5)


    def test_get_g_tvl(self):
        """Test the cochlear amplifier gain calculation."""
        f = np.array([125, 250, 500, 1000, 2000, 4000, 8000])
        g = tvl.get_g_tvl(f)

        # Compares with expected results drawn from MATLAB code.
        expected_g = np.array([0.1247383514, 0.5407543229, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])
        np.testing.assert_allclose(g, expected_g, rtol=1e-4)
        print("Comparison with MATLAB values passed.")
        # Print the results for verification
        print("Frequency (Hz) | Computed Gain | MATLAB Gain")
        for freq, gain, expected_gain in zip(f, g, expected_g):
            print(f"{freq:14} | {gain:14f} | {expected_gain:14f}")

        thresholds = tvl.excitation_threshold_tvl(f)
        calculated_g = 10 ** ((3.63 - thresholds) / 10)
        np.testing.assert_allclose(g, calculated_g, rtol=1e-4)
        print("Comparison with calculated values passed.")
        print("get_g_tvl tests passed.\n")


    def test_erb_number_to_frequency(self):
        """Test the conversion from ERB numbers to frequencies."""
        erb_numbers = np.array([1, 10, 20, 30, 40])
        frequencies = tvl.erb_number_to_frequency(erb_numbers)
        # Expected frequencies calculated using the formula
        expected_frequencies = (10 ** (erb_numbers / 21.366) - 1) / 0.004368
        np.testing.assert_allclose(frequencies, expected_frequencies, rtol=1e-5)


    def test_excitation_threshold_tvl(self):
        """Test the excitation threshold calculation."""
        f = np.array([100, 250, 500, 750, 1000, 2000, 4000])
        thresholds = tvl.excitation_threshold_tvl(f)
        # For frequencies >= 500 Hz, threshold is 3.63, as stated in excitation_threshold_tvl
        expected_thresholds = np.where(f < 500, thresholds, 3.63)
        np.testing.assert_allclose(thresholds, expected_thresholds, rtol=1e-5)


    def test_input_level_per_erb(self):
        """Test the input levels per ERB calculation."""
        f = np.array([1000.0, 2000.0])
        in_levels = np.array([60.0, 70.0])
        input_levels = tvl.input_level_per_erb(f, in_levels)
        # Checks if the lengths are equal and if levels are positive
        self.assertEqual(len(input_levels), len(in_levels))
        self.assertTrue(np.all(input_levels >= 0), "Input levels should be non-negative.")


    def test_synthesize_sound(self):
        """Test the sound synthesis function."""
        frequency = 1000  # Hz
        duration = 0.1    # seconds
        rate = 32000      # Hz
        sound = tvl.synthesize_sound(frequency, duration, rate)
        expected_samples = int(rate * duration)
        self.assertEqual(sound.shape, (expected_samples, 2), "Sound shape mismatch.")
        # Plotting the synthesized waveform
        plt.figure(figsize=(10, 5))
        t = np.linspace(0, duration, expected_samples, endpoint=False)
        plt.plot(t, sound[:, 0], label='Left Channel')
        plt.plot(t, sound[:, 1], label='Right Channel', linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Synthesized 1 kHz Tone')
        plt.legend()
        plt.grid(True)
        plot_filename = os.path.join('results', 'test_synthesize_sound_plot.png')
        plt.savefig(plot_filename)
        plt.close()
        self.assertTrue(os.path.exists(plot_filename))


    def test_signal_segment_to_spectrum(self):
        """Test the spectrum calculation of a signal segment."""
        rate = 32000
        npts = int(rate / 1000 * 64)
        t = np.linspace(0, npts / rate, npts, endpoint=False)
        freq = 1000
        db_max = 100
        data = np.sin(2 * np.pi * freq * t)
        data = np.column_stack((data, data))
        # Prepare Hann windows and indices
        w_hann = np.zeros((npts, 6))
        for i in range(6):
            half_window_size = npts // (2 ** i)
            pad_size = (npts - half_window_size) // 2
            w_hann[:, i] = np.concatenate([
                np.zeros(pad_size),
                np.hanning(half_window_size),
                np.zeros(npts - half_window_size - pad_size)
            ])
        v_limiting_f = [20, 80, 500, 1250, 2540, 4050, 15000]
        v_limiting_indices = [int(f / (rate / npts)) + 1 for f in v_limiting_f]
        # Call the function
        f_left_relevant, l_left_relevant, f_right_relevant, l_right_relevant = \
            tvl.signal_segment_to_spectrum(data, rate, db_max, w_hann, v_limiting_indices)
        # Verify the peak frequency
        peak_freq_left = f_left_relevant[np.argmax(l_left_relevant)]
        self.assertAlmostEqual(peak_freq_left, freq, delta=50,
                               msg="Left channel peak frequency mismatch.")
        peak_freq_right = f_right_relevant[np.argmax(l_right_relevant)]
        self.assertAlmostEqual(peak_freq_right, freq, delta=50,
                               msg="Right channel peak frequency mismatch.")


    def test_filtered_signal_to_monaural_instantaneous_specific_loudness(self):
        """Test the calculation of instantaneous specific loudness."""
        rate = 32000
        duration = 0.1  # seconds
        frequency = 1000  # Hz
        data = tvl.synthesize_sound(frequency, duration, rate)
        db_max = 100
        instantaneous_specific_loudness_left, instantaneous_specific_loudness_right = \
            tvl.filtered_signal_to_monaural_instantaneous_specific_loudness(data, rate, db_max)
        n_segments = instantaneous_specific_loudness_left.shape[0]
        self.assertEqual(instantaneous_specific_loudness_left.shape,
                         (n_segments, 150), "Left channel shape mismatch.")
        self.assertEqual(instantaneous_specific_loudness_right.shape,
                         (n_segments, 150), "Right channel shape mismatch.")
        # Plotting the specific loudness over ERB rate
        plt.figure(figsize=(10, 5))
        erb_rate = np.arange(1.75, 39.25, 0.25)
        plt.plot(erb_rate, instantaneous_specific_loudness_left[0, :], label='Left Ear')
        plt.plot(erb_rate, instantaneous_specific_loudness_right[0, :], label='Right Ear', linestyle='--')
        plt.xlabel('ERB Rate')
        plt.ylabel('Specific Loudness (Sone)')
        plt.title('Instantaneous Specific Loudness')
        plt.legend()
        plt.grid(True)
        plot_filename = os.path.join('results', 'test_instantaneous_specific_loudness_plot.png')
        plt.savefig(plot_filename)
        plt.close()
        self.assertTrue(os.path.exists(plot_filename))


    def test_instantaneous_specific_loudness_to_shortterm_specific_loudness(self):
        """Test the AGC from instantaneous to short-term specific loudness."""
        instantaneous_specific_loudness = np.abs(np.random.randn(100, 150))
        short_term_specific_loudness, short_term_loudness = \
            tvl.instantaneous_specific_loudness_to_shortterm_specific_loudness(
                instantaneous_specific_loudness)
        self.assertEqual(short_term_specific_loudness.shape,
                         instantaneous_specific_loudness.shape, "Shape mismatch.")
        self.assertEqual(len(short_term_loudness), instantaneous_specific_loudness.shape[0],
                         "Length mismatch.")

        plt.figure(figsize=(10, 5))
        plt.plot(np.sum(instantaneous_specific_loudness, axis=1) / 4, label='Instantaneous Loudness')
        plt.plot(short_term_loudness, label='Short-term Loudness')
        plt.xlabel('Time Frames')
        plt.ylabel('Loudness (Sone)')
        plt.title('AGC Effect on Loudness')
        plt.legend()
        plt.grid(True)
        plot_filename = os.path.join('results', 'test_agc_loudness_plot.png')
        plt.savefig(plot_filename)
        plt.close()
        self.assertTrue(os.path.exists(plot_filename))


    def test_shortterm_loudness_to_longterm_loudness(self):
        """Test the AGC from short-term to long-term loudness."""
        Nst = np.linspace(0, 10, 100)
        Nlt = tvl.shortterm_loudness_to_longterm_loudness(Nst)
        self.assertEqual(len(Nlt), len(Nst), "Length mismatch.")

        plt.figure(figsize=(10, 5))
        plt.plot(Nst, label='Short-term Loudness')
        plt.plot(Nlt, label='Long-term Loudness')
        plt.xlabel('Time Frames')
        plt.ylabel('Loudness (Sone)')
        plt.title('AGC Effect from Short-term to Long-term Loudness')
        plt.legend()
        plt.grid(True)
        plot_filename = os.path.join('results', 'test_longterm_loudness_plot.png')
        plt.savefig(plot_filename)
        plt.close()
        self.assertTrue(os.path.exists(plot_filename))


    def test_main_tv2018(self):
        """Test the main function with a synthesized signal."""
        filename_or_sound = 'synthesize_1khz_100ms'
        db_max = 60
        filter_filename = 'transfer functions/ff_32000.mat'
        rate = 32000
        debug_plot_filename = 'results/test_main_tv2018_plot.png'
        debug_summary_filename = 'results/test_main_tv2018_summary.txt'
        os.makedirs('results', exist_ok=True)
        loudness, short_term_loudness, long_term_loudness = tvl.main_tv2018(
            filename_or_sound, db_max, filter_filename, rate=rate,
            debug_plot=True,
            debug_plot_filename=debug_plot_filename,
            debug_summary_filename=debug_summary_filename
        )
        self.assertTrue(loudness >= 0, "Negative loudness value.")
        self.assertEqual(len(short_term_loudness), len(long_term_loudness),
                         "Length mismatch.")
        self.assertTrue(os.path.exists(debug_summary_filename),
                        "Summary file not created.")
        # Load and check the summary file for expected content
        with open(debug_summary_filename, 'r') as f:
            content = f.read()
        self.assertIn("Maximum of long-term loudness", content,
                      "Summary file content missing.")
        # Check that the plot file is created
        self.assertTrue(os.path.exists(debug_plot_filename),
                        "Plot file not created.")


if __name__ == '__main__':
    absltest.main()
