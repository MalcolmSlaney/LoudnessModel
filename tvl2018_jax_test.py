import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np  # For testing
from absl.testing import absltest
import cProfile

import tvl2018_jax as tvl


class LoudnessModelTests(absltest.TestCase):
    """Test suite for the TVL2018 loudness model implementation."""

    def test_basic_example(self):
        """Test the main_tv2018 function with a synthesized 1 kHz tone."""
        # Ensure results directory exists
        os.makedirs('results', exist_ok=True)

        debug_plot_filename = os.path.join(
            'results', 'synthesize_1khz_100ms_50dB_loudness_plot.png')
        debug_summary_filename = os.path.join(
            'results', 'synthesize_1khz_100ms_50dB_calibration_level_TVL_2018.txt')

        # Here, you can modify the input values into tvl.main_tv2018
        db_max = 50  # Example SPL value
        filename_or_sound = 'synthesize_1khz_100ms'
        filter_filename = 'transfer functions/ff_32000.mat'
        _, short_term_loudness, long_term_loudness = tvl.main_tv2018(
            filename_or_sound,
            db_max,
            filter_filename,
            debug_plot=True,
            debug_plot_filename=debug_plot_filename,
            debug_summary_filename=debug_summary_filename
        )

        # Weak sanity tests
        np.testing.assert_array_less(long_term_loudness, short_term_loudness + 1e-6)
        np.testing.assert_array_less(0.0, short_term_loudness)
        np.testing.assert_array_less(0.0, long_term_loudness)

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

    # DO NOT MODIFY BELOW TESTS INPUT
    # The tests below compare to a set of expected values
    # based on the set inputs below as a baseline for future changes

    @classmethod
    def setUpClass(cls):
        """Set up shared test parameters and expected outputs."""
        cls.frequency = 1000
        cls.duration = 0.1
        cls.sample_rate = 32000  # Hz
        cls.db_max = 50  # Example SPL value
        cls.filename_or_sound = 'synthesize_1khz_100ms'
        cls.filter_filename = 'transfer functions/ff_32000.mat'

        cls.loudness, cls.short_term_loudness, cls.long_term_loudness = tvl.main_tv2018(
            cls.filename_or_sound,
            cls.db_max,
            cls.filter_filename,
            debug_plot=False,  # Disable plotting during tests
            debug_summary_filename=None  # Disable summary file
        )
        # Generate Hann windows for testing
        cls.npts = int(cls.sample_rate / 1000 * 64)  # 2048
        # cls.w_hann = jnp.zeros((cls.npts, 6))
        w_hann = []
        for i in range(6):
            half_window_size = cls.npts // (2 ** i)
            pad_size = int((1 - 1 / 2**i) / 2 * cls.npts)
            if half_window_size > 0:
                w_hann.append(jnp.concatenate([
                    jnp.zeros(pad_size),
                    jnp.hanning(half_window_size),
                    jnp.zeros(cls.npts - pad_size - half_window_size)
                ]))
            else:
                w_hann.append(jnp.zeros(cls.npts))
        cls.w_hann = jnp.stack(w_hann, axis=1)

        # Define limiting frequency indices for FFT windows
        cls.v_limiting_f = [20, 80, 500, 1250, 2540, 4050, 15000]
        cls.v_limiting_indices = [int(f / (cls.sample_rate / cls.npts)) + 1 for f in cls.v_limiting_f]

        # Expected Outputs from generate_expected_outputs.py (updated for 100 ms duration)
        cls.expected_outputs = {
            'loudness': 0.5513768526010244,  # sone
            'short_term_loudness_first5': jnp.array(
                [0.06962138, 0.12972933, 0.18023906, 0.22377568, 0.26341937]),
            'long_term_loudness_first5': jnp.array(
                [0.00069621, 0.00198654, 0.00376907, 0.00596914, 0.00854364]),
            'signal_segment_spectrum': {
                'f_left_relevant_first5': jnp.array(
                    [31.25, 46.875, 62.5, 78.125, 93.75]),
                'l_left_relevant_first5': jnp.array(
                    [-27.18885881, -21.78693278, -18.70249616, -16.82990504, -12.5730688]),
                'f_right_relevant_first5': jnp.array(
                    [31.25, 46.875, 62.5, 78.125, 93.75]),
                'l_right_relevant_first5': jnp.array(
                    [-27.18885881, -21.78693278, -18.70249616, -16.82990504, -12.5730688])
            },

            # These are arbitrary indexes chosen because comparing the entire array might take too long.
            'excitation_pattern_selected': jnp.array([
                -100.0,                # Index 0
                -100.0,                # Index 25
                43.206926024798676,    # Index 50
                65.3881013396141       # Index 100
            ]),
            'specific_loudness_selected': jnp.array([
                1.232447851133203e-09,  # Index 0
                0.0445007179518514,    # Index 25
                0.20997358085795026,    # Index 50
                1.3016835085413805      # Index 100
            ]),
            'instantaneous_specific_loudness_left_selected': jnp.array([
                2.866694731009077e-13,  # ist_loudness_left[0][1]
                5.225246937851411e-34,  # ist_loudness_left[25][1]
                5.225246937851411e-34,  # ist_loudness_left[50][1]
                2.8666350625317323e-13   # ist_loudness_left[100][1]
            ]),
            'instantaneous_specific_loudness_right_selected': jnp.array([
                2.866694731009077e-13,  # ist_loudness_right[0][1]
                5.225246937851411e-34,  # ist_loudness_right[25][1]
                5.225246937851411e-34,  # ist_loudness_right[50][1]
                2.8666350625317323e-13   # ist_loudness_right[100][1]
            ]),
        }

    def test_overall_loudness(self):
        """Test overall loudness against expected maximum long-term loudness."""
        # Ensure results directory exists
        os.makedirs('results', exist_ok=True)

        loudness = self.loudness

        # Uncomment this code for debug files and plot if needed
        # debug_plot_filename = os.path.join(
        #     'results', 'test_overall_loudness_plot.png')
        # debug_summary_filename = os.path.join(
        #     'results', 'test_overall_loudness_summary.txt')
        # loudness, _, _ = tvl.main_tv2018(
        #     self.filename_or_sound,
        #     self.db_max,
        #     self.filter_filename,
        #     debug_plot=True,
        #     debug_plot_filename=debug_plot_filename,
        #     debug_summary_filename=debug_summary_filename
        # )

        # Assert maximum loudness
        self.assertAlmostEqual(
            loudness,
            self.expected_outputs['loudness'],
            places=5,
            msg="Overall loudness does not match expected maximum value."
        )

    def test_short_term_loudness(self):
        """Test short-term loudness against expected first five values."""
        # Ensure results directory exists
        os.makedirs('results', exist_ok=True)
        short_term_loudness = self.short_term_loudness

        # Uncomment this code for debug files and plot if needed
        # debug_plot_filename = os.path.join(
        #     'results', 'test_short_term_loudness_plot.png')
        # debug_summary_filename = os.path.join(
        #     'results', 'test_short_term_loudness_summary.txt')
        # loudness, short_term_loudness, long_term_loudness = tvl.main_tv2018(
        #     self.filename_or_sound,
        #     self.db_max,
        #     self.filter_filename,
        #     debug_plot=True,
        #     debug_plot_filename=debug_plot_filename,
        #     debug_summary_filename=debug_summary_filename
        # )

        # Assert first five short-term loudness values
        np.testing.assert_allclose(
            short_term_loudness[:5],
            self.expected_outputs['short_term_loudness_first5'],
            rtol=1e-5,
            err_msg="First five short-term loudness values do not match expected."
        )

    def test_long_term_loudness(self):
        """Test long-term loudness against expected first five values."""
        # Ensure results directory exists
        os.makedirs('results', exist_ok=True)

        # Call main_tv2018
        long_term_loudness = self.long_term_loudness

        #  Uncomment this code for debug files and plot if needed
        # debug_plot_filename = os.path.join(
        #     'results', 'test_long_term_loudness_plot.png')
        # debug_summary_filename = os.path.join(
        #     'results', 'test_long_term_loudness_summary.txt')
        # _, _, long_term_loudness = tvl.main_tv2018(
        #     self.filename_or_sound,
        #     self.db_max,
        #     self.filter_filename,
        #     debug_plot=True,
        #     debug_plot_filename=debug_plot_filename,
        #     debug_summary_filename=debug_summary_filename
        # )
        

        # Assert first five long-term loudness values
        np.testing.assert_allclose(
            long_term_loudness[:5],
            self.expected_outputs['long_term_loudness_first5'],
            rtol=1e-5,
            err_msg="First five long-term loudness values do not match expected."
        )

    def test_signal_segment_to_spectrum(self):
        """Test the signal segment to spectrum conversion against expected values."""
        # Use the filtered signal from test_basic_example
        filter_filename = self.filter_filename
        # Synthesize and filter the sound
        sound = tvl.synthesize_sound(self.frequency, self.duration, rate=self.sample_rate)
        cochlea_filtered = tvl.sound_field_to_cochlea(sound, filter_filename)
        # Process the first segment
        segment = cochlea_filtered[:2048, :]  # First segment
        f_left_relevant, l_left_relevant, f_right_relevant, l_right_relevant = tvl.signal_segment_to_spectrum(
            data=segment,
            rate=self.sample_rate,
            db_max=self.db_max,
            w_hann=self.w_hann,
            v_limiting_indizes=self.v_limiting_indices
        )

        # Compare the first five relevant frequencies and levels for left
        np.testing.assert_allclose(
            f_left_relevant[:5],
            self.expected_outputs['signal_segment_spectrum']['f_left_relevant_first5'],
            rtol=1e-5,
            err_msg="Left relevant frequencies do not match expected."
        )
        np.testing.assert_allclose(
            l_left_relevant[:5],
            self.expected_outputs['signal_segment_spectrum']['l_left_relevant_first5'],
            rtol=1e-2,
            err_msg="Left relevant levels do not match expected."
        )

        # Compare the first five relevant frequencies and levels for right
        np.testing.assert_allclose(
            f_right_relevant[:5],
            self.expected_outputs['signal_segment_spectrum']['f_right_relevant_first5'],
            rtol=1e-5,
            err_msg="Right relevant frequencies do not match expected."
        )
        np.testing.assert_allclose(
            l_right_relevant[:5],
            self.expected_outputs['signal_segment_spectrum']['l_right_relevant_first5'],
            rtol=1e-2,
            err_msg="Right relevant levels do not match expected."
        )

    def test_spectrum_to_excitation_pattern_025_selected(self):
        """Test selected points of spectrum_to_excitation_pattern_025 against expected values."""
        f = jnp.array([1000.0, 2000.0, 3000.0])  # Hz
        in_levels = jnp.array([60.0, 70.0, 80.0])  # dB

        # Call the function under test
        excitation = tvl.spectrum_to_excitation_pattern_025(f, in_levels)

        # Define selected indices to skip the initial -100 dB values
        selected_indices = [0, 25, 50, 100]  # Indices corresponding to printed expected values

        # Extract excitation values at selected indices
        excitation_selected = excitation[jnp.array(selected_indices)]

        # Load expected excitation (manually input)
        expected_excitation = self.expected_outputs['excitation_pattern_selected']

        # Assertions
        self.assertEqual(
            len(excitation_selected),
            len(expected_excitation),
            msg="Selected excitation pattern length mismatch."
        )
        self.assertTrue(
            jnp.all(excitation_selected >= -100),
            msg="Selected excitation pattern contains values below -100 dB."
        )

        # Compare against expected excitation values
        np.testing.assert_allclose(
            excitation_selected,
            expected_excitation,
            rtol=1e-5,
            err_msg="Selected excitation pattern values do not match expected."
        )

        # Optional: Plot for manual inspection
        # Uncomment the following lines if you wish to visualize the comparison
        # plt.figure()
        # plt.plot(excitation_selected, label='Computed Excitation Selected')
        # plt.plot(expected_excitation, '--', label='Expected Excitation Selected')
        # plt.legend()
        # plt.title('Spectrum to Excitation Pattern Comparison (Selected Indices)')
        # plt.xlabel('Selected Indices')
        # plt.ylabel('Excitation Levels (dB)')
        # plt.grid(True)
        # plt.show()

    def test_excitation_to_specific_loudness_binaural_025_selected(self):
        """Test selected points of excitation_to_specific_loudness_binaural_025 against expected values."""
        excitation_levels = jnp.linspace(0, 100, 150)  # 150 ERB steps

        # Call the function under test
        specific_loudness = tvl.excitation_to_specific_loudness_binaural_025(excitation_levels)

        # Define selected indices for specific loudness
        selected_indices = [0, 25, 50, 100]  # Indices corresponding to printed expected values

        # Extract specific loudness values at selected indices
        specific_loudness_selected = specific_loudness[jnp.array(selected_indices)]

        # Load expected specific loudness (manually input)
        expected_specific_loudness = self.expected_outputs['specific_loudness_selected']

        # Assertions
        self.assertEqual(
            len(specific_loudness_selected),
            len(expected_specific_loudness),
            msg="Selected specific loudness length mismatch."
        )
        self.assertTrue(
            jnp.all(specific_loudness_selected >= 0),
            msg="Selected specific loudness contains negative values."
        )

        # Compare against expected specific loudness values
        np.testing.assert_allclose(
            specific_loudness_selected,
            expected_specific_loudness,
            rtol=1e-5,
            err_msg="Selected specific loudness values do not match expected."
        )

        # Optional: Plot for manual inspection
        # Uncomment the following lines if you wish to visualize the comparison
        # plt.figure()
        # plt.plot(specific_loudness_selected, label='Computed Specific Loudness Selected')
        # plt.plot(expected_specific_loudness, '--', label='Expected Specific Loudness Selected')
        # plt.legend()
        # plt.title('Excitation to Specific Loudness Comparison (Selected Indices)')
        # plt.xlabel('Selected Indices')
        # plt.ylabel('Specific Loudness (Sone)')
        # plt.grid(True)
        # plt.show()

    def test_filtered_signal_to_monaural_instantaneous_specific_loudness_selected(self):
        """Test selected point against expected values."""
        # Example parameters (consistent with expected value generation)
        frequency = self.frequency  # Hz
        duration = self.duration  # seconds
        rate = self.sample_rate
        db_max = self.db_max
        filter_filename = self.filter_filename

        # Synthesize sound
        sound = tvl.synthesize_sound(frequency, duration, rate)

        # Filter sound
        cochlea_filtered = tvl.sound_field_to_cochlea(sound, filter_filename)

        # Call the function under test
        ist_loudness_left, ist_loudness_right =\
            tvl.filtered_signal_to_monaural_instantaneous_specific_loudness(
              cochlea_filtered, rate, db_max)

        # Define selected indices to test
        selected_segments = [0, 25, 50, 100]
        erb_index = 1  # Corresponds to ist_loudness_left[segment][erb_index]

        # Extract instantaneous specific loudness at selected segments and ERB index
        ist_left_selected = jnp.array([ist_loudness_left[seg][erb_index] for seg in selected_segments])
        ist_right_selected = jnp.array([ist_loudness_right[seg][erb_index] for seg in selected_segments])

        # Load expected instantaneous specific loudness (manually input)
        expected_ist_left = self.expected_outputs['instantaneous_specific_loudness_left_selected']
        expected_ist_right = self.expected_outputs['instantaneous_specific_loudness_right_selected']

        # Optional: Plot for manual inspection
        plt.figure(figsize=(6, 10))
        plt.subplot(2, 1, 1)
        plt.plot(expected_ist_left, 'x', label='Expected IST Loudness Left Selected')
        plt.plot(ist_left_selected, label='Computed IST Loudness Left Selected')
        plt.title('Instantaneous Specific Loudness Comparison (Selected Indices)')
        plt.xlabel('Selected Indices')
        plt.ylabel('Specific Loudness (Sone)')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(expected_ist_right, 'x', label='Expected IST Loudness Right Selected')
        plt.plot(ist_right_selected, label='Computed IST Loudness Right Selected')
        plt.legend()
        plt.xlabel('Selected Indices')
        plt.ylabel('Specific Loudness (Sone)')
        plt.grid(True)
        plot_filename_pchip = os.path.join(
            'results', 'test_instantaneous_specific_loudness.png')
        plt.savefig(plot_filename_pchip)

        # Assertions for left ear
        self.assertTrue(
            jnp.all(ist_left_selected >= 0),
            msg="Selected instantaneous specific loudness left contains negative values."
        )
        np.testing.assert_allclose(
            ist_left_selected,
            expected_ist_left,
            rtol=1e-5,
            err_msg="Selected instantaneous specific loudness left does not match expected."
        )

        # Assertions for right ear
        self.assertTrue(
            jnp.all(ist_right_selected >= 0),
            msg="Selected instantaneous specific loudness right contains negative values."
        )
        np.testing.assert_allclose(
            ist_right_selected,
            expected_ist_right,
            rtol=1e-5,
            err_msg="Selected instantaneous specific loudness right does not match expected."
        )


    def test_get_alpha_and_p_functions(self):
        """Test get_alpha and get_p functions for correctness."""
        f = jnp.array([500, 1000, 2000, 4000, 8000])  # Hz
        alpha = tvl.get_alpha(f)
        p = tvl.get_p(f)

        # Since get_alpha and get_p depend on internal interpolations and tables,
        # we'll perform sanity checks based on expected behavior
        self.assertEqual(len(alpha), len(f), "get_alpha output length mismatch.")
        self.assertEqual(len(p), len(f), "get_p output length mismatch.")
        self.assertTrue(jnp.all(alpha > 0), "Alpha values should be positive.")
        self.assertTrue(jnp.all(p > 0), "P values should be positive.")

    def test_interpolation(self):
        """Test the interpolation function with 'pchip' and 'linear' methods."""
        # Test data
        x = jnp.arange(0, 1, 0.1)
        y = jnp.sin(x * 2 * jnp.pi)
        x_probe = jnp.linspace(0, 1, 1000)
        y_true = jnp.sin(x_probe * 2 * jnp.pi)

        # PCHIP Interpolation
        y_pchip = tvl.interpolation(x, y, x_probe, method='pchip')
        error_pchip = y_true - y_pchip
        std_error_pchip = jnp.std(error_pchip)

        # Assert standard error for pchip
        self.assertLess(std_error_pchip, 0.04, "Standard error too high for pchip interpolation")

        # Linear Interpolation
        y_linear = tvl.interpolation(x, y, x_probe, method='linear')
        error_linear = y_true - y_linear
        std_error_linear = jnp.std(error_linear)

        # Assert standard error for linear
        self.assertLess(std_error_linear, 0.05, "Standard error too high for linear interpolation")

        # Optionally, plot and save the interpolation results
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, y, 'o', label='Original Data')
        plt.plot(x_probe, y_pchip, '.', label='PCHIP Interpolated')
        plt.legend()
        plt.title('Interpolation with PCHIP')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.hist(error_pchip, bins=50, alpha=0.7, label='PCHIP Error')
        plt.xlabel('Error')
        plt.title('Error Distribution (PCHIP)')
        plt.grid(True)
        plt.legend()

        plot_filename_pchip = os.path.join('results', 'test_interpolation_pchip.png')
        plt.savefig(plot_filename_pchip)
        plt.close()
        self.assertTrue(os.path.exists(plot_filename_pchip),
                        msg="PCHIP interpolation plot file was not created.")

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, y, 'o', label='Original Data')
        plt.plot(x_probe, y_linear, '.', label='Linear Interpolated')
        plt.legend()
        plt.title('Interpolation with Linear')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.hist(error_linear, bins=50, alpha=0.7, label='Linear Error')
        plt.xlabel('Error')
        plt.title('Error Distribution (Linear)')
        plt.grid(True)
        plt.legend()

        plot_filename_linear = os.path.join('results', 'test_interpolation_linear.png')
        plt.savefig(plot_filename_linear)
        plt.close()
        self.assertTrue(os.path.exists(plot_filename_linear),
                        msg="Linear interpolation plot file was not created.")

    def test_agc_functions(self):
        """Test both AGC next frame functions with known inputs."""
        # Test agc_next_frame_of_vector
        v_last_frame = jnp.array([1.0, 2.0, 3.0])
        v_this_input = jnp.array([2.0, 1.5, 4.0])
        aA = 0.5  # Attack parameter
        aR = 0.1  # Release parameter

        expected_output_of_vector = jnp.array([
            aA * v_this_input[0] + (1 - aA) * v_last_frame[0],  # 0.5*2 + 0.5*1 = 1.5
            aR * v_this_input[1] + (1 - aR) * v_last_frame[1],  # 0.1*1.5 + 0.9*2 = 1.85
            aA * v_this_input[2] + (1 - aA) * v_last_frame[2],  # 0.5*4 + 0.5*3 = 3.5
        ])

        output_of_vector = tvl.agc_next_frame_of_vector(v_last_frame, v_this_input, aA, aR)
        np.testing.assert_allclose(
            output_of_vector,
            expected_output_of_vector,
            rtol=1e-5,
            err_msg="AGC next frame of vector output does not match expected."
        )
        # Test agc_next_frame
        # Attack condition
        d_last_frame_attack = 1.0
        d_this_input_attack = 2.0
        expected_attack = aA * d_this_input_attack + (1 - aA) * d_last_frame_attack  # 1.5
        output_attack = tvl.agc_next_frame(d_last_frame_attack, d_this_input_attack, aA, aR)
        self.assertAlmostEqual(
            output_attack,
            expected_attack,
            places=5,
            msg="AGC next frame attack output does not match expected."
        )

        # Release condition
        d_last_frame_release = 2.0
        d_this_input_release = 1.0
        expected_release = aR * d_this_input_release + (1 - aR) * d_last_frame_release  # 1.9
        output_release = tvl.agc_next_frame(d_last_frame_release, d_this_input_release, aA, aR)
        self.assertAlmostEqual(
            output_release,
            expected_release,
            places=5,
            msg="AGC next frame release output does not match expected."
        )

        # Equal inputs
        d_last_frame_equal = 1.0
        d_this_input_equal = 1.0
        expected_equal = d_this_input_equal  # 1.0
        output_equal = tvl.agc_next_frame(d_last_frame_equal, d_this_input_equal, aA, aR)
        self.assertAlmostEqual(
            output_equal,
            expected_equal,
            places=5,
            msg="AGC next frame equal input output does not match expected."
        )

    def test_input_level_per_erb(self):
        """Test the input levels per ERB calculation."""
        f = jnp.array([1000.0, 2000.0])
        in_levels = jnp.array([60.0, 70.0])
        input_levels = tvl.input_level_per_erb(f, in_levels)
        # Check if input_levels is a scalar or array. Based on the function, it should return an array
        # of the same length as 'in_levels'
        self.assertEqual(
            len(input_levels),
            len(in_levels),
            msg="Input levels per ERB length mismatch."
        )
        self.assertTrue(
            jnp.all(input_levels >= 0),
            msg="Input levels should be non-negative."
        )

    def test_synthesize_sound(self):
        """Test the sound synthesis function."""
        frequency = 1000  # Hz
        duration = 0.1    # seconds
        rate = 32000      # Hz
        sound = tvl.synthesize_sound(frequency, duration, rate)
        expected_samples = int(rate * duration)
        self.assertEqual(sound.shape, (expected_samples, 2), "Synthesized sound shape mismatch.")
        # Verify amplitude scaling (should be 10 dB below full scale)
        # Full scale amplitude is 1.0, 10 dB below is 10^(-10/20) = 0.316227766
        self.assertAlmostEqual(
            jnp.max(jnp.abs(sound)),
            0.316227766,
            places=5,
            msg="Synthesized sound amplitude scaling incorrect."
        )
        
        # Optionally, plot and save the waveform for manual inspection
        # plt.figure(figsize=(10, 4))
        # t = jnp.linspace(0, duration, expected_samples, endpoint=False)
        # plt.plot(t, sound[:, 0], label='Left Channel')
        # plt.plot(t, sound[:, 1], label='Right Channel', linestyle='--')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Amplitude')
        # plt.title('Synthesized 1 kHz Tone')
        # plt.legend()
        # plt.grid(True)
        # plot_filename = os.path.join('results', 'test_synthesize_sound_plot.png')
        # plt.savefig(plot_filename)
        # plt.close()
        # self.assertTrue(
        #     os.path.exists(plot_filename),
        #     msg="Synthesized sound plot file was not created."
        # )
        

    def test_excitation_threshold_tvl(self):
        """Test the excitation threshold calculation."""
        f = jnp.array([50, 100, 500, 1000, 2000])  # Hz
        expected = jnp.array([28.18, 15.68,  3.63,  3.63,  3.63])

        output = tvl.excitation_threshold_tvl(f)
        np.testing.assert_allclose(
            output,
            expected,
            rtol=1e-2,
            err_msg="Excitation threshold calculation does not match expected."
        )

    def test_get_g_tvl(self):
        """Test the cochlear amplifier gain calculation."""
        f = jnp.array([125, 250, 500, 1000, 2000, 4000, 8000])
        g = tvl.get_g_tvl(f)

        # Expected gain values calculated based on the formula in get_g_tvl
        # linear_threshold = 10 ** (excitation_threshold_tvl(f) / 10)
        # out = 10 ** (3.63 / 10) / linear_threshold
        thresholds = tvl.excitation_threshold_tvl(f)
        linear_threshold = 10 ** (thresholds / 10)
        expected_g = 10 ** (3.63 / 10) / linear_threshold
        np.testing.assert_allclose(
            g,
            expected_g,
            rtol=1e-4,
            err_msg="Cochlear amplifier gain calculation does not match expected."
        )


if __name__ == '__main__':
    absltest.main()
