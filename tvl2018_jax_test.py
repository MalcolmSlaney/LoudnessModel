import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np  # For testing
from absl.testing import absltest

from jax import random, lax

import transfer_functions 
import tvl2018_jax as tvl

# This line is only necessary to match the test's precision
jax.config.update("jax_enable_x64", True)

class LoudnessModelTests(absltest.TestCase):
    """Test suite for the TVL2018 loudness model implementation."""

    def test_basic_example(self):
        """Test the main_tv2018 function with a synthesized 1 kHz tone."""
        # Ensure results directory exists
        os.makedirs('results', exist_ok=True)

        frequency = 1000  # Hz
        duration = 0.1    # seconds
        rate = 32000      # Hz
        sound = tvl.synthesize_sound(frequency, duration, rate)

        # Here, you can modify the input values into tvl.main_tv2018
        db_max = 50  # Example SPL value
        _, short_term_loudness, long_term_loudness = tvl.main_tv2018(
            sound,
            db_max,
            transfer_functions.ff_32000,
            rate=rate,
        )

        # Plotting the short-term and long-term loudness for visualization

        plt.figure(figsize=(10, 5))
        short_term_loudness_np = np.asarray(short_term_loudness)
        long_term_loudness_np = np.asarray(long_term_loudness)
        plt.plot(short_term_loudness_np, label='Short-term Loudness')
        plt.plot(long_term_loudness_np, label='Long-term Loudness')
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
        cls.sound = tvl.synthesize_sound(cls.frequency, cls.duration, 
                                         cls.sample_rate)
        cls.filter = transfer_functions.ff_32000

        cls.loudness, cls.short_term_loudness, cls.long_term_loudness = tvl.main_tv2018(
            cls.sound,
            cls.db_max,
            cls.filter,
            cls.sample_rate,
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

    def test_peak_constrained_power_jax(self):
        """Test that phase adjustments increase power/loudness while maintaining peak constraint using JAX."""
        # Parameters
        duration, rate = 0.1, 32000  # seconds, Hz
        fundamental = 100  # Hz
        n_harmonics = 10
        peak_constraint = 0.8
        db_max = 50
        filter = transfer_functions.ff_32000

        def process_signal(mono_signal):
            """Process mono signal: normalize, make stereo, calculate metrics."""
            signal = mono_signal * (peak_constraint / jnp.max(jnp.abs(mono_signal)))
            stereo = jnp.column_stack((signal, signal))
            rms = jnp.sqrt(jnp.mean(signal ** 2))
            loudness, _, _ = tvl.main_tv2018(stereo, db_max, filter, rate=rate)
            return rms, loudness

        def create_signal(magnitudes, phases):
            """Create harmonic signal using JAX."""
            t = jnp.linspace(0, duration, int(rate * duration), endpoint=False)
            freqs = fundamental * jnp.arange(1, n_harmonics + 1)
            # Reshape magnitudes and phases to (n_harmonics, 1)
            magnitudes = magnitudes[:, None]
            phases = phases[:, None]
            # Compute the signals
            signals = magnitudes * jnp.cos(2 * jnp.pi * freqs[:, None] * t + phases)
            signal = jnp.sum(signals, axis=0)
            return signal

        # Create baseline (cosine phase)
        base_magnitudes = 1.0 / jnp.arange(1, n_harmonics + 1)
        baseline_phases = jnp.zeros(n_harmonics)
        baseline = create_signal(base_magnitudes, baseline_phases)
        baseline_rms, baseline_loudness = process_signal(baseline)

        # All-pass filter parameters
        freq_shift, bandwidth = 500, 250  # Hz
        omega_d = 2 * jnp.pi * freq_shift / rate
        bw = 2 * jnp.pi * bandwidth / rate
        c = (jnp.tan(bw / 2) - 1) / (jnp.tan(bw / 2) + 1)
        d = -jnp.cos(omega_d)
        b_allpass = jnp.array([-c, d * (1 - c), 1.0])
        a_allpass = jnp.array([1.0, d * (1 - c), -c])
        
        def lfilter_jax(b, a, x):
            """Implement a second-order IIR filter in JAX."""
            # Normalize coefficients if a[0] != 1
            b = b / a[0]
            a = a / a[0]
            a_rest = a[1:]

            def step(carry, x_n):
                x_n_1, x_n_2, y_n_1, y_n_2 = carry
                y_n = (b[0] * x_n + b[1] * x_n_1 + b[2] * x_n_2
                      - a_rest[0] * y_n_1 - a_rest[1] * y_n_2)
                new_carry = (x_n, x_n_1, y_n, y_n_1)
                return new_carry, y_n

            # Initial conditions
            init_carry = (0.0, 0.0, 0.0, 0.0)

            _, y = lax.scan(step, init_carry, x)
            return y

        # Apply the all-pass filter using JAX
        filtered = lfilter_jax(b_allpass, a_allpass, baseline)
        filtered_rms, filtered_loudness = process_signal(filtered)

        # Random phases using JAX's random number generator
        key = random.PRNGKey(42)
        random_phases = random.uniform(key, shape=(n_harmonics,), minval=0.0, maxval=2 * jnp.pi)
        random_signal = create_signal(base_magnitudes, random_phases)
        random_rms, random_loudness = process_signal(random_signal)

        # Test that at least one method improves RMS and loudness
        best_rms = jnp.maximum(filtered_rms, random_rms)
        best_loudness = jnp.maximum(filtered_loudness, random_loudness)

        # Assertions
        self.assertGreater(best_rms, baseline_rms,
                          "Phase adjustment should increase RMS")
        self.assertGreater(best_loudness, baseline_loudness,
                          "Phase adjustment should increase loudness")
 
    def test_overall_loudness(self):
        """Test overall loudness against expected maximum long-term loudness."""
        # Ensure results directory exists
        os.makedirs('results', exist_ok=True)

        loudness = self.loudness

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

        # Assert first five short-term loudness values
        np.testing.assert_allclose(
            short_term_loudness[:5],
            self.expected_outputs['short_term_loudness_first5'],
            rtol=1e-3,
            err_msg="First five short-term loudness values do not match expected."
        )

    def test_long_term_loudness(self):
        """Test long-term loudness against expected first five values."""
        # Ensure results directory exists
        os.makedirs('results', exist_ok=True)

        # Call main_tv2018
        long_term_loudness = self.long_term_loudness

        # Assert first five long-term loudness values
        np.testing.assert_allclose(
            long_term_loudness[:5],
            self.expected_outputs['long_term_loudness_first5'],
            rtol=1e-3,
            err_msg="First five long-term loudness values do not match expected."
        )

    def test_signal_segment_to_spectrum(self):
        """Test the signal segment to spectrum conversion against expected values."""
        # Use the filtered signal from test_basic_example
        # Synthesize and filter the sound
        sound = tvl.synthesize_sound(self.frequency, self.duration, 
                                     rate=self.sample_rate)
        cochlea_filtered = tvl.sound_field_to_cochlea(sound, self.filter)
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
            rtol=1e-3,
            err_msg="Left relevant frequencies do not match expected."
        )
        np.testing.assert_allclose(
            l_left_relevant[:5],
            self.expected_outputs['signal_segment_spectrum']['l_left_relevant_first5'],
            rtol=1e-3,
            err_msg="Left relevant levels do not match expected."
        )

        # Compare the first five relevant frequencies and levels for right
        np.testing.assert_allclose(
            f_right_relevant[:5],
            self.expected_outputs['signal_segment_spectrum']['f_right_relevant_first5'],
            rtol=1e-3,
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
            rtol=1e-3,
            err_msg="Selected excitation pattern values do not match expected."
        )

        # Optional: Plot for manual inspection
        # Uncomment the following lines if you wish to visualize the comparison
        # plt.figure()
        # plt.plot(np.asarray(excitation_selected), label='Computed Excitation Selected')
        # plt.plot(np.asarray(expected_excitation), '--', label='Expected Excitation Selected')
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
            rtol=1e-3,
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
        filter = self.filter

        # Synthesize sound
        sound = tvl.synthesize_sound(frequency, duration, rate)

        # Filter sound
        cochlea_filtered = tvl.sound_field_to_cochlea(sound, filter)

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

        # Convert JAX arrays to NumPy arrays for plotting
        ist_left_selected_np = np.asarray(ist_left_selected)
        ist_right_selected_np = np.asarray(ist_right_selected)
        expected_ist_left_np = np.asarray(expected_ist_left)
        expected_ist_right_np = np.asarray(expected_ist_right)

        # Assertions for left ear
        self.assertTrue(
            jnp.all(ist_left_selected >= 0),
            msg="Selected instantaneous specific loudness left contains negative values."
        )
        np.testing.assert_allclose(
            ist_left_selected_np,
            expected_ist_left_np,
            rtol=1e-4,
            err_msg="Selected instantaneous specific loudness left does not match expected."
        )

        # Assertions for right ear
        self.assertTrue(
            jnp.all(ist_right_selected >= 0),
            msg="Selected instantaneous specific loudness right contains negative values."
        )
        np.testing.assert_allclose(
            ist_right_selected_np,
            expected_ist_right_np,
            rtol=1e-3,
            err_msg="Selected instantaneous specific loudness right does not match expected."
        )

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

        # to plot, convert to numpy 
        x_np = np.asarray(x)
        y_np = np.asarray(y)
        y_pchip_np = np.asarray(y_pchip)
        error_pchip_plot = np.asarray(error_pchip)
        error_linear_plot = np.asarray(error_linear)
        

        # Optionally, plot and save the interpolation results
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x_np, y_np, 'o', label='Original Data')
        plt.plot(x_probe, y_pchip_np, '.', label='PCHIP Interpolated')
        plt.legend()
        plt.title('Interpolation with PCHIP')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.hist(error_pchip_plot, bins=50, alpha=0.7, label='PCHIP Error')
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
        plt.plot(x_np, y_np, 'o', label='Original Data')
        plt.plot(x_probe, y_linear, '.', label='Linear Interpolated')
        plt.legend()
        plt.title('Interpolation with Linear')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.hist(error_linear_plot, bins=50, alpha=0.7, label='Linear Error')
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
            rtol=1e-3,
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
            rtol=1e-3,
            err_msg="Excitation threshold calculation does not match expected."
        )


if __name__ == '__main__':
    absltest.main()
