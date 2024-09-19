import numpy as np
import os

from absl.testing import absltest

import tvl2018 as tvl

class LoudnessTest(absltest.TestCase):
  def test_basic_example(self):
    # filename_or_sound must be a string, numpy array, or one of the provided 
    # synthesized sounds ('synthesize_{}khz_{}ms')
    filename_or_sound = 'synthesize_1khz_100ms' 
    db_max = 50
    filter_filename = 'transfer functions/ff_32000.mat'
    debug_plot_filename = os.path.join(
        'results', 'synthesize_1khz_100ms_50dB_loudness_plot.png')
    debug_summary_filename = os.path.join(
        'results', 'synthesize_1khz_100ms_50dB_calibration_level_TVL_2018.txt')
    (loudness, 
     short_term_loudness, 
     long_term_loudness) = tvl.main_tv2018(
       filename_or_sound, db_max, filter_filename, 
       debug_plot=True,
       debug_plot_filename=debug_plot_filename,
       debug_summary_filename=debug_summary_filename)

    # Weak sanity tests 
    np.testing.assert_array_less(long_term_loudness, short_term_loudness)
    np.testing.assert_array_less(0.0, short_term_loudness)
    np.testing.assert_array_less(0.0, long_term_loudness)

    # Make sure debug files are created.
    self.assertTrue(os.path.exists(debug_plot_filename))
    self.assertTrue(os.path.exists(debug_summary_filename))


if __name__ == '__main__':
  absltest.main()