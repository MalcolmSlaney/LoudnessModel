import tvl2018 as tvl
# to specify rate, add rate = your_rate_value in the parameters
# to specify output_path, add output_path = your_output_path in the parameters

# Add more tests

def test_loudness_model():
    filename_or_sound = 'synthesize_1khz_100ms' # filename_or_sound must be a string, numpy array, or one of the provided synthesized sounds ('synthesize_{}khz_{}ms')
    db_max = 50
    filename_filter = 'transfer functions/ff_32000.mat'
    loudness, short_term_loudness, long_term_loudness = tvl.main_tv2018(filename_or_sound, db_max, filename_filter)
    print("Test completed successfully. Figure and results text file saved in results folder.")
    
