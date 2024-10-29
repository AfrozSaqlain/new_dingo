from pycbc.filter.matchedfilter import match
from pycbc.types import FrequencySeries
import h5py
import numpy as np
import gwmat
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
import os
import warnings

# Ignore ComplexWarnings
warnings.filterwarnings("ignore")


def process_file(file_no):
    output_lines = []
    # print(f"Processing file number: {file_no}")
    
    with h5py.File('./training_data/waveform_dataset.hdf5', 'r') as f:
        polarizations = f['polarizations']
        h_plus = polarizations['h_plus']
        h_cross = polarizations['h_cross']
        h_plus = FrequencySeries(h_plus[file_no], delta_f=0.25)
        h_cross = FrequencySeries(h_cross[file_no], delta_f=0.25)
        param = f['parameters'][file_no]

    cbc_and_lens_prms = {}
    for field_name in param.dtype.names:
        cbc_and_lens_prms[field_name] = param[field_name]

    cbc_and_lens_prms['coa_phase'] = cbc_and_lens_prms.pop('phase')
    cbc_and_lens_prms.pop("geocent_time")

    from sympy import symbols, Eq, solve

    # Define symbols
    chirp_mass, m1, q = symbols('chirp_mass m1 q', real=True, positive=True)

    # Express m2 in terms of m1 and q
    m2 = q * m1

    # Define the chirp mass equation
    chirp_mass_eq = Eq(chirp_mass, ((m1 * m2)**(3/5)) / ((m1 + m2)**(1/5)))

    # Solve for m1 in terms of chirp_mass and q
    m1_solution = solve(chirp_mass_eq, m1)[0]

    # Substitute m1 into m2 equation to get m2 in terms of chirp_mass and q
    m2_solution = q * m1_solution

    # Substitute specific values for chirp_mass and q
    chirp_mass_value = cbc_and_lens_prms['chirp_mass']
    mass_ratio_value = cbc_and_lens_prms['mass_ratio']

    # Perform substitution
    m1_numeric = m1_solution.subs({chirp_mass: chirp_mass_value, q: mass_ratio_value})
    m2_numeric = m2_solution.subs({chirp_mass: chirp_mass_value, q: mass_ratio_value})

    # Store the results in the dictionary
    cbc_and_lens_prms['mass_1'] = m1_numeric
    cbc_and_lens_prms['mass_2'] = m2_numeric
    cbc_and_lens_prms.pop('chirp_mass')
    cbc_and_lens_prms.pop('mass_ratio')

    converted_dict = {key: float(value) for key, value in cbc_and_lens_prms.items()}

    init_prms = dict(f_lower=20., f_final=1024, f_ref=20, delta_f=0.25, wf_domain="FD", wf_approximant="IMRPhenomPv2")

    prms = {**init_prms, **converted_dict, 'psi': 0.0, 'geocent_time': 0.0}

    res = gwmat.injection.generate_gw_polarizations_hp_hc(**prms)
    
    # Check if the lens mass is not zero
    if prms['m_lens'] != 0.0:
        wf_lensed = res['hp_FD_Lensed']

    if 'wf_lensed' in locals():
        match_result = match(h_plus, wf_lensed)
        output_lines.append(f"Results from file no. {file_no} (where m_lens: {converted_dict['m_lens']} and y_lens: {converted_dict['y_lens']}):")
        output_lines.append(str(match_result))
        
        # Ensure the output directory exists
        output_dir = 'output_logs/plots'
        os.makedirs(output_dir, exist_ok=True)

        # Save the plot
        plt.figure()  # Create a new figure for each plot
        plt.plot(wf_lensed.sample_frequencies, wf_lensed, linestyle='--', label='GWMAT')
        plt.plot(h_plus.sample_frequencies, h_plus, linestyle=':', label='DINGO')
        plt.xscale('log')
        plt.xlim(10, 1024)
        plt.legend()
        plt.title('Lensed')

        # Use a unique filename for each plot
        filename = f"{output_dir}/lensed_file_{file_no}_m_lens_{converted_dict['m_lens']}_y_lens_{converted_dict['y_lens']}.png"
        plt.savefig(filename)
        plt.close()  # Close the figure to free memory

    return output_lines

def main():
    # Create output logs directory if it doesn't exist
    os.makedirs('output_logs', exist_ok=True)
    
    # Create a Pool for multiprocessing
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        file_numbers = range(50)  # File numbers from 0 to 99

        # Use tqdm to display a progress bar
        with tqdm(total=len(file_numbers), desc="Processing files") as pbar:
            for result in pool.imap(process_file, file_numbers):
                # Write output to file
                if result:
                    with open('output_logs/all_outputs.txt', 'a') as output_file:
                        output_file.write("\n".join(result) + "\n")
                pbar.update(1)

if __name__ == "__main__":
    main()
