#!/usr/bin/env bash

# GW150914_data0_1126259462-4_generation
# PARENTS 
# CHILDREN GW150914_data0_1126259462-4_sampling
if [[ "GW150914_data0_1126259462-4_generation" == *"$1"* ]]; then
    echo "Running: /home/qubit/Documents/Master's Thesis/Original Dingo/original-dingo-venv/bin/dingo_pipe_generation outdir_GW150914/GW150914_config_complete.ini --label GW150914_data0_1126259462-4_generation --idx 0 --trigger-time 1126259462.4 --outdir outdir_GW150914"
    /home/qubit/Documents/Master's Thesis/Original Dingo/original-dingo-venv/bin/dingo_pipe_generation outdir_GW150914/GW150914_config_complete.ini --label GW150914_data0_1126259462-4_generation --idx 0 --trigger-time 1126259462.4 --outdir outdir_GW150914
fi

# GW150914_data0_1126259462-4_sampling
# PARENTS GW150914_data0_1126259462-4_generation
# CHILDREN GW150914_data0_1126259462-4_sampling_plot
if [[ "GW150914_data0_1126259462-4_sampling" == *"$1"* ]]; then
    echo "Running: /home/qubit/Documents/Master's Thesis/Original Dingo/original-dingo-venv/bin/dingo_pipe_sampling outdir_GW150914/GW150914_config_complete.ini --label GW150914_data0_1126259462-4_sampling --event-data-file outdir_GW150914/data/GW150914_data0_1126259462-4_generation_event_data.hdf5"
    /home/qubit/Documents/Master's Thesis/Original Dingo/original-dingo-venv/bin/dingo_pipe_sampling outdir_GW150914/GW150914_config_complete.ini --label GW150914_data0_1126259462-4_sampling --event-data-file outdir_GW150914/data/GW150914_data0_1126259462-4_generation_event_data.hdf5
fi

# GW150914_data0_1126259462-4_sampling_plot
# PARENTS GW150914_data0_1126259462-4_sampling
# CHILDREN 
if [[ "GW150914_data0_1126259462-4_sampling_plot" == *"$1"* ]]; then
    echo "Running: /home/qubit/Documents/Master's Thesis/Original Dingo/original-dingo-venv/bin/dingo_pipe_plot --label GW150914_data0_1126259462-4_sampling_plot --result outdir_GW150914/result/GW150914_data0_1126259462-4_sampling.hdf5 --outdir outdir_GW150914/result --corner --weights --log_probs"
    /home/qubit/Documents/Master's Thesis/Original Dingo/original-dingo-venv/bin/dingo_pipe_plot --label GW150914_data0_1126259462-4_sampling_plot --result outdir_GW150914/result/GW150914_data0_1126259462-4_sampling.hdf5 --outdir outdir_GW150914/result --corner --weights --log_probs
fi

