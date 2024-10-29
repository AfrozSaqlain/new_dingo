# from dingo.core.models import PosteriorModel
# import dingo.gw.injection as injection
# from dingo.gw.noise.asd_dataset import ASDDataset

# main_pm = PosteriorModel(
#     device="cpu",
#     model_filename="./training/model_latest.pt", 
#     load_training_info=False
# )

# init_pm = PosteriorModel(
#     device='cpu',
#     model_filename="./training/model_030.pt",
#     load_training_info=False
# )

# injection_generator = injection.Injection.from_posterior_model_metadata(main_pm.metadata)
# asd_fname = main_pm.metadata["train_settings"]["training"]["stage_0"]["asd_dataset_path"]
# asd_dataset = ASDDataset(file_name=asd_fname)
# injection_generator.asd = {k:v[0] for k,v in asd_dataset.asds.items()}

# intrinsic_parameters = {
#     "mass_1": 35,
#     "mass_2": 50,
#     "a_1": 0.3,
#     "a_2": 0.5,
#     "tilt_1": 0.,
#     "tilt_2": 0.,
#     "phi_jl": 0.,
#     "phi_12": 0.,
#     "m_lens": 200.,
#     "y_lens": 0.12,
#     "z_lens": 0.,
# }

# extrinsic_parameters = {
#     'phase': 0.,
#     'theta_jn': 2.3,
#     'geocent_time': 0.,
#     'luminosity_distance': 1000.,
#     'ra': 0.,
#     'dec': 0.,
#     'psi': 0.,
# }

# theta = {**intrinsic_parameters, **extrinsic_parameters}
# strain_data = injection_generator.injection(theta)


# from dingo.gw.inference.gw_samplers import GWSampler

# # init_sampler = GWSampler(model=init_pm)
# sampler = GWSampler(model=main_pm)
# # sampler = GWSampler(model=main_pm, init_sampler=init_sampler, num_iterations=30)
# sampler.context = strain_data
# sampler.run_sampler(num_samples=50_000, batch_size=10_000)
# result = sampler.to_result()
# result.plot_corner()


from dingo.core.models.posterior_model import PosteriorModel
from dingo.gw.inference.gw_samplers import GWSampler
from dingo.gw.injection import Injection
from dingo.gw.noise.asd_dataset import ASDDataset
import os
import warnings

warnings.filterwarnings("ignore")

model_path = "training/model_latest.pt"
asd_path = "training_data/asd_dataset/asds_O3.hdf5"

# Load the network into the GWSampler class
pm = PosteriorModel(model_path, device="cpu")
sampler = GWSampler(model=pm)

# Generate an injection consistent with the data the model was trained on.
injection = Injection.from_posterior_model_metadata(pm.metadata)
injection.asd = ASDDataset(asd_path, ifos=["H1", "L1", "V1"])
theta = injection.prior.sample()
theta = {key: float(value) for key, value in theta.items()}
print(theta)
inj = injection.injection(theta)

# Generate 10,000 samples from the DINGO model based on the generated injection data.
sampler.context = inj
sampler.run_sampler(10_000)
result = sampler.to_result()

# # The following are only needed for importance-sampling the result.
# result.importance_sample(num_processes=8)

# Make a corner plot and save the result.
result.print_summary()
os.makedirs('inference', exist_ok=True)
kwargs = {"legend_font_size": 15, "truth_color": "black"}
result.plot_corner(parameters=["chirp_mass", "mass_ratio", "m_lens", "y_lens"],
                   filename="inference/corner.pdf",
                   truths=theta,
                   **kwargs)


result.to_file("inference/result.hdf5")