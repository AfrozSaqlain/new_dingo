{"legend_font_size": 15, "truth_color": "black"}
result.plot_corner(parameters=["chirp_mass", "mass_ratio"],
                   filename="03_inference/injection/corner.pdf",
                   truths=theta,
                   **kwargs)