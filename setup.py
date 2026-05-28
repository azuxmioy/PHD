from setuptools import setup, find_packages

setup(
    name="phd",
    version="0.1.0",
    description="PHD: Personalized 3D Human Body Fitting with Point Diffusion (ICCV 2025).",
    author="Hsuan-I Ho",
    url="https://github.com/azuxmioy/test_opensource",
    license="CC-BY-NC-4.0",
    packages=find_packages(include=["phd", "phd.*", "shapify", "shapify.*"]),
    include_package_data=True,
    package_data={
        "phd": ["../assets/*"],
    },
    python_requires=">=3.8",
)
