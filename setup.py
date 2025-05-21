from setuptools import setup, find_namespace_packages

setup(
    name="fashion-recommender",
    version="0.1.0",
    packages=find_namespace_packages(include=["*"]),
    package_dir={"": "."},
    install_requires=[
        "streamlit>=1.31.0",
        "numpy==1.24.3",
        "pandas==1.5.3",
        "plotly>=5.18.0",
        "matplotlib>=3.7.1",
        "statsmodels>=0.14.1",
        "joblib>=1.3.2",
        "scikit-learn>=1.0.2",
        "fastapi>=0.109.0",
        "uvicorn>=0.27.0",
        "python-dotenv>=1.0.0",
        "httpx>=0.27.0",
    ],
    python_requires=">=3.8",
)
