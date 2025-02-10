# Numeric_methods

This repository contains solutions to tasks for the Numerical Methods course. The tasks are divided into stages (A-G) and are organized into corresponding folders. Each folder includes Python scripts that provide solutions for each task, detailed reports in PDF format, and conclusions for all tasks within the folder. Additionally, the repository contains a description of the environment setup in the file `env_Metody_Numeryczne.yml` and the project description with task requirements in `MEN_Projekt_Ind_MGR_24L.pdf`.

## Repository Contents

# Task Descriptions

1. **Folder `1/`**: Contains solutions for tasks A and B:
    - **Task A1**: Convert numbers between decimal, octal, and hexadecimal formats.
    - **Task A2**: Develop a function to calculate machine precision.
    - **Task A3**: Evaluate the relative and absolute error for approximating `sin(x)` using Taylor series.
    - **Task B4**: Solve systems of linear equations using Gaussian elimination.
    - **Task B6**: Verify positive-definiteness of matrices and solve equations using Cholesky decomposition.
    - **Task B7**: Implement Jacobi and Gauss-Seidel methods and compare their convergence.
   - `MEN_Proj_Ind_Oleg_Łyżwiński.pdf`: Report and conclusions for tasks A and B.

2. **Folder `2/`**: Contains solutions for tasks C, D, and E:
    - **Task C8**: Implement the Secant and Newton methods to solve equations.
    - **Task D9**: Analyze Runge's phenomenon using evenly spaced and Chebyshev nodes.
    - **Task D10**: Perform polynomial interpolation with Lagrange's method.
    - **Task D11**: Interpolate temperature data using Newton’s method.
    - **Task E12**: Perform least squares approximation for a given function.
    - **Task E13**: Fit an exponential model using linear regression.
    - `MEN_Proj_IND_Oleg_Łyżwiński_CDE.pdf`: Report and conclusions for tasks C, D, and E.

3. **Folder `3/`**: Contains solutions for tasks F and G:
    - **Task F14**: Implement Newton-Cotes quadrature methods for integration.
    - **Task F15**: Evaluate integration errors using the Newton-3/8 method.
    - **Task F16-17**: Use Gauss-Chebyshev and Gauss-Hermite quadrature methods.
    - **Task G18**: Compute derivatives numerically and compare with analytical results.
    - `MEN_Proj_IND_Oleg_Łyżwiński_FG.pdf`: Report and conclusions for tasks F and G.

4. **Folder `Difusion/`**: Contains the group project:
   - `Metody.py`: Implementation of algorithms for analyzing drug diffusion through biomembranes or biofilms. 
     - The goal of this project was to determine drug diffusion using phase-shift interferometry with a CMOS sensor in a Mach-Zehnder interferometer. Phase data reconstruction was used to analyze the biofilm behavior.
   - `MEN_Proj_Grup.pdf`: Report detailing the methodology and results of the group project.

5. **Additional files**:
   - `env_Metody_Numeryczne.yml`: Conda environment file for setting up the computational environment.
   - `MEN_Projekt_Ind_MGR_24L.pdf`: A document with the full descriptions and requirements for all tasks in the project.

## Environment Setup
To recreate the computational environment, use the Conda environment file:
```bash
conda env create -f env_Metody_Numeryczne.yml