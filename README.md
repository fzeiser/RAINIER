## RAINIER - Randomizer of Assorted Initial Nuclear Intensities and Emissions of Radiation

#### This is the current repository for the code RAINIER, which is described in following paper:
L.E.Kirsch, L.A.Bernstein. RAINIER: A simulation tool for distributions of excited nuclear states and cascade fluctuations. NIM A. Volume 892, 2018, Pages 30-40

DOI:    https://doi.org/10.1016/j.nima.2018.02.096  
See the arXiv: https://arxiv.org/abs/1709.04006

To cite the latest published version of RAINIER, you can *in addition* (not replacing the above) cite following, or create you own release: [![DOI](https://zenodo.org/badge/120598933.svg)](https://zenodo.org/badge/latestdoi/120598933)

## Abstract:
A new code has been developed named RAINIER that simulates the $\gamma$-ray decay of discrete and quasi-continuum nuclear levels for a user-specified range of energy, angular momentum, and parity including a realistic treatment of level spacing and transition width fluctuations. A similar program, DICEBOX, uses the Monte Carlo method to simulate level and width fluctuations but is restricted to $\gamma$-ray decay from no more than two initial states such as de-excitation following thermal neutron capture. On the other hand, modern reaction codes such as TALYS and EMPIRE populate a wide range of states in the residual nucleus prior to $\gamma$-ray decay, but do not go beyond the use of deterministic functions and therefore neglect cascade fluctuations. This combination of capabilities allows RAINIER to be used to determine quasi-continuum properties through comparison with experimental data. Several examples are given that demonstrate how cascade fluctuations influence experimental high-resolution $\gamma$-ray spectra from reactions that populate a wide range of initial states. 

## Installation and usage:
Fork/clone this repository or download zipfiles and extract to `/path/to/RAINIER`, eg. in your home folder

Location independent usage:

 * Set the environment variable `RAINIER_PATH`; To do this, execute the following in terminal, or save in `.bashrc` / `.bash_profile`. *Remember* to restart the terminal, such that the variable is actually exported!
    
    ```shell
    export RAINIER_PATH=/path/to/RAINIER
    ```

 * `cd` to `sample_folder` or create a `settings.h` file using this sceleton

    ```shell
    cd /path/to/RAINIER/sample_folder
    ../runRAINIER.sh, #or $RAINIER_PATH/runRAINIER.sh, if you create it in a different location
    ```

Possible error messages:
 * You might need to give you the permissions to execute the expect script:

    ```shell
    chmod u+x /path/to/RAINIER/runRAINIER.sh
    ```

### Running the Analysis script separately:
Run in the directory with your simulation results:

    root /path/to/Analyze.C++
Followed, for example by `AnalyzeGamma()`
