This repository contains the scripts used within {placeholder for publication}.

This repository contains the necessary scripts for determining the bending modulus of membranes using 3 approaches. The q^-4 approach utilizes the fourier transform of the lipid bilayer, transforming from a height function into the frequency domain of undulations which can then be used to extract the bending modulus through a linear regression within log space. The Bedeaux-Weeks Density Correlation Functions approach implements density correlation functions into capillary wave theory to then extract the bending modulus. Finally, Real Space Fluctuations (RSF) approach is used by doing a neighbor search and takes the difference in angle of lipid pairs to determine the splay, which is then used to compute the bending modulus. 

MDP Files contains the .mdp file parameters utilized in the membrane simulations in GROMACS 2023.2.

Analysis files contain the scripts utilized for analysis of trajectories. A second README is present within explaining what can be found in the python scripts. 
