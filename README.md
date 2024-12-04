The rows of `eml_fate_network.npy` are:

0. Regulator gene IDs
1. Target gene IDs
2. Regulator effect on target

Gene IDs are in `gene_ids.csv`.
`0` is inhibition ($J=-1$); `1` is activation ($J=1$).

I need the following figures:

0. Histograms of time-averaged states at zero T and field of c-Kit, GATA1, and PU.1 for biological network
1. Histograms of frustrations of final states at zero T and field for biological and random networks
2. Mean frustration vs. temperature at zero field for biological network
3. Correlation times vs. temperature at zero field for biological and random networks
4. Susceptibility by temperature for biological network
5. Mean $q_{ab}$ by temperature at zero field for biological network
6. Topological speed limit at zero T for biological and random networks
7. Free energy vs. temperature for biological network (Appendix)
8. Table of gene interactions (Appendix)

I need the following simulations:

0. Biological network: (1000 replicas) x ($N^4$ timesteps) x (21 temperatures) at $h=0$ (`bio_vary_T`)
1. Biological network: (1000 replicas) x ($N^4$ timesteps) x (21 fields) at $T=T_c$ (`bio_vary_h`)
2. Biological network: (100000 replicas) x ($N^2$ timesteps) at $T=0$, $h=0$ (`bio_short`)
3. (1000 random network-replicas) x ($N^4$ timesteps) x (21 temperatures) at $h=0$ (`rand_vary_T`)
4. (1000 random network-replicas) x ($N^4$ timesteps) x (21 fields) at $T=T_c$ (`rand_vary_h`)
5. (100000 random network-replicas) x ($N^2$ timesteps) at $T=0$, $h=0$ (`rand_short`)
6. Random network: (1000 replicas) x ($N^4$ timesteps) x (21 temperatures) at $h=0$ (`rand_qab_vary_T`)
7. Random network: (1000 replicas) x ($N^4$ timesteps) x (21 temperatures) at $T=T_c$ (`rand_qab_vary_T`)
