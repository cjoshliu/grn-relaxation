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
3. Frustration correlation times vs. temperature at zero field for biological network
4. Mean $q_{EA}$ by temperature at zero field for biological and random networks
5. Mean $q_{ab}$ by temperature at zero field for biological and random networks
6. Mean $q_{EA}$ by field at critical T for biological and random networks
7. Mean $q_{ab}$ by field at critical T for biological and random networks
8. Topological speed limit at zero T for biological and random networks
9. Free energy vs. temperature for biological network (Appendix)
10. Table of gene interactions (Appendix)

I need the following simulations:

0. Biological network: (1000 replicas) x ($N^4$ timesteps) x (21 temperatures)
1. Biological network: (1000 replicas) x ($N^4$ timesteps) x (21 fields)
2. Biological network: (100000 replicas) x ($N^2$ timesteps) at $T=0$, $h=0$
3. (1000 random networks) x ($N^4$ timesteps) x (21 temperatures)
4. (1000 random networks) x ($N^4$ timesteps) x (21 fields)
5. (100000 random networks) x ($N^2$ timesteps) at $T=0$, $h=0$
