# criteria.py: classes for Markov-chain Monte-Carlo acceptance criteria


import abc
from jax import lax
from jax import numpy as jnp


class AcceptanceCriterion(abc.ABC):
    '''Base class for acceptance criteria.'''
    def __call__(self, grn_energy: int, proposed_energy: int, T: float) -> float:
        '''Evaluate acceptance probability.

        Parameters
        ----------
        grn_energy : jnp.ndarray, 0D, int
            Pseudo-Hamiltonian of GRN state

        proposed_energy : jnp.ndarray, 0D, int
            Pseudo-Hamiltonian of proposed GRN state

        T : jnp.ndarray, 0D, float
            Pseudo-temperature
        
        Returns
        -------
        p : jnp.ndarray, 0D, float
            Acceptance probability
        '''
        # T == 0 acceptance probability is always Indicator(grn_energy >= proposed_energy)
        return jnp.where(T == 0,
                         jnp.array(grn_energy>=proposed_energy).astype(float),
                         self.get_finite_T_p(grn_energy, proposed_energy, (T==0)+(T!=0)*T)
                         )

    @abc.abstractmethod
    def get_finite_T_p(self, grn_energy: int, proposed_energy: int, T: float) -> float:
        '''Evaluate acceptance probability at finite temperature.'''


class MetropolisCriterion(AcceptanceCriterion):
    '''Metropolis-Hastings acceptance criterion.'''
    def get_finite_T_p(self, grn_energy: int, proposed_energy: int, T: float) -> float:
        '''Evaluate acceptance probability at finite temperature.'''
        return jnp.where(grn_energy < proposed_energy,
                         jnp.exp((grn_energy-proposed_energy)/T),
                         jnp.array(1.0)
                         )


class GlauberCriterion(AcceptanceCriterion):
    '''Glauber acceptance criterion.'''
    def get_finite_T_p(self, grn_energy: int, proposed_energy: int, T: float) -> float:
        '''Evaluate acceptance probability at finite temperature.'''
        return 1.0/(1.0+jnp.exp((proposed_energy-grn_energy)/T))
