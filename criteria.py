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
        return (T==0)*(grn_energy>=proposed_energy)+(T!=0)*self.get_finite_T_p(grn_energy, proposed_energy, (T==0)+(T!=0)*T)

    @abc.abstractmethod
    def get_finite_T_p(self, grn_energy: int, proposed_energy: int, T: float) -> float:
        '''Evaluate acceptance probability at finite temperature.'''


class MetropolisCriterion(AcceptanceCriterion):
    '''Metropolis-Hastings acceptance criterion.'''
    def get_finite_T_p(self, grn_energy: int, proposed_energy: int, T: float) -> float:
        '''Evaluate acceptance probability at finite temperature.'''
        return jnp.exp((grn_energy-proposed_energy)/T)*(grn_energy<proposed_energy)+1.0*(grn_energy>=proposed_energy)


class GlauberCriterion(AcceptanceCriterion):
    '''Glauber acceptance criterion.'''
    def get_finite_T_p(self, grn_energy: int, proposed_energy: int, T: float) -> float:
        '''Evaluate acceptance probability at finite temperature.'''
        return 1.0/(1.0+jnp.exp((proposed_energy-grn_energy)/T))



# class AcceptanceCriterion(abc.ABC):
#     '''Base class for acceptance criteria.'''
#     def __call__(self, grn_energy: int, proposed_energy: int, T: float) -> float:
#         '''Evaluate acceptance probability.

#         Parameters
#         ----------
#         grn_energy : jnp.ndarray, 0D, int
#         Pseudo-Hamiltonian of GRN state

#         proposed_energy : jnp.ndarray, 0D, int
#             Pseudo-Hamiltonian of proposed GRN state

#         T : jnp.ndarray, 0D, float
#             Pseudo-temperature
        
#         Returns
#         -------
#         p : jnp.ndarray, 0D, float
#             Acceptance probability
#         '''
#         # T=0 acceptance probability is always Indicator(grn_energy >= proposed_energy)
#         def zero_temperature_case():
#             return jnp.array(grn_energy >= proposed_energy).astype(float)

#         def finite_temperature_case():
#             print('I am being evaluated.')
#             return self.get_finite_T_p(grn_energy, proposed_energy, T)

#         return lax.cond(
#             T == 0,
#             zero_temperature_case,
#             finite_temperature_case
#         )
#         # if T == 0: return jnp.array(0.0) if grn_energy < proposed_energy else jnp.array(1.0)
#         # else: return self.get_finite_T_p(grn_energy, proposed_energy, T)

#     @abc.abstractmethod
#     def get_finite_T_p(self, grn_energy: int, proposed_energy: int, T: float) -> float:
#         '''Evaluate acceptance probability at finite temperature.'''


# class MetropolisCriterion(AcceptanceCriterion):
#     '''Metropolis-Hastings acceptance criterion.'''
#     def get_finite_T_p(self, grn_energy: int, proposed_energy: int, T: float) -> float:
#         '''Evaluate acceptance probability at finite temperature.'''
#         def relaxation_case():
#             return jnp.array(1.0)

#         def excitation_case():
#             return jnp.exp((grn_energy-proposed_energy)/T)

#         return lax.cond(
#             grn_energy >= proposed_energy,
#             relaxation_case,
#             excitation_case
#         )
#         # if grn_energy >= proposed_energy: return jnp.array(1.0)
#         # else: return jnp.exp((grn_energy-proposed_energy)/T)


# class GlauberCriterion(AcceptanceCriterion):
#     '''Glauber acceptance criterion.'''
#     def get_finite_T_p(self, grn_energy: int, proposed_energy: int, T: float) -> float:
#         '''Evaluate acceptance probability at finite temperature.'''
#         return 1.0/(1.0+jnp.exp((proposed_energy-grn_energy)/T))
    