'''

defining system model parameters

'''

# Imports
import numpy as np
from dataclasses import dataclass


@dataclass
class SystemModelParams:
    """Class for setting parameters of a system model.
    Initialize the SystemModelParams object.

    Parameters:
        None

    Attributes:
        M (int): Number of sources.
        N (int): Number of sensors.
        T (int): Number of observations.
        signal_type (str): Signal type ("NarrowBand" or "Broadband").
        freq_values (list): Frequency values for Broadband signal.
        signal_nature (str): Signal nature ("non-coherent" or "coherent").

    Returns:
        None
    """

    M = None  # Number of sources
    N = None  # Number of sensors
    T = None  # Number of observations
    signal_type = "NarrowBand"  # Signal type ("NarrowBand" or "Broadband")
    signal_nature = "non-coherent"  # Signal nature ("non-coherent" or "coherent")

    def set_parameter(self, name: str, value):
        """
        Set the value of the desired system model parameter.

        Args:
            name(str): the name of the SystemModelParams attribute.
            value (int, float, optional): the desired value to assign.

        Returns:
            SystemModelParams: The SystemModelParams object.
        """
        self.__setattr__(name, value)
        return self

class SystemModel(object):
    def __init__(self, system_model_params: SystemModelParams):
        self.params = system_model_params

    def __str__(self):
        """Returns a string representation of the SystemModel object.
        ...

        """
        print("System Model Summery:")
        for key, value in self.__dict__.items():
            print(key, " = ", value)
        return "End of Model"
