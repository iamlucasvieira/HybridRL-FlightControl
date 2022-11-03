from pydantic import BaseModel, validator, Extra
import yaml
from helpers.paths import Path
from typing import Optional


class SymmetricDerivatives(BaseModel):
    """Symmetric derivatives."""
    o: Optional[float]
    u: float
    a: float
    a_dot: float
    q: float
    de: float

    class Config:
        extra = Extra.forbid


class AsymmetricDerivative(BaseModel):
    """Asymmetric derivatives."""
    b: float
    p: float
    r: float
    da: float
    dr: float

    class Config:
        extra = Extra.forbid


class SymmetricData(BaseModel):
    """Data validation for the symmetric aircraft model."""
    c_bar: float
    mu_c: float
    ky_2: float
    x_cg: float
    cx: SymmetricDerivatives
    cz: SymmetricDerivatives
    cm: SymmetricDerivatives

    @validator('cx', 'cz')
    def check_zero_derivative(cls, v: SymmetricDerivatives) -> SymmetricDerivatives:
        if v.o is None:
            raise ValueError(f'A value for _o must be provided.')
        return v

    class Config:
        extra = Extra.forbid


class AssymmetricData(BaseModel):
    """Data validation for the asymmetric aircraft model."""
    b: float
    mu_b: float
    kx_2: float
    kz_2: float
    kxz: float
    c_l: float
    cy: AsymmetricDerivative
    cl: AsymmetricDerivative
    cn: AsymmetricDerivative

    class Config:
        extra = Extra.forbid


class AircraftData(BaseModel):
    """Aircraft data validation."""
    name: str
    v: float
    m: float
    s: float
    l_h: float
    symmetric: Optional[SymmetricData]
    asymmetric: Optional[AssymmetricData]

    class Config:
        extra = Extra.forbid

    @validator('symmetric', always=True)
    def check_type(cls, v: AsymmetricDerivative, values: dict) -> AsymmetricDerivative:
        """Required either symmetric or asymmetric variables"""
        if not values.get('asymmetric') and not v:
            raise ValueError('Either symmetric or asymmetric must be provided.')
        return v


def load_aircraft(filename: str) -> AircraftData:
    """Load aircraft data from YAML file, validate, and return data object."""

    with open(Path.aircraft_data / filename) as f:
        aircraft = yaml.safe_load(f)
    return AircraftData(**aircraft)
