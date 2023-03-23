from typing import Optional

from pydantic import BaseModel, Extra, validator


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

    @validator("cx", "cz")
    def check_zero_derivative(cls, v: SymmetricDerivatives) -> SymmetricDerivatives:
        if v.o is None:
            raise ValueError("A value for o must be provided.")
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
    """Aircraft _data validation."""

    name: str
    v: float
    m: float
    s: float
    l_h: float
    symmetric: Optional[SymmetricData]
    asymmetric: Optional[AssymmetricData]

    class Config:
        extra = Extra.forbid

    @validator("asymmetric", always=True)
    def check_type(cls, v: AsymmetricDerivative, values: dict) -> AsymmetricDerivative:
        """Required either symmetric or asymmetric variables"""
        if not values.get("symmetric") and not v:
            raise ValueError("Either symmetric or asymmetric must be provided.")
        return v
