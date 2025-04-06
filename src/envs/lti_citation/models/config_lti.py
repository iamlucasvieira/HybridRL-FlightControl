from typing import Optional

from pydantic import BaseModel, field_validator, ConfigDict, ValidationInfo


class SymmetricDerivatives(BaseModel):
    """Symmetric derivatives."""

    model_config = ConfigDict(extra="forbid")

    o: Optional[float] = None
    u: float
    a: float
    a_dot: float
    q: float
    de: float



class AsymmetricDerivative(BaseModel):
    """Asymmetric derivatives."""

    model_config = ConfigDict(extra="forbid")

    b: float
    p: float
    r: float
    da: float
    dr: float



class SymmetricData(BaseModel):
    """Data validation for the symmetric aircraft model."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    c_bar: float
    mu_c: float
    ky_2: float
    x_cg: float
    cx: SymmetricDerivatives
    cz: SymmetricDerivatives
    cm: SymmetricDerivatives

    @field_validator("cx", "cz")
    def check_zero_derivative(cls, v: SymmetricDerivatives) -> SymmetricDerivatives:
        if v.o is None:
            raise ValueError("A value for o must be provided.")
        return v


class AssymmetricData(BaseModel):
    """Data validation for the asymmetric aircraft model."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    b: float
    mu_b: float
    kx_2: float
    kz_2: float
    kxz: float
    c_l: float
    cy: AsymmetricDerivative
    cl: AsymmetricDerivative
    cn: AsymmetricDerivative


class AircraftData(BaseModel):
    """Aircraft _data validation."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str
    v: float
    m: float
    s: float
    l_h: float
    symmetric: Optional[SymmetricData]
    asymmetric: Optional[AssymmetricData]


    @field_validator("asymmetric")
    def check_type(cls, v: AsymmetricDerivative, info: ValidationInfo) -> AsymmetricDerivative:
        """Required either symmetric or asymmetric variables"""
        if not info.data.get("symmetric") and not v:
            raise ValueError("Either symmetric or asymmetric must be provided.")
        return v
