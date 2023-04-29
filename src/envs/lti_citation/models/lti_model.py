"""Class that creates LTI aircraft model."""
from pathlib import Path

import numpy as np
import yaml

from envs.lti_citation.models.config_lti import AircraftData


class Aircraft:
    """Class that creates the LTI aircraft state space model."""

    CONFIGURATIONS = [
        "symmetric",
        "sp",
        "asymmetric",
    ]  # Available aircraft configurations

    def __init__(
        self,
        filename: str = "citation.yaml",
        auto_build: bool = True,
        configuration: str = "symmetric",
        dt: float = 0.01,
    ) -> None:
        """Initialize aircraft model.

        Args:
            filename (str): Aircraft data filename.
            auto_build (bool, optional): Automatically build state space model. Defaults to True.
            configuration (str, optional): Aircraft configuration. Defaults to symmetric.
            dt (float, optional): Time step. Defaults to 1e-3.

        """
        self.filename = filename
        self.data = self.load_aircraft()
        self.configuration = configuration
        self.dt = dt
        self.ss = None
        self.states = None
        self.current_state = None

        if auto_build:
            self.build_state_space()

    def load_aircraft(self) -> AircraftData:
        """Load aircraft _data from YAML file, validate, and return _data object."""

        with open(Path(__file__).parent / self.filename) as f:
            aircraft = yaml.safe_load(f)
        return AircraftData(**aircraft)

    def build_state_space(self) -> None:
        """Build state space model."""
        has_symmetric_data = self.data.symmetric is not None
        has_asymmetric_data = self.data.asymmetric is not None

        # Aircraft configuration
        configuration = self.configuration.lower()

        if configuration not in self.CONFIGURATIONS:
            raise ValueError(f"Invalid configuration: {configuration}")

        if configuration == "symmetric" and has_symmetric_data:
            self.ss = Symmetric(self.data)
        elif configuration == "sp" and has_symmetric_data:
            self.ss = SymmetricShortPeriod(self.data)
        elif configuration == "asymmetric" and has_asymmetric_data:
            self.ss = Asymmetric(self.data)
        else:
            raise ValueError(
                f"No data available for {self.configuration} configuration"
            )

        self.current_state = np.zeros((self.ss.nstates, 1))
        self.states = [self.current_state]

    def response(self, u: np.ndarray, x: np.ndarray = None) -> np.ndarray:
        """Return the response of the aircraft model to the input u."""
        u = np.array(u).reshape(self.ss.ninputs, 1)

        if x is None:
            x = self.current_state

        dx = self.ss.A @ x + self.ss.B @ u

        current_state = x + dx * self.dt
        self.current_state = current_state
        self.states.append(current_state)
        return current_state

    def set_initial_state(self, inital_state: np.ndarray) -> None:
        """Set initial state of the aircraft."""
        self.current_state = inital_state.reshape(self.ss.nstates, 1)


class StateSpace:
    a: np.ndarray
    b: np.ndarray
    x_names: list[str]
    u_names: list[str]

    def __init__(
        self, a: np.ndarray, b: np.ndarray, x_names: list[str], u_names: list[str]
    ) -> None:
        """Initialize state space model."""
        self.A = a
        self.B = b
        self.x_names = x_names
        self.u_names = u_names

        self.C = np.eye(a.shape[0])
        self.D = np.zeros((a.shape[0], b.shape[1]))

        self.nstates = a.shape[1]
        self.ninputs = b.shape[1]


class Symmetric(StateSpace):
    """Symmetric aircraft state space model."""

    def __init__(self, data: AircraftData) -> None:
        """Initialize symmetric aircraft model.

        args:
            data: AircraftData object
        """
        x_names, u_names = self.get_names()
        a = self.build_a(data)
        b = self.build_b(data)

        super().__init__(a, b, x_names, u_names)

    def get_names(self):
        """Get state and input names."""
        x_names = ["u_hat", "alpha", "theta", "q"]
        u_names = ["de"]
        return x_names, u_names

    @staticmethod
    def build_a(data: AircraftData) -> np.ndarray:
        """Build A matrix"""
        # Load variables
        v = data.v

        c_bar = data.symmetric.c_bar
        mu_c = data.symmetric.mu_c
        ky_2 = data.symmetric.ky_2

        cx = data.symmetric.cx
        cz = data.symmetric.cz
        cm = data.symmetric.cm

        # Declare repetitive calculations
        v_c = v / c_bar  # v/c_bar
        mu_c_cz_a = 2 * mu_c - cz.a_dot
        cm_a_mu_c = cm.a_dot / (2 * mu_c - cz.a_dot)
        mu_c_ky_2 = 2 * mu_c * ky_2

        xu = v_c * cx.u / (2 * mu_c)
        xa = v_c * cx.a / (2 * mu_c)
        xt = v_c * cz.o / (2 * mu_c)

        zu = v_c * cz.u / mu_c_cz_a
        za = v_c * cz.a / mu_c_cz_a
        zt = v_c * cx.o / mu_c_cz_a
        zq = v_c * (2 * mu_c + cz.q) / mu_c_cz_a

        mu = v_c * (cm.u + cz.u * cm_a_mu_c) / mu_c_ky_2
        ma = v_c * (cm.a + cz.a * cm_a_mu_c) / mu_c_cz_a
        mt = -v_c * (cx.o * cm_a_mu_c) / mu_c_ky_2
        mq = v_c * (cm.q + cm.a_dot * (2 * mu_c + cz.q) / mu_c_cz_a) / mu_c_cz_a

        a = np.array(
            [[xu, xa, xt, 0], [zu, za, zt, zq], [0, 0, 0, v_c], [mu, ma, mt, mq]]
        )

        # unormalize qc/v by doing multiplying the last row by v_c
        a[3, :] *= v_c
        a[:, 3] /= v_c

        return a

    @staticmethod
    def build_b(data: AircraftData) -> np.ndarray:
        """Build B matrix"""
        # Load variables
        v = data.v

        c_bar = data.symmetric.c_bar
        mu_c = data.symmetric.mu_c
        ky_2 = data.symmetric.ky_2

        cx = data.symmetric.cx
        cz = data.symmetric.cz
        cm = data.symmetric.cm

        # Declare repetitive calculations
        v_c = v / c_bar  # v/c_bar
        mu_c_cz_a = 2 * mu_c - cz.a_dot

        x_de = v_c * cx.de / (2 * mu_c)
        z_de = v_c * cz.de / mu_c_cz_a
        m_de = v_c * (cm.de + cz.de * cm.a_dot / mu_c_cz_a) / (2 * mu_c * ky_2)

        b = np.array([[x_de], [z_de], [0], [m_de]])

        # unormalize qcv
        b[3, :] *= v_c

        return b


class Asymmetric(StateSpace):
    """Asymmetric aircraft state space model."""

    def __init__(self, data: AircraftData) -> None:
        """Initialize asymmetric aircraft model.

        args:
            data: AircraftData object
        """
        x_names = ["beta", "phi", "pb_2v", "rb_2v"]
        u_names = ["da", "dr"]
        a = self.build_a(data)
        b = self.build_b(data)

        super().__init__(a, b, x_names, u_names)

    @staticmethod
    def build_a(data: AircraftData) -> np.ndarray:
        """Build A matrix"""
        # Load variables
        v = data.v

        b = data.asymmetric.b
        mu_b = data.asymmetric.mu_b
        c_l = data.asymmetric.c_l
        kx_2 = data.asymmetric.kx_2
        kz_2 = data.asymmetric.kz_2
        kxz = data.asymmetric.kxz

        cy = data.asymmetric.cy
        cl = data.asymmetric.cl
        cn = data.asymmetric.cn

        # Declare repetitive calculations
        v_b = v / b
        mu_b_k = 4 * mu_b * (kx_2 * kz_2 - kxz**2)

        yb = v_b * cy.b / (2 * mu_b)
        yphi = v_b * c_l / (2 * mu_b)
        yp = v_b * cy.p / (2 * mu_b)
        yr = v_b * (cy.r - 4 * mu_b) / (2 * mu_b)

        lb = v_b * (cl.b * kz_2 + cn.b * kxz) / mu_b_k
        lp = v_b = (cl.p * kz_2 + cn.p * kxz) / mu_b_k
        lr = v_b = (cl.r * kz_2 + cn.r * kxz) / mu_b_k

        nb = v_b * (cl.b * kxz + cn.b * kx_2) / mu_b_k
        n_p = v_b * (cl.p * kxz + cn.p * kx_2) / mu_b_k
        nr = v_b * (cl.r * kxz + cn.r * kx_2) / mu_b_k

        a = np.array(
            [[yb, yphi, yp, yr], [0, 0, 2 * v_b, 0], [lb, 0, lp, lr], [nb, 0, n_p, nr]]
        )

        return a

    @staticmethod
    def build_b(data: AircraftData) -> np.ndarray:
        """Build B matrix."""
        # Load variables
        v = data.v

        mu_b = data.asymmetric.mu_b
        kx_2 = data.asymmetric.kx_2
        kz_2 = data.asymmetric.kz_2
        kxz = data.asymmetric.kxz

        cy = data.asymmetric.cy
        cl = data.asymmetric.cl
        cn = data.asymmetric.cn

        # Declare repetitive calculations
        v_b = v / mu_b
        mu_b_k = 4 * mu_b * (kx_2 * kz_2 - kxz**2)

        y_dr = v_b * cy.dr / (2 * mu_b)

        l_da = v_b * (cl.da * kz_2 + cn.da * kxz) / mu_b_k
        l_dr = v_b * (cl.dr * kz_2 + cn.dr * kxz) / mu_b_k

        n_da = v_b * (cl.da * kxz + cn.da * kx_2) / mu_b_k
        n_dr = v_b * (cl.dr * kxz + cn.dr * kx_2) / mu_b_k

        b = np.array([[0, y_dr], [0, 0], [l_da, l_dr], [n_da, n_dr]])

        return b


class SymmetricShortPeriod(Symmetric):
    """Creates the short period symmetric aircraft."""

    @staticmethod
    def mask_a(a: np.ndarray) -> np.ndarray:
        """mask matrix A for short period."""
        masked_a = np.zeros((2, 2))
        masked_a[0, 0] = a[1, 1]  # za
        masked_a[0, 1] = a[1, 3]  # zq
        masked_a[1, 0] = a[3, 1]  # ma
        masked_a[1, 1] = a[3, 3]  # mq
        return masked_a

    @staticmethod
    def mask_b(b: np.ndarray) -> np.ndarray:
        """mask matrix B for short period."""
        masked_b = np.zeros((2, 1))
        masked_b[0, 0] = b[1, 0]  # zde
        masked_b[1, 0] = b[3, 0]  # mde
        return masked_b

    def get_names(self):
        """Get names of states and inputs."""
        x_names = ["alpha", "q"]
        u_names = ["de"]
        return x_names, u_names

    def build_a(self, data: AircraftData) -> np.ndarray:
        """Builds A matrix for short period"""
        a = super().build_a(data)
        return self.mask_a(a)

    def build_b(self, data: AircraftData) -> np.ndarray:
        """Builds B matrix for short period"""
        b = super().build_b(data)
        return self.mask_b(b)
