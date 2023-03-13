from agents.config import ConfigIDHPSAC

class TestConfigIDHPSAC:
    """Test the IDHP-SAC interface with its config."""
    def test_config_idhp_sac(self):
        config = ConfigIDHPSAC()
        assert config.name == "IDHP-SAC"
