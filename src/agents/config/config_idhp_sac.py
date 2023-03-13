from typing import Optional, Literal
from pydantic import BaseModel, Extra
from helpers.config_auto import get_auto

class ConfigIDHPSACLearn(BaseModel):
    """Allows defining parameters that can be passed to learn method."""
    total_timesteps: Optional[int] = 1_000
    callback: Optional[list] = ["tensorboard"]
    log_interval: Optional[int] = 1
    tb_log_name: Optional[str] = get_auto("tb_log_name")
    reset_num_timesteps: Optional[bool] = True
    progress_bar: Optional[bool] = False


class ConfigIDHPSAC(BaseModel):
    """Configuration of SAC."""
    name: Literal['IDHP-SAC'] = "IDHP-SAC"
    learn: Optional[ConfigIDHPSACLearn] = ConfigIDHPSACLearn()

    class Config:
        extra = Extra.forbid
