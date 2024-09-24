import enum

class AgentType(enum.StrEnum):
    PPO = "PPO"
    SAC = "SAC"
    ON_POLICY = "on_policy"
    OFF_POLICY = "off_policy" 
