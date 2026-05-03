"""
Global configuration for the Oncolytica framework.
"""


class Settings:
    # Whether to print GPU details on initialization
    SHOW_GPU_INFO: bool = False

    # Save wgsl shader to this path
    SAVE_WGSL_PATH = "shader.wgsl"

# Single instance to be used across the library
settings = Settings()