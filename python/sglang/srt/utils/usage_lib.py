# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Adapted from vLLM: https://github.com/vllm-project/vllm

"""Usage statistics collection for SGLang."""

import datetime
import json
import logging
import os
import platform
import time
from enum import Enum
from pathlib import Path
from threading import Thread
from typing import Any
from uuid import uuid4

import cpuinfo
import psutil
import requests
import torch

from sglang.version import __version__ as SGLANG_VERSION

logger = logging.getLogger(__name__)

# Configuration paths
_config_home = os.path.join(
    os.getenv("XDG_CONFIG_HOME", os.path.expanduser("~/.config")), "sglang"
)
_USAGE_STATS_JSON_PATH = os.path.join(_config_home, "usage_stats.json")
_USAGE_STATS_DO_NOT_TRACK_PATH = os.path.join(_config_home, "do_not_track")
_USAGE_STATS_ENABLED = None
_USAGE_STATS_SERVER = os.getenv(
    "SGLANG_USAGE_STATS_SERVER", "https://stats.sglang.ai/api/v1/usage"
)

_GLOBAL_RUNTIME_DATA = dict[str, str | int | bool]()

_USAGE_ENV_VARS_TO_COLLECT = [
    "SGLANG_USE_MODELSCOPE",
    "SGLANG_ENABLE_TORCH_COMPILE",
    "SGLANG_IS_FLASHINFER_AVAILABLE",
]


def set_runtime_usage_data(key: str, value: str | int | bool) -> None:
    """Set global usage data that will be sent with every usage heartbeat."""
    _GLOBAL_RUNTIME_DATA[key] = value


def is_usage_stats_enabled():
    """Determine whether or not we can send usage stats to the server.
    The logic is as follows:
    - By default, it should be enabled.
    - Three environment variables can disable it:
        - SGLANG_DO_NOT_TRACK=1
        - DO_NOT_TRACK=1
        - SGLANG_NO_USAGE_STATS=1
    - A file in the home directory can disable it if it exists:
        - $HOME/.config/sglang/do_not_track
    """
    global _USAGE_STATS_ENABLED
    if _USAGE_STATS_ENABLED is None:
        do_not_track = os.getenv("SGLANG_DO_NOT_TRACK", "0").lower() in (
            "1",
            "true",
            "yes",
        ) or os.getenv("DO_NOT_TRACK", "0").lower() in ("1", "true", "yes")
        no_usage_stats = os.getenv("SGLANG_NO_USAGE_STATS", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        do_not_track_file = os.path.exists(_USAGE_STATS_DO_NOT_TRACK_PATH)

        _USAGE_STATS_ENABLED = not (do_not_track or no_usage_stats or do_not_track_file)
    return _USAGE_STATS_ENABLED


def _get_current_timestamp_ns() -> int:
    return int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1e9)


def _detect_cloud_provider() -> str:
    # Try detecting through vendor file
    vendor_files = [
        "/sys/class/dmi/id/product_version",
        "/sys/class/dmi/id/bios_vendor",
        "/sys/class/dmi/id/product_name",
        "/sys/class/dmi/id/chassis_asset_tag",
        "/sys/class/dmi/id/sys_vendor",
    ]
    # Mapping of identifiable strings to cloud providers
    cloud_identifiers = {
        "amazon": "AWS",
        "microsoft corporation": "AZURE",
        "google": "GCP",
        "oraclecloud": "OCI",
    }

    for vendor_file in vendor_files:
        path = Path(vendor_file)
        if path.is_file():
            file_content = path.read_text().lower()
            for identifier, provider in cloud_identifiers.items():
                if identifier in file_content:
                    return provider

    # Try detecting through environment variables
    env_to_cloud_provider = {
        "RUNPOD_DC_ID": "RUNPOD",
    }
    for env_var, provider in env_to_cloud_provider.items():
        if os.environ.get(env_var):
            return provider

    return "UNKNOWN"


class UsageContext(str, Enum):
    UNKNOWN_CONTEXT = "UNKNOWN_CONTEXT"
    LLM_CLASS = "LLM_CLASS"
    API_SERVER = "API_SERVER"
    OPENAI_API_SERVER = "OPENAI_API_SERVER"
    ENGINE_CONTEXT = "ENGINE_CONTEXT"


class UsageMessage:
    """Collect platform information and send it to the usage stats server."""

    def __init__(self) -> None:
        self.uuid = str(uuid4())

        # Environment Information
        self.provider: str | None = None
        self.num_cpu: int | None = None
        self.cpu_type: str | None = None
        self.cpu_family_model_stepping: str | None = None
        self.total_memory: int | None = None
        self.architecture: str | None = None
        self.platform: str | None = None
        self.cuda_runtime: str | None = None
        self.gpu_count: int | None = None
        self.gpu_type: str | None = None
        self.gpu_memory_per_device: int | None = None
        self.env_var_json: str | None = None

        # SGLang Information
        self.model_architecture: str | None = None
        self.sglang_version: str | None = None
        self.context: str | None = None

        # Metadata
        self.log_time: int | None = None
        self.source: str | None = None

    def report_usage(
        self,
        model_architecture: str,
        usage_context: UsageContext,
        extra_kvs: dict[str, Any] | None = None,
    ) -> None:
        t = Thread(
            target=self._report_usage_worker,
            args=(model_architecture, usage_context, extra_kvs or {}),
            daemon=True,
        )
        t.start()

    def _report_usage_worker(
        self,
        model_architecture: str,
        usage_context: UsageContext,
        extra_kvs: dict[str, Any],
    ) -> None:
        self._report_usage_once(model_architecture, usage_context, extra_kvs)
        self._report_continuous_usage()

    def _report_usage_once(
        self,
        model_architecture: str,
        usage_context: UsageContext,
        extra_kvs: dict[str, Any],
    ) -> None:
        # Platform information
        if torch.cuda.is_available():
            self.gpu_count = torch.cuda.device_count()
            if self.gpu_count > 0:
                props = torch.cuda.get_device_properties(0)
                self.gpu_type = props.name
                self.gpu_memory_per_device = props.total_memory
            self.cuda_runtime = torch.version.cuda

        self.provider = _detect_cloud_provider()
        self.architecture = platform.machine()
        self.platform = platform.platform()
        self.total_memory = psutil.virtual_memory().total

        info = cpuinfo.get_cpu_info()
        self.num_cpu = info.get("count", None)
        self.cpu_type = info.get("brand_raw", "")
        self.cpu_family_model_stepping = ",".join(
            [
                str(info.get("family", "")),
                str(info.get("model", "")),
                str(info.get("stepping", "")),
            ]
        )

        # SGLang information
        self.context = usage_context.value
        self.sglang_version = SGLANG_VERSION
        self.model_architecture = model_architecture

        # Environment variables
        self.env_var_json = json.dumps(
            {env_var: os.getenv(env_var) for env_var in _USAGE_ENV_VARS_TO_COLLECT}
        )

        # Metadata
        self.log_time = _get_current_timestamp_ns()
        self.source = os.getenv("SGLANG_USAGE_SOURCE", "production")

        data = vars(self)
        if extra_kvs:
            data.update(extra_kvs)

        self._write_to_file(data)
        self._send_to_server(data)

    def _report_continuous_usage(self):
        """Report usage every 10 minutes.

        This helps us to collect more data points for uptime of SGLang usages.
        This function can also help send over performance metrics over time.
        """
        while True:
            time.sleep(600)
            data = {
                "uuid": self.uuid,
                "log_time": _get_current_timestamp_ns(),
            }
            data.update(_GLOBAL_RUNTIME_DATA)

            self._write_to_file(data)
            self._send_to_server(data)

    def _send_to_server(self, data: dict[str, Any]) -> None:
        if not is_usage_stats_enabled():
            return
        try:
            requests.post(_USAGE_STATS_SERVER, json=data, timeout=5)
        except requests.exceptions.RequestException:
            # silently ignore unless we are using debug log
            logging.debug("Failed to send usage data to server")

    def _write_to_file(self, data: dict[str, Any]) -> None:
        if not is_usage_stats_enabled():
            return
        try:
            os.makedirs(os.path.dirname(_USAGE_STATS_JSON_PATH), exist_ok=True)
            Path(_USAGE_STATS_JSON_PATH).touch(exist_ok=True)
            with open(_USAGE_STATS_JSON_PATH, "a") as f:
                json.dump(data, f)
                f.write("\n")
        except OSError:
            logging.debug("Failed to write usage data to file")


usage_message = UsageMessage()


def report_usage_stats(
    server_args,
    model_architecture: str,
    usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
) -> None:
    """Report usage statistics if enabled.

    Args:
        server_args: SGLang ServerArgs instance containing server configuration.
        model_architecture: The model architecture name (e.g., "LlamaForCausalLM").
        usage_context: The context in which SGLang is being used.
    """
    if not is_usage_stats_enabled():
        return

    usage_message.report_usage(
        model_architecture,
        usage_context,
        extra_kvs={
            # Common configuration
            "dtype": str(server_args.dtype),
            "mem_fraction_static": server_args.mem_fraction_static,
            "context_length": server_args.context_length,
            # Quantization
            "quantization": server_args.quantization,
            "kv_cache_dtype": str(server_args.kv_cache_dtype),
            # Feature flags
            "enable_lora": bool(server_args.enable_lora),
            "is_embedding": server_args.is_embedding,
            "enable_multimodal": server_args.enable_multimodal,
            # Distributed parallelism settings
            "tensor_parallel_size": server_args.tp_size,
            "data_parallel_size": server_args.dp_size,
            "pipeline_parallel_size": server_args.pp_size,
            # Kernel backends
            "attention_backend": server_args.attention_backend,
            "sampling_backend": server_args.sampling_backend,
            # Speculative decoding
            "speculative_algorithm": server_args.speculative_algorithm,
        },
    )