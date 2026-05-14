from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from temporalio.client import Client
from temporalio.service import TLSConfig

DEFAULT_HOST_PORT = "localhost:7233"
DEFAULT_NAMESPACE = "default"


@dataclass(frozen=True)
class TemporalConnectionConfig:
    host_port: str = DEFAULT_HOST_PORT
    namespace: str = DEFAULT_NAMESPACE
    tls: TLSConfig | None = None
    data_converter: Any | None = None
    interceptors: list[Any] = field(default_factory=list)
    identity: str | None = None
    runtime: Any | None = None

    def connect_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"namespace": self.namespace}
        if self.tls is not None:
            kwargs["tls"] = self.tls
        if self.data_converter is not None:
            kwargs["data_converter"] = self.data_converter
        if self.interceptors:
            kwargs["interceptors"] = list(self.interceptors)
        if self.identity is not None:
            kwargs["identity"] = self.identity
        if self.runtime is not None:
            kwargs["runtime"] = self.runtime
        return kwargs


def _read_file_if_present(path: str) -> bytes | None:
    if not path:
        return None
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except OSError:
        return None


def _load_tls_config_from_env() -> TLSConfig | None:
    cert_path = os.getenv("TEMPORAL_TLS_CERT", "")
    key_path = os.getenv("TEMPORAL_TLS_KEY", "")
    server_root_ca_path = os.getenv("TEMPORAL_TLS_CA", "") or os.getenv("TEMPORAL_TLS_SERVER_ROOT_CA_CERT", "")
    domain = os.getenv("TEMPORAL_TLS_SERVER_NAME", "") or None
    cert_data = _read_file_if_present(cert_path)
    key_data = _read_file_if_present(key_path)
    server_root_ca_data = _read_file_if_present(server_root_ca_path)
    if not any([cert_data, key_data, server_root_ca_data, domain]):
        return None
    return TLSConfig(
        client_cert=cert_data,
        client_private_key=key_data,
        server_root_ca_cert=server_root_ca_data,
        domain=domain,
    )


def load_client_config(host_port_override: str = "", namespace_override: str = "") -> TemporalConnectionConfig:
    host_port = host_port_override or os.getenv("TEMPORAL_HOST_URL") or DEFAULT_HOST_PORT
    namespace = namespace_override or os.getenv("TEMPORAL_NAMESPACE") or DEFAULT_NAMESPACE
    return TemporalConnectionConfig(
        host_port=host_port,
        namespace=namespace,
        tls=_load_tls_config_from_env(),
    )


def client_config_from_overrides(
    *,
    host_port: str = "",
    namespace: str = "",
    tls: TLSConfig | None = None,
    data_converter: Any | None = None,
    interceptors: list[Any] | None = None,
    identity: str | None = None,
    runtime: Any | None = None,
) -> TemporalConnectionConfig:
    base = load_client_config(host_port, namespace)
    return TemporalConnectionConfig(
        host_port=base.host_port,
        namespace=base.namespace,
        tls=tls if tls is not None else base.tls,
        data_converter=data_converter,
        interceptors=list(interceptors or []),
        identity=identity,
        runtime=runtime,
    )


async def connect_client(
    host_port_override: str = "",
    namespace_override: str = "",
    *,
    config: TemporalConnectionConfig | None = None,
) -> Client:
    resolved = config or load_client_config(host_port_override, namespace_override)
    return await Client.connect(resolved.host_port, **resolved.connect_kwargs())
