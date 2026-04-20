from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class LeaseToken:
    token_id: str
    resource_name: str
    owner: str
    priority: int


class DeviceLeaseManager:
    def __init__(self):
        self._condition = threading.Condition()
        self._owners: dict[str, LeaseToken] = {}

    def acquire(self, resource_name: str, owner: str, priority: int, timeout: float | None = None) -> LeaseToken:
        with self._condition:
            while resource_name in self._owners:
                self._condition.wait(timeout=timeout)
            token = LeaseToken(
                token_id=str(uuid.uuid4()),
                resource_name=str(resource_name),
                owner=str(owner),
                priority=int(priority),
            )
            self._owners[token.resource_name] = token
            return token

    def release(self, token: LeaseToken) -> None:
        with self._condition:
            current = self._owners.get(token.resource_name)
            if current and current.token_id == token.token_id:
                self._owners.pop(token.resource_name, None)
                self._condition.notify_all()

    def current_owner(self, resource_name: str) -> str | None:
        with self._condition:
            token = self._owners.get(str(resource_name))
            return token.owner if token else None

    def current_token(self, resource_name: str) -> LeaseToken | None:
        with self._condition:
            return self._owners.get(str(resource_name))
