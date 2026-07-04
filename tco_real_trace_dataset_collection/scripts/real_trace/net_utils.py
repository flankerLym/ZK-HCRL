#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Network helpers for real-trace collectors.

By default these collectors set ``requests.Session.trust_env=False`` so Python
will not silently read Windows system proxy settings from Internet Options.
This is important when a VPN runs in TUN/global mode: traffic is already routed
at the OS/network layer, while stale Windows proxy settings can make requests
fail with ``ProxyError: Unable to connect to proxy``.

Use ``--trust-env`` only when you intentionally want Python to use
HTTP_PROXY/HTTPS_PROXY/ALL_PROXY or Windows system proxy settings.
"""

from __future__ import annotations

import requests

USER_AGENT = "TCO-DRL-real-trace-collector/1.1"


def make_requests_session(trust_env: bool = False, timeout_note: str | None = None) -> requests.Session:
    """Return a requests session with explicit proxy behavior.

    Parameters
    ----------
    trust_env:
        If False, requests ignores HTTP_PROXY/HTTPS_PROXY/ALL_PROXY and Windows
        system proxy settings. If True, requests uses the normal environment.
    """
    session = requests.Session()
    session.trust_env = bool(trust_env)
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def make_web3_http_provider(rpc_url: str, timeout: int = 60, trust_env: bool = False):
    """Create a Web3 HTTPProvider with a requests session.

    Newer Web3.py versions accept ``session=...``. If an older version does not,
    the fallback still creates a provider, but users should upgrade Web3 if they
    need strict proxy isolation for RPC calls.
    """
    from web3 import Web3

    session = make_requests_session(trust_env=trust_env)
    try:
        return Web3.HTTPProvider(
            rpc_url,
            request_kwargs={"timeout": timeout},
            session=session,
        )
    except TypeError:
        provider = Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": timeout})
        # Best-effort compatibility with older Web3.py internals.
        for attr in ("_session", "session"):
            try:
                setattr(provider, attr, session)
            except Exception:
                pass
        return provider
