import pytest
from unittest.mock import patch


def test_make_engine_returns_engine():
    from backend.db import make_engine
    engine = make_engine("sqlite:///:memory:")
    assert engine is not None
    engine.dispose()


def test_make_session_local_returns_factory():
    from backend.db import make_engine, make_session_local
    engine = make_engine("sqlite:///:memory:")
    factory = make_session_local(engine)
    assert factory is not None
    engine.dispose()


def test_namespaced_cache_prefixes_keys():
    from backend.db import NamespacedKey
    ns = NamespacedKey("fantasy")
    assert ns.key("ros:2026-04-04") == "fantasy:ros:2026-04-04"
    assert ns.key("token") == "fantasy:token"


def test_namespaced_cache_different_prefixes_do_not_collide():
    from backend.db import NamespacedKey
    edge = NamespacedKey("edge")
    fantasy = NamespacedKey("fantasy")
    assert edge.key("token") != fantasy.key("token")
