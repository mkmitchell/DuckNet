"""HTTP-level behaviour of the Flask app: path handling on file routes, the
loopback-only shutdown route, and the server-sent-events stream releasing its
worker thread when idle.
"""

import os
import time

import pytest

from base.backend import pubsub


@pytest.fixture(scope="module")
def app():
    from backend.app import App

    application = App()
    application.config.update({"TESTING": True})
    return application


@pytest.fixture
def client(app):
    return app.test_client()


def test_delete_image_does_not_escape_cache(app, client, tmp_path):
    outside = tmp_path / "victim.txt"
    outside.write_text("keep me")
    relative = os.path.relpath(outside, app.cache_path).replace(os.sep, "/")
    response = client.get(f"/delete_image/{relative}")
    assert response.status_code == 400
    assert outside.exists()


def test_delete_image_removes_only_basename_in_cache(app, client):
    target = os.path.join(app.cache_path, "gone.jpg")
    open(target, "wb").close()
    assert client.get("/delete_image/gone.jpg").status_code == 200
    assert not os.path.exists(target)


def test_process_image_rejects_traversal(client):
    assert client.get("/process_image/..%2Fsettings.json").status_code in (
        400,
        404,
    )


def test_save_model_rejects_traversal(client):
    response = client.get("/save_model", query_string={"newname": "../evil"})
    assert response.status_code == 400


def test_http_errors_are_json(client):
    response = client.get("/save_model", query_string={"newname": "../evil"})
    assert response.status_code == 400
    assert response.get_json()["error"] == 400
    assert "plain filename" in response.get_json()["description"]


def test_training_status_idle(client):
    status = client.get("/training_status").get_json()
    assert status == {"running": False, "last": None}


def test_shutdown_refused_from_non_loopback(client):
    response = client.get(
        "/shutdown", environ_base={"REMOTE_ADDR": "10.0.0.7"}
    )
    assert response.status_code == 403


def test_stream_emits_keepalive_and_unsubscribes_when_client_leaves(client):
    before = len(pubsub.PubSub.subscribers)
    response = client.get("/stream")
    iterator = response.iter_encoded()
    first = next(iterator)
    assert first.startswith(b":")
    assert len(pubsub.PubSub.subscribers) == before + 1
    response.close()
    deadline = time.time() + 5
    while len(pubsub.PubSub.subscribers) != before and time.time() < deadline:
        time.sleep(0.05)
    assert len(pubsub.PubSub.subscribers) == before


def test_pubsub_unsubscribe_is_idempotent():
    queue = pubsub.PubSub.subscribe()
    pubsub.PubSub.unsubscribe(queue)
    pubsub.PubSub.unsubscribe(queue)
    assert queue not in pubsub.PubSub.subscribers
