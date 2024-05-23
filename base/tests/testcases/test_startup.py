import pytest
from backend.app import App



@pytest.fixture()
def app():
    app = App()
    app.config.update({
        'TESTING':True,
    })

    yield app
    #clean up here


@pytest.fixture()
def client(app):
    return app.test_client()


def test_request_index(client):
    response = client.get("/")
    assert '200' in response.status
    assert b"<html>" in response.data
    #assert 0


def test_request_settings(client, app):
    #TODO: delete/move settings.json file before request
    #      to make sure the response is still valid
    #import shutil, os

    response = client.get("/settings")
    assert '200' in response.status

    import json
    data     = json.loads(response.data)
    assert 'settings'          in data
    assert 'active_models'     in data['settings']
    assert 'available_models'  in data


