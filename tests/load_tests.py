"""Locust load test hitting the FastAPI endpoints."""

from locust import HttpUser, task, between


class SimulatorUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(3)
    def fetch_status(self):
        self.client.get("/status")

    @task(1)
    def fetch_snapshot(self):
        self.client.get("/snapshot?limit=50")
