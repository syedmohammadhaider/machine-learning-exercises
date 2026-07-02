import json
import urllib.request
import urllib.error
from typing import Optional


class LumenError(Exception):
    pass


class Lumen:
    """Python client for the Lumen experiment tracker API.

    Usage:
        lm = Lumen("http://localhost:8000")
        run = lm.create_run(name="my-run", tags=["vision"], config={"lr": 0.001})
        lm.log_metric(run["id"], "loss", 0.5, step=100)
        lm.log_params(run["id"], {"optimizer": "adam", "dropout": 0.2})
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict | list:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req) as resp:
                raw = resp.read().decode()
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            detail = e.read().decode()
            try:
                msg = json.loads(detail).get("detail", detail)
            except Exception:
                msg = detail
            raise LumenError(f"{e.code}: {msg}") from None

    def create_run(
        self,
        name: str = "",
        tags: Optional[list[str]] = None,
        config: Optional[dict] = None,
    ) -> dict:
        """Create a new run. Returns the created run."""
        return self._request("POST", "/runs", {
            "name": name,
            "tags": tags or [],
            "config": config or {},
        })

    def log_metric(self, run_id: int, key: str, value: float, step: int = 0) -> dict:
        """Log a metric for a run."""
        return self._request("POST", f"/runs/{run_id}/metrics", {
            "key": key,
            "value": value,
            "step": step,
        })

    def log_params(self, run_id: int, params: dict) -> dict:
        """Log hyperparameters for a run."""
        return self._request("POST", f"/runs/{run_id}/params", {"params": params})

    def list_runs(self) -> list:
        """List all runs with latest metrics."""
        return self._request("GET", "/runs")

    def get_run(self, run_id: int) -> dict:
        """Get full details for a single run."""
        return self._request("GET", f"/runs/{run_id}")

    def update_run(
        self,
        run_id: int,
        name: Optional[str] = None,
        tags: Optional[list[str]] = None,
        notes: Optional[str] = None,
        is_best: Optional[bool] = None,
        config: Optional[dict] = None,
    ) -> dict:
        """Update a run's metadata. Only provided fields are changed."""
        body = {}
        if name is not None:
            body["name"] = name
        if tags is not None:
            body["tags"] = tags
        if notes is not None:
            body["notes"] = notes
        if is_best is not None:
            body["is_best"] = is_best
        if config is not None:
            body["config"] = config
        return self._request("PATCH", f"/runs/{run_id}", body)

    def delete_run(self, run_id: int) -> dict:
        """Delete a run and all its metrics/params."""
        return self._request("DELETE", f"/runs/{run_id}")

    def compare_runs(self, *ids: int) -> list:
        """Compare two or more runs side-by-side."""
        ids_str = ",".join(str(i) for i in ids)
        return self._request("GET", f"/runs/compare?ids={ids_str}")
