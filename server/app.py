"""FastAPI application for the OnCallEnv environment."""

from openenv.core.env_server.http_server import create_app

from oncall_env.models import OnCallAction, OnCallObservation
from oncall_env.server.environment import OnCallEnvironment

app = create_app(
    env=OnCallEnvironment,
    action_cls=OnCallAction,
    observation_cls=OnCallObservation,
    env_name="oncall_env",
)


def main():
    import uvicorn
    uvicorn.run(
        "oncall_env.server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
