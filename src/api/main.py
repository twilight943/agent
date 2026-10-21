from __future__ import annotations

import argparse
import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from src.agent.graph import RecommendationAgent
from src.api.routes import router
from src.common.config import data_path, load_config
from src.common.logger import configure_logging
from src.services.item_service import ItemService
from src.services.llm_client import LLMClient
from src.services.recommendation_service import RecommendationService


class ServiceContainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.details: list[str] = []
        self.item_service: ItemService | None = None
        self.recommendation_service: RecommendationService | None = None
        self.agent: RecommendationAgent | None = None
        self.llm = LLMClient(config.get("llm", {}), config.get("agent", {}).get("temperature", 0.1))
        processed = data_path(config, config["data"].get("processed_dir", "processed"))
        items_path = processed / "items.parquet"
        if items_path.exists():
            try:
                self.item_service = ItemService.from_parquet(items_path)
            except Exception as error:
                self.details.append(f"item metadata: {error}")
        else:
            self.details.append(f"item metadata missing: {items_path}")
        try:
            self.recommendation_service = RecommendationService.from_config(config)
            self.item_service = self.recommendation_service.item_service
            self.agent = RecommendationAgent(
                self.recommendation_service, self.llm, dict(config.get("agent", {}))
            )
        except Exception as error:
            self.details.append(f"recommendation resources: {error}")

    def health(self) -> dict[str, Any]:
        resources = {
            "items": self.item_service is not None,
            "recommendation": self.recommendation_service is not None,
            "agent": self.agent is not None,
            "llm": self.llm.configured,
        }
        required_ready = resources["items"] and resources["recommendation"] and resources["agent"]
        return {
            "status": "ok" if required_ready else "degraded",
            "environment": str(self.config.get("environment", "unknown")),
            "resources": resources,
            "details": self.details,
        }


def create_app(
    config: dict[str, Any] | None = None,
    container: ServiceContainer | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if not hasattr(application.state, "container"):
            application.state.container = ServiceContainer(config or load_config())
        yield

    application = FastAPI(
        title="AgentRec API",
        version="0.1.0",
        description="Hybrid recommendation system with a constrained LLM Agent",
        lifespan=lifespan,
    )
    if container is not None:
        application.state.container = container
    application.include_router(router)
    return application


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentRec API utilities")
    parser.add_argument("--env", choices=["local", "server"], default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--check", action="store_true", help="Validate resource configuration and exit")
    args = parser.parse_args()
    configure_logging()
    config = load_config(args.env, args.config)
    if args.check:
        print(json.dumps(ServiceContainer(config).health(), ensure_ascii=False, indent=2))
        return
    import uvicorn

    uvicorn.run(
        create_app(config),
        host=str(config.get("api", {}).get("host", "0.0.0.0")),
        port=int(config.get("api", {}).get("port", 8080)),
    )


if __name__ == "__main__":
    main()
