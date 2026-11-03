from __future__ import annotations

from atas_geometry.service.app import create_app
from atas_geometry.service.config import Settings
def main() -> None:
    import uvicorn

    settings = Settings.from_env()
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
