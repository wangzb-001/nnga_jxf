import uvicorn
from config import Config
from fastapi import FastAPI
from logger import init_log

from server import simple_router, show_router, search_router, manage_router

config = Config.config

app = FastAPI(title="es_server")
init_log(log_dir="log_dir", file_name="es_server", console_output=True)


app.include_router(simple_router)
app.include_router(show_router)
app.include_router(search_router)
app.include_router(manage_router)


def main():
    uvicorn.run(app=app, host=config['host'], port=config['port'], debug=config["debug"])


if __name__ == "__main__":
    main()
