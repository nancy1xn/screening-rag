from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from screening_rag.cnn_pipeline.crime import generate_crime_events_report
from screening_rag.cnn_pipeline.news import generate_background_report

app = FastAPI()


@app.get("/click-button/{entity_name}")
def click_button(entity_name: str):
    background, appendix1 = generate_background_report(entity_name)
    content, appendix = generate_crime_events_report(entity_name)
    print(content)
    return {
        "background": background,
        "appendix1": appendix1,
        "content": content,
        "appendix": appendix,
    }


app.mount(
    "/static",
    StaticFiles(packages=[("screening_rag", "static")], html=True),
    name="static",
)

app.mount(
    "/",
    StaticFiles(packages=[("screening_rag", "static")], html=True),
    name="root_static",
)
