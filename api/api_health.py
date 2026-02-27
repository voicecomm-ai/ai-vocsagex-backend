from fastapi.responses import JSONResponse

from api.api import app

@app.get('/health')
async def health_check():
    if app.state.unhealth_event.is_set():
        return JSONResponse(content={"status": "error"}, status_code=500)
    else:
        return JSONResponse(content={"status": "ok"})
