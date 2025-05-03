#Jaider Alfonso Perez Gutierrez
#codigo: 30000079946
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np

# Cargar los modelos desde la carpeta 'model'
modelo = joblib.load("model/modelo_final.pkl")
scaler = joblib.load("model/scaler.pkl")
encoder = joblib.load("model/label_encoder.pkl")

# Crear la app
app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
def predict(
    request: Request,
    price_usd: float = Form(...),
    battery_mah: float = Form(...),
    ram_gb: float = Form(...),
    storage_gb: float = Form(...),
    camera_mp: float = Form(...),
    screen_size_in: float = Form(...),
    weight_g: float = Form(...)
):
    datos = np.array([[price_usd, battery_mah, ram_gb, storage_gb, camera_mp, screen_size_in, weight_g]])
    datos_escalados = scaler.transform(datos)
    pred = modelo.predict(datos_escalados)[0]
    pred_label = encoder.inverse_transform([pred])[0]
    return templates.TemplateResponse("index.html", {"request": request, "resultado": pred_label})
