from fastapi import FastAPI, HTTPException
from model.model import get_model
from LSL_listener import LSLStreamListener
from contextlib import asynccontextmanager
from pydantic import BaseModel
import time
from model.data_ingestion import DataCruncher


class StreamNameUpdate(BaseModel):
    name: str


initialised_model = get_model('model/estimator.joblib')
channel = open('model/channel.txt', 'r').read().replace('\n', '')
stream_name = open('model/stream_name.txt', 'r').read().replace('\n', '')


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start LSL listener when app starts
    lsl_listener.start()
    yield
    # Cleanup when app stops
    lsl_listener.stop()


lsl_listener = LSLStreamListener(
    buffer_duration_seconds=12,
    selected_channels=(channel,),  # This depends on the model you trained. Ordered.
    stream_name=stream_name,  # Match your LSL stream name
    data_cruncher=DataCruncher(n_processes=1),  # Ideally one per channel, if using multiple channels
)


def predict():
    return initialised_model(lsl_listener.get_buffer())


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {'Yo'}


@app.put("/set_stream_name")
async def set_stream_name(update: StreamNameUpdate):
    """
    Update the LSL stream name.
    """
    try:
        lsl_listener.set_stream_name(update.name)
        return {"message": f"Stream name set to: {update.name}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/get_prediction")
async def get_prediction():
    prediction = predict()
    if prediction is None:
        raise HTTPException(status_code=425, detail="Too Early: LSL still buffering")
    return prediction


@app.get("/get_prediction_time")
async def get_prediction_time():
    start = time.time()
    _ = predict()
    return time.time() - start
