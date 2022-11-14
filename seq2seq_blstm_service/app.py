from fastapi import FastAPI
from pydantic import BaseModel
from inference_model import Inference_model
import uvicorn


app = FastAPI()
model = Inference_model('seq2seq_blstm_hcm')

class Request_Item(BaseModel):
    text: str

class Response_Item(BaseModel):
    text: str
    execution_time: float


@app.post("/api/v1/seq2seq_blstm")
def answer(Request: Request_Item):
    question = Request.text

    if question is None:
        return {'error': 'Please input a question'}
    
    answer, execution_time = model.generate_answer(question)

    return Response_Item(text = answer, execution_time= round(execution_time, 4))

if __name__ == '__main__':
    uvicorn.run("app:app",host='0.0.0.0', port=8080)
