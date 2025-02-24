import uvicorn

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from semantic_kernel import Kernel
from semantic_kernel.contents.chat_history import ChatHistory
from src.ragPlugin import RAGPlugin

from src.settings import setup, load_settings_from_yaml

from langchain.embeddings import OpenAIEmbeddings

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use this only for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = load_settings_from_yaml("settings.yaml")
llm, text_embedder, chat_completion, execution_settings = setup(settings)
kernel = Kernel()

# Prepare OpenAI service using credentials stored in the `.env` file
kernel.add_service(chat_completion)

# Register the Math and String plugins
rag_plugin = RAGPlugin(settings, text_embedder, llm)

kernel.add_plugin(
    rag_plugin,
    plugin_name="RAG",
)

# Create a history of the conversation
history = ChatHistory()

# Load system prompt from file
with open("system_prompt.txt", "r") as f:
    system_prompt = f.read().strip()
history.add_message({"role": "system", "content": system_prompt})

embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)

@app.get("/ask")
async def local_search(query: str = Query(..., description="Ask anything")):
    global history

    # query = "The user lives in Cobham and uses M25J10 for commute to Imperial College London. Ask the following question:\n" + query
    try:
        history.add_message({"role": "user", "content": query})

        # Get the response from the AI
        response = await chat_completion.get_chat_message_content(
            chat_history=history,
            settings=execution_settings,
            kernel=kernel,
        )

        # Add the message from the agent to the chat history
        history.add_message(response)

        raw_result = None
        if rag_plugin.raw_RAG_result:
            raw_result = rag_plugin.raw_RAG_result

        response_dict = {
            "llm_reply": response.content,
            "plugin_called": "RAG",
            "raw_json": raw_result
        }

        rag_plugin.raw_RAG_result = None
        return JSONResponse(content=response_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/refresh_history")
async def local_search():
    global history
    try:
        history = ChatHistory()
        rag_plugin.raw_RAG_result = None
        return JSONResponse(content={"status": "Chat history has been refreshed"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def status():
    return JSONResponse(content={"status": "Server is up and running"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)