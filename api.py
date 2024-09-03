from langserve import add_routes
from fastapi import FastAPI, HTTPException, UploadFile, File
import uvicorn 
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
import time 
import requests

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

# Initialize the Google Generative AI model
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

# Define the prompt template
Prompt_Template = """ 
You are an intelligent assistant that classifies input based on department maintenance and staff behavior. Analyze the given text or image and determine the classification accurately, focusing on maintenance quality and staff behavior. Provide a detailed explanation for your classification. Give me a one-word answer.
Query:\n {query}\n
Answer:
"""

# Create a PromptTemplate instance
prompt = PromptTemplate(template=Prompt_Template, input_variables=["query"])

# Define a chain using the LangChain runnables
from langchain_core.runnables.base import RunnableSequence

output_parser=StrOutputParser()

# Combining the prompt and model into a runnable sequence
chain = RunnableSequence(prompt, model,output_parser)



app = FastAPI(
    title="railway server",
    version="1.0",
    description="API server for railway"
)

class QueryModel(BaseModel):
    query: str
    key: str

# Define a route for classification
@app.post("/classify")
async def classify_query(query_model: QueryModel):
    # Validate the request key
    if query_model.key != os.getenv("EXPECTED_REQUEST_KEY"):
        raise HTTPException(status_code=403, detail="Unauthorized request key")

    # Invoke the LangChain model with the query
    result = chain.invoke({"query": query_model.query})
    return {"classification": result}

########################## PHOTO ###################################################################

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import os
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import google.generativeai as genai
import logging

# Define the prompt template
prompt_template_2 = """
You are an image classification model. Analyze the image provided and classify it into one of the following categories: "staff behaviour," "maintenance," "cleanliness," or "none."

    Extract Information:
        Describe the key elements and actions visible in the image.
        Identify any notable objects, interactions, or text within the image.

    Classification Criteria:
        Staff Behaviour: If the image depicts interactions between staff members, customer service scenarios, or staff-related actions.
        Maintenance: If the image shows maintenance activities, repair tasks, or equipment related to upkeep.
        Cleanliness: If the image depicts aspects related to cleanliness, such as tidiness or sanitation.
        None: If the image does not fit into any of the above categories.

    Output:
        Provide only the classification category in one word: "staff behaviour," "maintenance," "cleanliness," or "none."
"""
model_2 = genai.GenerativeModel(model_name="gemini-1.5-flash")

class ImageURLModel(BaseModel):
    url: str

@app.post("/classify-image")
async def classify_image(image_url: ImageURLModel):
    try:
        # Directly use the url from the ImageURLModel instance
        response = requests.get(image_url.url)
        image_bytes=BytesIO(response.content)
        image=Image.open(image_bytes)
        result=model_2.generate_content([prompt_template_2,image])

        classification_result = result.candidates[0].content.parts[0].text.strip()
        return JSONResponse(content={"classification": classification_result})
        
    except Exception as e:
        logging.error(f"Error during classification: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

    
####################################### Audio classification ##################################################
model_3 =genai.GenerativeModel('models/gemini-1.5-flash')

# Prompt template for audio classification
prompt_template_3= """
You are an expert in classifying audio content, capable of understanding both English and Hindi. 
Listen carefully to the following audio file. Based on the content, categorize the audio into one of the 
following four categories: 'Staff Behaviour', 'Maintenance', 'Cleanliness', or 'None'. 
Provide the category in English, regardless of the language spoken in the audio.
"""


# Function to handle audio classification
@app.post("/classify-audio")
async def classify_audio(file: UploadFile = File(...)):
    try:
        # Read the audio file content
        audio_bytes = await file.read()

        # Save the file temporarily for processing
        temp_file_path = f"temp_audio_{uuid.uuid4()}.wav"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(audio_bytes)

        # Upload the audio file to the generative AI model
        audio_file = genai.upload_file(path=temp_file_path)

        template_3=prompt_template_3

        # Generate classification result using the model
        response = model_3.generate_content([template_3, audio_file])

        logging.info(f"Model response: {response}")
        
        # Process response
        result = response.text

        # Clean up the temporary file
        os.remove(temp_file_path)

        return JSONResponse(content={"classification": result})

    except Exception as e:
        logging.error(f"Error during classification: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
########################################## Video Classifcation ######################################
prompt_template_4="""
Analyze the image provided and classify it into one of the following categories based on its content:
    Staff Behavior: If the image shows employees interacting with each other or customers, or displaying specific behaviors related to their roles.
    Maintenance: If the image depicts any maintenance activities or issues, such as repairs, equipment malfunctions, or maintenance tasks.
    Cleanliness: If the image shows aspects related to the cleanliness of the environment, such as tidy areas, cleaning activities, or visible dirt and mess.
    None: If the image does not fit into any of the above categories or is irrelevant to staff behavior, maintenance, or cleanliness.
Provide the most appropriate category for the image in one word.
"""

@app.post("/describe-video")
async def describe_video(file: UploadFile = File(...)):
    try:
        # Read the video file content
        video_bytes = await file.read()

        # Save the file temporarily for processing
        temp_file_path = f"temp_video_{uuid.uuid4()}.mp4"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(video_bytes)

        # Upload the video file to the generative AI model
        video_file = genai.upload_file(path=temp_file_path)

        # Wait for the video processing to complete
        while video_file.state.name == "PROCESSING":
            time.sleep(10)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise ValueError(video_file.state.name)

        # Set the model to Gemini 1.5 Flash
        model_4= genai.GenerativeModel(model_name="models/gemini-1.5-flash")

        # Define the prompt
        prompt_4=prompt_template_4 

        # Make the LLM request
        response = model_4.generate_content([prompt_4, video_file], request_options={"timeout": 300})

        # Process response
        result = response.text

        # Clean up the temporary file
        os.remove(temp_file_path)

        return JSONResponse(content={"description": result})

    except Exception as e:
        logging.error(f"Error during video processing: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

