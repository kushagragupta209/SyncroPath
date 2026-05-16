import os
import json
import io
import pypdf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from agents import GroqAgent, SupervisorAgent
from search_client import DuckDuckGoSearchManager

app = FastAPI(title="AI Preparation Pipeline API")

# Enable CORS so your frontend index.html can call this backend securely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_pdf_text(file_bytes: bytes) -> str:
    """Helper function to parse PDF bytes in memory using pypdf."""
    text = ""
    try:
        pdf_file = io.BytesIO(file_bytes)
        reader = pypdf.PdfReader(pdf_file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process PDF resume: {str(e)}")
    return text.strip()

@app.post("/api/analyze")
async def analyze_job_package(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY environment variable is not set on the server.")

    print(f"Processing Request. Attached File: {resume.filename}")

    # 1. Read and parse the uploaded PDF file bytes asynchronously
    resume_bytes = await resume.read()
    resume_content = extract_pdf_text(resume_bytes)
    
    if not resume_content:
        raise HTTPException(status_code=400, detail="The uploaded resume PDF contains no extractable text.")

    input_context = f"RESUME:\n{resume_content}\n\nJOB DESCRIPTION:\n{job_description}"

    # 2. Instantiate the Agents
    try:
        agent1 = GroqAgent("Agent 1 (Interview Insights)", "prompts/agent1_interview.txt")
        agent2 = GroqAgent("Agent 2 (Tech Stack Extractor)", "prompts/agent2_tech_stack.txt")
        agent3 = GroqAgent("Agent 3 (Resource Curator)", "prompts/agent3_sources.txt")
        agent4 = GroqAgent("Agent 4 (Timetable Planner)", "prompts/agent4_timetable.txt")
        supervisor = SupervisorAgent("prompts/supervisor.txt")
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Configuration Error: {str(e)}")

    # 3. Pipeline Execution
    print("Running Agent 1 (Web Discovery)...")
    search_manager = DuckDuckGoSearchManager()
    search_query = f"site:reddit.com OR site:quora.com interview experience questions job description"
    ddg_search_results = search_manager.execute_search(search_query)
    
    agent1_input = f"{input_context}\n\nLive Web Search Reference Context:\n{ddg_search_results}"
    agent1_output = agent1.run(user_input=agent1_input)

    print("Running Agent 2 (Technical Deep Dive)...")
    agent2_output = agent2.run(user_input=input_context)

    print("Running Agent 3 (Resource Compilation)...")
    agent3_output = agent3.run(user_input=agent2_output)

    print("Running Agent 4 (Timeline Orchestration)...")
    agent4_output = agent4.run(user_input=input_context, context=agent2_output)

    print("Assembling payload with Supervisor...")
    aggregated_payload = (
        f"### AGENT 1 OUTPUT (INTERVIEW EXPERIENCES):\n{agent1_output}\n\n"
        f"### AGENT 2 OUTPUT (TECH STACK NEEDED):\n{agent2_output}\n\n"
        f"### AGENT 3 OUTPUT (BEST LEARNING SOURCES):\n{agent3_output}\n\n"
        f"### AGENT 4 OUTPUT (TIMETABLE PLAN):\n{agent4_output}"
    )

    final_json_string = supervisor.compile_final_json(aggregated_payload)
    
    # 4. Return valid JSON payload directly back to the browser
    try:
        return json.loads(final_json_string)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="The Multi-Agent framework failed to render structural formatting.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)