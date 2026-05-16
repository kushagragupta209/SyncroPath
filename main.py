import os
import json
import asyncio
import pypdf
from agents import GroqAgent, SupervisorAgent
from search_client import DuckDuckGoSearchManager

def read_text_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def read_pdf_file(file_path: str) -> str:
    text = ""
    with open(file_path, 'rb') as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text.strip()

async def main():
    resume_path = "resume.pdf"
    jd_path = "job_description.txt"
    
    if not os.environ.get("GROQ_API_KEY"):
        print("Error: Please set the GROQ_API_KEY environment variable.")
        return

    print("--- Initializing Input Files ---")
    try:
        resume_content = read_pdf_file(resume_path)
        jd_content = read_text_file(jd_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}. Please ensure inputs exist.")
        return

    input_context = f"RESUME:\n{resume_content}\n\nJOB DESCRIPTION:\n{jd_content}"

    print("\n--- Spawning Agents Pipeline ---")
    agent1 = GroqAgent("Agent 1 (Interview Insights)", "prompts/agent1_interview.txt")
    agent2 = GroqAgent("Agent 2 (Tech Stack Extractor)", "prompts/agent2_tech_stack.txt")
    agent3 = GroqAgent("Agent 3 (Resource Curator)", "prompts/agent3_sources.txt")
    agent4 = GroqAgent("Agent 4 (Timetable Planner)", "prompts/agent4_timetable.txt")
    supervisor = SupervisorAgent("prompts/supervisor.txt")

    # Execute Agent 1 (Web Discovery)
    print("[Executing Agent 1] Fetching live Reddit & Quora experiences...")
    search_manager = DuckDuckGoSearchManager()
    search_query = f"site:reddit.com OR site:quora.com interview experience questions job description"
    ddg_search_results = search_manager.execute_search(search_query)
    
    agent1_input = f"{input_context}\n\nLive Web Search Reference Context:\n{ddg_search_results}"
    agent1_output = agent1.run(user_input=agent1_input)
    print("-> Agent 1 complete.")

    # Execute Agent 2 (Technical Deep Dive)
    print("[Executing Agent 2] Analyzing core technical requirements...")
    agent2_output = agent2.run(user_input=input_context)
    print("-> Agent 2 complete.")

    # Execute Agent 3 (Resource Compilation)
    print("[Executing Agent 3] Gathering high-quality tutorials and docs...")
    agent3_output = agent3.run(user_input=agent2_output)
    print("-> Agent 3 complete.")

    # Execute Agent 4 (Timeline Orchestration)
    print("[Executing Agent 4] Crafting optimized daily preparation schedules...")
    agent4_output = agent4.run(user_input=input_context, context=agent2_output)
    print("-> Agent 4 complete.")

    # Supervisor Agent Compilation
    print("\n[Executing Supervisor] Compiling and validating final response...")
    aggregated_payload = (
        f"### AGENT 1 OUTPUT (INTERVIEW EXPERIENCES):\n{agent1_output}\n\n"
        f"### AGENT 2 OUTPUT (TECH STACK NEEDED):\n{agent2_output}\n\n"
        f"### AGENT 3 OUTPUT (BEST LEARNING SOURCES):\n{agent3_output}\n\n"
        f"### AGENT 4 OUTPUT (TIMETABLE PLAN):\n{agent4_output}"
    )

    final_json_string = supervisor.compile_final_json(aggregated_payload)
    
    try:
        json_data = json.loads(final_json_string)
        print("\n================ FINAL SUPERVISOR JSON OUTPUT ================")
        print(json.dumps(json_data, indent=2))
        print("================================================================")
        
        with open("preparation_package.json", "w") as out_file:
            json.dump(json_data, out_file, indent=2)
            print("\nSaved output successfully to preparation_package.json")
            
    except json.JSONDecodeError:
        print("Failed to parse raw output into perfect JSON. Raw String:")
        print(final_json_string)

if __name__ == "__main__":
    asyncio.run(main())