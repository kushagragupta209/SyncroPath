import os
from groq import Groq

class GroqAgent:
    def __init__(self, name: str, prompt_path: str, model: str = "llama-3.1-8b-instant"):
        self.name = name
        self.model = model
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.system_prompt = self._load_prompt(prompt_path)

    def _load_prompt(self, path: str) -> str:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def run(self, user_input: str, context: str = "") -> str:
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        if context:
            messages.append({"role": "system", "content": f"Context from previous steps:\n{context}"})
        
        messages.append({"role": "user", "content": user_input})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3
        )
        return response.choices[0].message.content

class SupervisorAgent(GroqAgent):
    def __init__(self, prompt_path: str):
        # Using json-mode optimized settings or models if available
        super().__init__(name="Supervisor", prompt_path=prompt_path, model="llama-3.1-8b-instant")

    def compile_final_json(self, combined_outputs: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Consolidate these agent reports into a clean JSON:\n\n{combined_outputs}"}
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.1
        )
        return response.choices[0].message.content