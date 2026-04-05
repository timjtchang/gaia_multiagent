import os
import gradio as gr
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# --- Your Agent (wrapping your LangGraph supervisor) ---
class BasicAgent:
    def __init__(self):
        # Import and build your supervisor graph ONCE at init
        from main import build_supervisor
        self.graph = build_supervisor()
        print("LangGraph Supervisor Agent initialized.")

    def __call__(self, question: str) -> str:
        """Called once per question. Must return ONLY the answer string."""
        from main import solve
        try:
            answer = solve(question, file_path="", graph=self.graph)
        except Exception as e:
            print(f"Agent error: {e}")
            answer = ""
        
        # Clean the answer — GAIA uses exact match!
        # Strip any "FINAL ANSWER:" prefix, whitespace, etc.
        answer = answer.strip()
        return answer


def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the agent, submits answers, displays results.
    """
    space_id = os.getenv("SPACE_ID")

    if profile:
        username = f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent
    try:
        agent = BasicAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None

    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            return "Fetched questions list is empty.", None
        print(f"Fetched {len(questions_data)} questions.")
    except Exception as e:
        return f"Error fetching questions: {e}", None

    # 3. Run Agent on each question
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            continue

        # Check if the question has an associated file
        file_name = item.get("file_name", "")
        
        try:
            # If there's a file, download it first
            if file_name:
                file_path = f"./data/{file_name}"
                from main import solve
                submitted_answer = solve(question_text, file_path=file_path, graph=agent.graph)
            else:
                submitted_answer = agent(question_text)
            
            submitted_answer = submitted_answer.strip()
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({
                "Task ID": task_id,
                "Question": question_text,
                "Submitted Answer": submitted_answer,
            })
        except Exception as e:
            print(f"Error running agent on task {task_id}: {e}")
            results_log.append({
                "Task ID": task_id,
                "Question": question_text,
                "Submitted Answer": f"AGENT ERROR: {e}",
            })

    if not answers_payload:
        return "Agent did not produce any answers.", pd.DataFrame(results_log)

    # 4. Submit
    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload,
    }
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=120)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        return final_status, pd.DataFrame(results_log)
    except Exception as e:
        return f"Submission Failed: {e}", pd.DataFrame(results_log)


# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# GAIA Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**
        1. Log in to your Hugging Face account using the button below.
        2. Click 'Run Evaluation & Submit All Answers' to run the agent and submit.
        """
    )
    gr.LoginButton()
    run_button = gr.Button("Run Evaluation & Submit All Answers")
    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)
    run_button.click(fn=run_and_submit_all, outputs=[status_output, results_table])

if __name__ == "__main__":
    print("Launching Gradio Interface...")
    demo.launch(debug=True, share=False)