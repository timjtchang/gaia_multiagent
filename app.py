import os
import re
import shutil
import gradio as gr
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
GAIA_DATASET_DIR = None       # set once by _init_gaia_files()
GAIA_FILE_MAP = {}            # task_id → absolute local path


# ============================================================================
# GAIA file pre-download from HuggingFace dataset repo
# ============================================================================
def _init_gaia_files():
    """
    Download the GAIA dataset repo once and build a task_id → file_path map.
    This avoids relying on the scoring API's /files/ endpoint which 404s.
    """
    global GAIA_DATASET_DIR, GAIA_FILE_MAP

    if GAIA_DATASET_DIR is not None:
        return  # already initialised

    try:
        from huggingface_hub import snapshot_download
        from datasets import load_dataset

        print("Downloading GAIA dataset from HuggingFace Hub...")
        GAIA_DATASET_DIR = snapshot_download(
            repo_id="gaia-benchmark/GAIA",
            repo_type="dataset",
        )
        print(f"GAIA dataset downloaded to: {GAIA_DATASET_DIR}")

        # Load validation split metadata for all levels
        for config in ("2023_level1", "2023_level2", "2023_level3"):
            try:
                ds = load_dataset(
                    GAIA_DATASET_DIR,
                    config,
                    split="validation",
                    trust_remote_code=True,
                )
                for example in ds:
                    task_id = example.get("task_id", "")
                    file_path_rel = example.get("file_path", "")
                    if task_id and file_path_rel:
                        abs_path = os.path.join(GAIA_DATASET_DIR, file_path_rel)
                        if os.path.exists(abs_path):
                            GAIA_FILE_MAP[task_id] = abs_path
                            print(f"  Mapped: {task_id} → {abs_path}")
                        else:
                            print(f"  WARN: file not on disk: {abs_path}")
            except Exception as e:
                print(f"  Could not load config {config}: {e}")

        print(f"GAIA file map built: {len(GAIA_FILE_MAP)} files indexed.")

    except Exception as e:
        print(f"WARNING: Failed to download GAIA dataset: {e}")
        print("Will fall back to scoring API for file downloads.")
        GAIA_DATASET_DIR = ""  # mark as attempted


def _resolve_file(task_id: str, file_name: str, api_url: str) -> str:
    """
    Resolve the local file path for a GAIA task.
    Priority:
      1. HuggingFace dataset repo (pre-downloaded)
      2. Scoring API /files/{task_id} endpoint (fallback)
    Returns the local path, or "" if unavailable.
    """
    # --- Strategy 1: HF dataset repo ---
    hf_path = GAIA_FILE_MAP.get(task_id, "")
    if hf_path and os.path.exists(hf_path):
        # Copy to /tmp with the original file_name so the agent sees
        # a recognisable extension (.mp3, .xlsx, .png, etc.)
        local_path = f"/tmp/{file_name}"
        if not os.path.exists(local_path):
            shutil.copy2(hf_path, local_path)
        file_size = os.path.getsize(local_path)
        print(f"File from HF dataset: {local_path} ({file_size} bytes)")
        return local_path

    # --- Strategy 2: Scoring API fallback ---
    try:
        file_url = f"{api_url}/files/{task_id}"
        print(f"Falling back to scoring API: {file_url}")
        resp = requests.get(file_url, timeout=60)
        resp.raise_for_status()

        local_path = f"/tmp/{file_name}"
        with open(local_path, "wb") as f:
            f.write(resp.content)

        file_size = os.path.getsize(local_path)
        print(f"File from scoring API: {local_path} ({file_size} bytes)")
        if file_size == 0:
            print(f"WARNING: Downloaded file is empty for {task_id}")
            return ""
        return local_path

    except Exception as e:
        print(f"WARNING: File download failed for {task_id}: {e}")
        return ""


# --- Your Agent (wrapping your LangGraph supervisor) ---
class BasicAgent:
    def __init__(self):
        from main import build_supervisor
        self.graph = build_supervisor()
        print("LangGraph Supervisor Agent initialized.")

    def __call__(self, question: str) -> str:
        from main import solve
        try:
            answer = solve(question, file_path="", graph=self.graph)
        except Exception as e:
            print(f"Agent error: {e}")
            answer = ""
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

    # 1. Pre-download GAIA files from HF dataset
    _init_gaia_files()

    # 2. Instantiate Agent
    try:
        agent = BasicAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None

    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 3. Fetch Questions
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

    # 4. Run Agent on each question
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            continue

        file_name = item.get("file_name", "")
        
        try:
            file_path = ""

            # Resolve file from HF dataset or scoring API
            if file_name:
                file_path = _resolve_file(task_id, file_name, api_url)

                if file_path:
                    # Replace embedded file paths in the question text
                    question_text = re.sub(
                        r'\./data/[^\s\)\"\']+',
                        file_path,
                        question_text
                    )
                else:
                    print(f"Could not resolve file for task {task_id} — proceeding without it.")

            # Run the agent
            from main import solve
            submitted_answer = solve(question_text, file_path=file_path, graph=agent.graph)

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

    # 5. Submit
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