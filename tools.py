"""
tools.py — LangChain-compatible tools for GAIA Level 1 multi-agent system.
Each tool is decorated with @tool for seamless LangGraph integration.
"""

import os
import math
import subprocess
import tempfile
import logging
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.document_loaders import ArxivLoader

from typing import List, Optional


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Tavily Web Search
# ---------------------------------------------------------------------------
from tavily import TavilyClient


@tool
def tavily_search(query: str, max_results: int = 5, include_raw_content: bool = True) -> dict:
    """
    Search using Tavily API. For GAIA tasks, always set
    include_raw_content=True to get full page text, not just snippets.
    """

    logger.info("Tavily Search...")
    client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
    
    search_results = client.search(
        query=query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        search_depth="advanced",
    )

    urls = [r["url"] for r in search_results.get("results", [])[:3]]
    
    if urls:
        try:
            extracted = client.extract(urls=urls)
            # search_results["extracted_content"] = extracted
        except Exception:
            pass
    
    return extracted



# ---------------------------------------------------------------------------
# 2. ArXiv Search
# ---------------------------------------------------------------------------

@tool
def arxiv_search(query: str) -> str:
    """Search Arxiv for a query and return maximum 3 result.
    Args:
        query: The search query."""
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()

    logger.info("ArXiv Searching...")

    formatted_search_docs = ""

    try:
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
                for doc in search_docs
            ]
        )

    except Exception as e:
        logger.warning(f"Formatting failed, falling back to raw output: {e}")
        formatted_search_docs = str(search_docs)
        
    return {"arxiv_results": formatted_search_docs}



# ---------------------------------------------------------------------------
# 3. Wikipedia Search (Discovery Router)
# ---------------------------------------------------------------------------

def _clean_wiki_content(text: str) -> str:
    """Strip Wikipedia/Jina noise while preserving actual content and tables."""
    import re

    # Remove Jina reader metadata/header (everything before first real heading)
    jina_header = re.search(r'^.*?(?=^#\s)', text, flags=re.DOTALL | re.MULTILINE)
    if jina_header:
        text = text[jina_header.end():]

    # Remove "See also", "References", "External links", etc.
    text = re.sub(
        r'\n#{1,3}\s*(See also|References|External links|Notes|Further reading|'
        r'Bibliography|Sources|Cited sources).*',
        '', text, flags=re.DOTALL | re.IGNORECASE
    )

    # Remove image markdown ![alt](url)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)

    # Remove inline links but keep text: [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # Collapse excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()

@tool
def wiki_search(query: str) -> str:
    """
    Searches Wikipedia and loads the ENTIRE page content as perfectly formatted Markdown.
    Use this to read everything about an entity, including tables, discographies, and history.
    """
    import wikipedia
    import requests
    from wikipedia.exceptions import DisambiguationError, PageError

    logger.info("Searching Wiki...")
    
    try:
        results = wikipedia.search(query, results=1)
        if not results:
            return f"No results found for {query}."
            
        page = wikipedia.page(results[0], auto_suggest=False)
        url = page.url
        
        markdown_url = f"https://r.jina.ai/{url}"
        response = requests.get(markdown_url)
        
        if response.status_code == 200:
            return _clean_wiki_content(response.text)
        else:
            # FIX: Fall back to wikipedia library content if Jina fails
            logger.warning(f"Jina Reader failed ({response.status_code}), falling back to wikipedia library")
            return page.content[:15000]
            
    except DisambiguationError as e:
        return f"Ambiguous query. Try one of these: {', '.join(e.options[:5])}"
    except PageError:
        return f"Page not found for {query}."
    except Exception as e:
        return f"Error: {str(e)}"
    

# ---------------------------------------------------------------------------
# 5. Calculator / Math Evaluator
# ---------------------------------------------------------------------------
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.
    Supports standard math functions: sin, cos, sqrt, log, pi, e, pow, etc.
    Example: 'sqrt(144) + pi * 2'
    """
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed_names.update({"abs": abs, "round": round, "int": int, "float": float})
    result = eval(expression, {"__builtins__": {}}, allowed_names)
    return str(result)


# ---------------------------------------------------------------------------
# Tool 1: Dynamic Code Executor
# ---------------------------------------------------------------------------
@tool
def run_python(code: str, timeout: int = 100) -> str:
    """
    Execute a dynamically generated Python code string in a sandboxed subprocess.
    Use this to write scripts for data processing (e.g., pandas with Excel), 
    complex calculations, or file manipulation.
    
    CRITICAL: You MUST use print() to output your final answer to standard output.
    """

    logger.info("Run generated python code...")

    # 1. Sanitize the input (strip markdown backticks if the LLM includes them)
    code = code.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        if len(lines) >= 2:
            code = "\n".join(lines[1:-1])

    # 2. Write the cleaned code to a temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    # 3. Execute the temp file in an isolated subprocess
    try:
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True, 
            text=True, 
            timeout=timeout,
        )
        
        output = result.stdout
        if result.stderr:
            output += f"\n[STDERR]\n{result.stderr}"
            
        print(f"\n=== DEBUG: LLM GENERATED CODE ===\n{code}")
        print(f"=== DEBUG: SUBPROCESS OUTPUT ===\n{output}\n=================================\n")
        
        return output.strip() or "Script executed successfully, but stdout is empty. You MUST use print() to output the results so I can read them."

    except subprocess.TimeoutExpired:
        return f"Execution timed out after {timeout} seconds. The script might be stuck in a loop or processing too much data."
    except Exception as e:
        return f"Execution failed to start: {str(e)}"
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Tool 2: Existing File Runner
# ---------------------------------------------------------------------------
@tool
def execute_python(filepath: str, timeout: int = 100) -> str:
    """
    Execute an EXISTING Python script file directly from the file system.
    Use this when the user asks for the output of a specific .py file.
    """

    logger.info(f"Run Python Script: {filepath}")
    if not os.path.exists(filepath):
        return f"Error: File not found at {filepath}"
        
    try:
        result = subprocess.run(
            ["python3", filepath],
            capture_output=True, text=True, timeout=timeout,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[STDERR]\n{result.stderr}"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Execution timed out after {timeout}s."
    except Exception as e:
        return f"Execution failed: {str(e)}"


# ---------------------------------------------------------------------------
# Tool 3: Safe File Reader
# ---------------------------------------------------------------------------
@tool
def read_file(filepath: str) -> str:
    """
    Read contents of a text file. 
    DO NOT use this for binary files (.xlsx, .pdf, .mp3, .png, etc.).
    """

    logger.info(f"Reading File: {filepath}")
    if not os.path.exists(filepath):
        return f"Error: File not found at {filepath}"

    # Guardrail against binary files crashing the agent
    if filepath.lower().endswith(('.xlsx', '.xls', '.pdf', '.zip', '.png', '.jpg', '.mp3', '.wav')):
        # FIX: Reference the correct tool name
        return (
            "System Warning: Cannot read binary files directly as text. "
            "Use `run_python` to write a script to analyze this file, "
            "or use `transcribe_audio` for audio, or `analyze_image` for images."
        )
        
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        if len(content) > 10000:
            return content[:10000] + "\n\n... [truncated to save context window. Use run_python if you need to process the whole file]"
        return content
    except Exception as e:
        return f"Failed to read file: {str(e)}"


import base64
import time
from google import genai

# Lazy-initialize the client to avoid crash at import time
load_dotenv()
_gemini_client = None

def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Set the GOOGLE_API_KEY environment variable.")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client

@tool
def analyze_image(filepath: str, prompt: str) -> str:
    """
    Analyze an image file using a vision-capable LLM.
    Use this for .png, .jpg, .jpeg, .gif, .webp files.
    Provide a detailed prompt describing what you need to extract or analyze.
    """

    logger.info(f"Analyzing image: {filepath}")
    if not os.path.exists(filepath):
        return f"Error: File not found at {filepath}"

    with open(filepath, "rb") as f:
        image_data = f.read()

    # Determine mime type
    ext = filepath.lower().rsplit(".", 1)[-1]
    mime_type = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
    }.get(ext, "image/png")

    # FIX: Retry logic for Gemini 400 errors (transient image processing failures)
    for attempt in range(3):
        try:
            response = _get_gemini_client().models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    {
                        "parts": [
                            {"inline_data": {"mime_type": mime_type, "data": base64.standard_b64encode(image_data).decode()}},
                            {"text": prompt},
                        ]
                    }
                ],
            )
            return response.text
        except Exception as e:
            logger.warning(f"Image analysis attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(2)
            else:
                return f"Image analysis failed after 3 attempts: {str(e)}"
    
@tool
def transcribe_audio(file_path: str) -> str:
    """Transcribe an audio file (.mp3, .wav, .ogg, etc.) to text using Gemini."""

    logger.info(f"Transcribing audio: {file_path}")

    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"

    # FIX: Detect MIME type from extension instead of hardcoding audio/mpeg
    ext = file_path.lower().rsplit(".", 1)[-1]
    mime_type = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "ogg": "audio/ogg",
        "flac": "audio/flac",
        "m4a": "audio/mp4",
        "aac": "audio/aac",
    }.get(ext, "audio/mpeg")

    with open(file_path, "rb") as f:
        audio_data = base64.b64encode(f.read()).decode()

    # FIX: Reuse the shared Gemini client
    client = _get_gemini_client()

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": audio_data
                            }
                        },
                        {
                            "text": "Transcribe this audio exactly word for word. Output only the transcription, nothing else."
                        }
                    ]
                }
            ]
        )

        logger.info(response.text)
        return response.text
    except Exception as e:
        logger.error(f"Audio transcription failed: {e}")
        return f"Audio transcription failed: {str(e)}"


# ---------------------------------------------------------------------------
# Tool Groups — used by agents.py to scope tools per sub-agent
# ---------------------------------------------------------------------------
RESEARCH_TOOLS = [tavily_search, wiki_search, transcribe_audio, analyze_image]
MATH_TOOLS = [calculator, run_python]
FILE_TOOLS = [read_file, run_python, execute_python, transcribe_audio, analyze_image]

# FIX: ALL_TOOLS was missing transcribe_audio and analyze_image
# This meant the Generalist agent couldn't handle audio or image files
ALL_TOOLS = [
    tavily_search, wiki_search,
    calculator, run_python, read_file, execute_python,
    transcribe_audio, analyze_image,
]