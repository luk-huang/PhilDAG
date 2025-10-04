from google import genai
from google.genai import types
from pathlib import Path
from dotenv import load_dotenv
from schemas.schema import GraphData

load_dotenv()

def analyze(file_path: Path) -> GraphData:
    client = genai.Client()
    return extract_graph(file_path, client)

def extract_graph(file_path: Path, client: genai.Client) -> GraphData:
    print(f"Analyzing {file_path}")
    with open("services/prompts/extract_arguments.txt", "r", encoding="utf-8") as f:
        prompt = f.read()

    uploaded_file = client.files.upload(
        file=file_path,
    )
    print(f"Finished Upload to Google Cloud {file_path}")

    output = client.models.generate_content(
        model = "gemini-2.5-flash", 
        contents=[
            uploaded_file,
            prompt
        ],
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=False,
                thinking_budget=-1
            ),
            response_schema=GraphData,
            response_mime_type='application/json'
        )
    )
    test_analysis: GraphData = output.parsed
    print("Done! Passing to frontend")
    return test_analysis

if __name__ == '__main__':
    client = genai.Client()
    test = "text/plato_republic_514b-518d_allegory-of-the-cave.pdf"
    file_path = Path(test)
    extract_graph(file_path, client)
