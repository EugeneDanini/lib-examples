import asyncio
from os import getenv
from pathlib import Path
import base64
from uuid import UUID

from dotenv import load_dotenv
from fusionbrain_sdk_python import AsyncFBClient, PipelineType

load_dotenv('.env')

PROMPT = 'A red cucumber sitting on a table'

def _save_images(files: list, uuid: UUID):
    if not files:
        raise FileNotFoundError('No images returned by the pipeline.')

    images_dir = Path(__file__).resolve().parent / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for idx, data in enumerate(files, start=1):
        # Support both plain base64 and data URL format
        if isinstance(data, str):
            b64_str = data.split(',', 1)[-1] if ',' in data else data
        else:
            # Fallback: convert bytes-like to str
            b64_str = data.decode('utf-8') if hasattr(data, 'decode') else str(data)
        img_bytes = base64.b64decode(b64_str)
        file_path = images_dir / f"{uuid}_{idx}.png"
        with open(file_path, 'wb') as f:
            f.write(img_bytes)
        saved_paths.append(str(file_path))
    print(f"Saved {len(saved_paths)} image(s) to: {images_dir}")
    for file in saved_paths:
        print(f" - {file}")


async def run():
    client = AsyncFBClient(x_key=getenv('FB_API_KEY'), x_secret=getenv('FB_API_SECRET'))
    # 1. Get a text-to-image pipeline
    pipelines = await client.get_pipelines_by_type(PipelineType.TEXT2IMAGE)
    text2image_pipeline = pipelines[0]  # Using the first available pipeline
    print(f"Using pipeline: {text2image_pipeline.name}")

    # 2. Run the generation
    run_status = await client.run_pipeline(
        pipeline_id=text2image_pipeline.id,
        prompt=PROMPT
    )

    # 3. Wait for the final result
    print(f"Task started with UUID: {run_status.uuid}")
    result = await client.wait_for_completion(
        request_id=run_status.uuid,
        initial_delay=run_status.status_time
    )

    if result.status == 'DONE':
        try:
            _save_images(getattr(getattr(result, 'result', result), 'files', []), run_status.uuid)
            print("Generation successful!")
        except Exception as e:
            print(f"Error while saving images: {e}")
    else:
        print(f"Generation failed with status: {result.status}")


if __name__ == '__main__':
    asyncio.run(run())
