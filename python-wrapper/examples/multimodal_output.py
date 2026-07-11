import asyncio
import base64
import os
from pathlib import Path
from ferrant import Agent

async def main() -> None:
    agent = Agent.openai("gpt-audio-1.5", os.environ["OPENAI_API_KEY"],
        modalities=["text", "audio"], audio_format="wav", audio_voice="alloy")
    response = await agent.run_multimodal([
        {"type": "text", "text": "Give Rust a friendly welcome."}
    ])
    print("Transcript:", response["content"])
    for part in response["content_parts"]:
        if part["type"] == "audio":
            path = Path(f"multimodal-output.{part['format']}")
            path.write_bytes(base64.b64decode(part["data"]))
            print("Wrote", path)

asyncio.run(main())
