"""Extract structured invoice fields from a local PDF with OpenAI."""
import asyncio
import base64
import os
from pathlib import Path

from ferrant import Agent


async def main() -> None:
    document_path = Path(os.getenv("DOCUMENT_PATH", "invoice.pdf"))
    encoded_pdf = base64.b64encode(document_path.read_bytes()).decode("ascii")
    agent = Agent.openai(
        "gpt-5-nano",
        os.environ["OPENAI_API_KEY"],
        instructions="Extract only facts visible in the supplied document. Use null for missing fields.",
    )
    response = await agent.run_multimodal([
        {
            "type": "file",
            "data": f"data:application/pdf;base64,{encoded_pdf}",
            "filename": document_path.name,
            "media_type": "application/pdf",
        },
        {
            "type": "text",
            "text": "Extract the invoice number, vendor, invoice date, currency, total, and line items as JSON.",
        },
    ])
    print(response["content"])


asyncio.run(main())
