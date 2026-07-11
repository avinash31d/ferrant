"""Extract invoice fields from retrieved passages of a local PDF.

OpenAI first creates a faithful searchable transcription, which is indexed
locally. The final extraction sees only the retrieved passages.
"""
import asyncio
import base64
import os
from pathlib import Path

from ferrant import Agent, Retriever


def chunks(text: str, size: int = 1_200, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks so facts near a boundary are retained."""
    return [text[start:start + size] for start in range(0, len(text), size - overlap)]


async def main() -> None:
    document_path = Path(os.getenv("DOCUMENT_PATH", "invoice.pdf"))
    encoded_pdf = base64.b64encode(document_path.read_bytes()).decode("ascii")
    agent = Agent.openai(
        "gpt-5-nano",
        os.environ["OPENAI_API_KEY"],
        instructions="Use only facts visible in the supplied document.",
    )
    transcription = await agent.run_multimodal([
        {
            "type": "file",
            "data": f"data:application/pdf;base64,{encoded_pdf}",
            "filename": document_path.name,
            "media_type": "application/pdf",
        },
        {
            "type": "text",
            "text": (
                "Transcribe this invoice faithfully into plain text. Preserve headings, "
                "labels, tables, numbers, and line items. Do not summarize or infer facts."
            ),
        },
    ])
    document_text = transcription["content"]
    if not document_text.strip():
        raise RuntimeError("OpenAI returned no searchable text for the PDF")

    # Keep this index scoped to the current document, so stale chunks cannot
    # affect a later extraction of a PDF with the same filename.
    retriever = Retriever()
    await retriever.upsert([
        {
            "id": f"{document_path.name}#{index}",
            "text": chunk,
            "metadata": {"source": str(document_path), "chunk": index},
        }
        for index, chunk in enumerate(chunks(document_text))
    ])

    extraction_request = (
        "Extract the code for Tool Calling"
    )
    matches = await retriever.retrieve(extraction_request, limit=6)
    context = "\n\n".join(
        f"[Source: {match['document']['metadata']['source']}; "
        f"chunk {match['document']['metadata']['chunk']}]\n"
        f"{match['document']['text']}"
        for match in matches
    )

    schema = {
        "type": "object",
        "properties": {
            "code": {"type": ["string", "null"]}
        },
        "required": ["code"],
        "additionalProperties": False,
    }
    result = await agent.run_structured(
        "Use only the retrieved context below. Return null when a value is not present.\n\n"
        f"Task: {extraction_request}\n\nRetrieved context:\n{context}",
        schema,
    )
    print(result)


asyncio.run(main())
