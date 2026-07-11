"""Retrieve local context, then ask an agent to answer from that context."""
import asyncio
import os

from ferrant import Agent, Retriever


async def main() -> None:
    retriever = Retriever(path=".ferrant/rag/vectors.json")
    await retriever.upsert([
        {
            "id": "refund-policy",
            "text": "Refunds are available within 30 days of purchase with an order number.",
            "metadata": {"source": "policies/refunds.md"},
        },
        {
            "id": "support-hours",
            "text": "Support replies Monday through Friday, 09:00 to 17:00 IST.",
            "metadata": {"source": "policies/support.md"},
        },
    ])

    question = "Can I get a refund after buying something last week?"
    matches = await retriever.retrieve(question, limit=3)
    context = "\n\n".join(
        f"Source: {match['document']['metadata']['source']}\n{match['document']['text']}"
        for match in matches
    )

    agent = Agent.openai("gpt-5-nano", os.environ["OPENAI_API_KEY"])
    answer = await agent.run(
        f"Answer only from the supplied context. Cite its source.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    print(answer)


asyncio.run(main())
