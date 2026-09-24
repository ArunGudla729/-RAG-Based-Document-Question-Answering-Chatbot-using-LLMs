# Example Questions by Document Type

The system is designed to work across diverse document types. Here are
example questions that showcase grounded retrieval and source attribution.

## Research papers
- "What dataset was used for evaluation, and how large is it?"
- "Summarise the main contribution in two sentences."
- "What limitations do the authors acknowledge?"

## Invoices
- "What is the total amount due and the due date?"
- "Which line item is the most expensive?"
- "What are the payment terms?"

## Policy documents
- "What is the notice period for cancellation?"
- "Are there any exclusions to the coverage?"
- "Who is the policy contact for disputes?"

## Out-of-scope (should return "I don't know")
- "What is the capital of France?"  ← not in the documents
- "What will the stock price be next year?"  ← not answerable from text

These last two demonstrate the anti-hallucination guardrail: when the
retrieved context is insufficient, the assistant declines instead of
fabricating an answer.
