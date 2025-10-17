# Benchmark API Specification

This document describes the API endpoint interface for evaluating AI systems with the SALUS-SafetyQA benchmark.

## Overview

The benchmark harness can evaluate any AI system that implements this API specification. This allows fair comparison between different approaches to construction safety question answering.

## Endpoint Specification

### Request

**Method:** `POST`  
**Endpoint:** Configurable via `benchmark_config.yaml`

#### Request Body

```json
{
  "questions": [
    {
      "id": "question_id", 
      "mc_question": "Question text?",
      "mc_options": [
        {"label": "a", "text": "Option A text", "is_correct": false},
        {"label": "b", "text": "Option B text", "is_correct": true},
        {"label": "c", "text": "Option C text", "is_correct": false},
        {"label": "d", "text": "Option D text", "is_correct": false}
      ],
      "mc_correct_answer": "b"
    }
  ]
}
```

### Response

**Status:** `200 OK`

#### Response Body

```json
{
  "data": {
    "results": [
      {
        "selected_answer": "b",
        "confidence": 0.95,
        "reasoning": "Optional explanation text",
        "retrieved_documents": 15,
        "context_used": true,
        "error": null
      }
    ]
  }
}
```

## Integration with Benchmark Harness

The evaluation script calls your endpoint with batches of questions:

```bash
# Configure your endpoint in benchmark_config.yaml
salus-evaluate --mode api --questions benchmark/SALUS_SafetyQA_v1.0.json
```

The harness will:
1. Load questions from the benchmark file
2. Send batches to your endpoint
3. Collect and validate responses
4. Calculate accuracy metrics
5. Generate statistical reports

## Minimal Example

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

class Result(BaseModel):
    selected_answer: str
    confidence: float
    reasoning: Optional[str] = None
    retrieved_documents: int = 0
    context_used: bool = False
    error: Optional[str] = None

@app.post("/evaluate")
async def evaluate(request: dict):
    results = []
    
    for question in request["questions"]:
        # Your AI system logic here
        answer = process_question(question)
        
        results.append(Result(
            selected_answer=answer,
            confidence=0.95,
            reasoning="..." if request.get("include_reasoning") else None,
            retrieved_documents=10,
            context_used=True
        ))
    
    return {"data": {"results": results}}
```

## Key Requirements

- Results must be returned in the same order as input questions
- Each question must have a corresponding result
- The `selected_answer` must be one of: "a", "b", "c", or "d"
- Confidence should be between 0.0 and 1.0

That's it! The benchmark harness handles all evaluation logic, statistics, and reporting.