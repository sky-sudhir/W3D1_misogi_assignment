# Prompt Evaluation Report – Medical Q&A

## Evaluation Metrics

- **Accuracy**: Matches expected real-world answer
- **Reasoning Clarity**: Logical explanation (esp. for CoT)
- **Hallucination**: Fabrication or unsafe info
- **Consistency**: Same query → same quality output
- **Relevance**: Stays on topic and addresses the query directly

## Comparative Table

| Prompt Type | Accuracy | Reasoning Clarity | Hallucination | Relevance | Notes                                                              |
| ----------- | -------- | ----------------- | ------------- | --------- | ------------------------------------------------------------------ |
| Zero-shot   | ✓✓✓✗✓✓   | Medium            | Y (see log)   | Medium    | Tends to provide comprehensive but sometimes off-topic information |
| Few-shot    | ✓✓✓✓✓✓   | Medium            | N             | High      | More focused responses, better at staying on topic                 |
| CoT         | ✓✓✓✓✓✓   | High              | N             | High      | Provides structured, step-by-step explanations                     |
| Meta-prompt | ✓✓✗✓✓✓   | Medium            | N             | Medium    | Asks for clarification but sometimes too cautious                 |

## Detailed Analysis

### Zero-shot Prompting

- **Strengths**:
  - Provides comprehensive information about the main query
  - Often includes additional relevant context
- **Weaknesses**:
  - Tends to include information not asked for (e.g., adding dengue info to unrelated queries)
  - Some responses mix multiple topics together
  - Occasional factual inaccuracies (see Hallucination Log)

### Few-shot Prompting

- **Strengths**:
  - More consistent in format
  - Better at staying focused on the specific query
  - Provides clear, concise answers
- **Weaknesses**:
  - Sometimes too brief
  - Limited by the quality of the examples provided

### Chain-of-Thought (CoT) Prompting

- **Strengths**:
  - Excellent reasoning clarity
  - Structured, step-by-step explanations
  - Better at handling complex queries
  - More likely to acknowledge limitations
- **Weaknesses**:
  - Responses can be lengthy
  - May include more detail than necessary

### Meta-prompt

- **Strengths**:
  - Cautious approach by asking clarifying questions
  - Avoids providing potentially incorrect information
  - Helps gather more context before responding
- **Weaknesses**:
  - Sometimes too cautious, even for straightforward queries
  - Doesn't provide immediate answers
  - May frustrate users looking for quick information

## Key Observations

1. **Topic Drift**: Zero-shot prompts frequently include information about dengue even when not relevant to the query.
2. **Consistency**: Few-shot and CoT prompts provide more consistent and reliable responses.
3. **Depth vs Breadth**: CoT excels in depth of explanation, while few-shot provides better breadth of coverage.
4. **Error Handling**: CoT is better at identifying and handling edge cases or ambiguous queries.
5. **Clarification Needed**: Meta-prompt consistently asks for clarification but may be overly cautious for general knowledge questions.

## Recommendations

1. Use few-shot prompting for general queries where concise, accurate information is needed.
2. Employ CoT prompting for complex queries requiring detailed explanations.
3. Be cautious with zero-shot for medical queries due to potential for off-topic information.
4. Use meta-prompt for user interactions where safety is critical and clarification is beneficial.
5. Consider implementing a hybrid approach that combines the strengths of different prompting strategies.
