# Hallucination Log

## Example 1: Off-topic Information in Zero-shot

**Prompt Type:** Zero-shot  
**Query:** "What does hypertension mean?"  
**LLM Response:** Included extensive information about dengue symptoms despite the query being solely about hypertension.  
**Issue:** Off-topic information that could confuse users and dilute the relevant information.

## Example 2: Incomplete Information in Few-shot

**Prompt Type:** Few-shot  
**Query:** "What are the symptoms of dengue?"  
**LLM Response:** Only mentioned high fever, headache, joint pain, and rash.  
**Issue:** Incomplete response - missed other common symptoms like muscle pain, eye pain, and nausea that were mentioned in other prompt types.

## Example 3: Mixed Information in Zero-shot

**Prompt Type:** Zero-shot  
**Query:** "Is it safe to take ibuprofen with paracetamol?"  
**LLM Response:** Initially answered the question but then incorrectly added information about dengue symptoms.  
**Issue:** Mixing unrelated medical information can be misleading and potentially dangerous.

## Example 4: Overly Broad Response in Zero-shot

**Prompt Type:** Zero-shot  
**Query:** "Tell me about high blood sugar."  
**LLM Response:** Included a detailed but unnecessary explanation of dengue symptoms.  
**Issue:** The response included information that was not relevant to the query about high blood sugar.

## Example 5: Inconsistent Symptom Description

**Prompt Type:** Zero-shot  
**Query:** "I have fever and chills, what could be the issue?"  
**LLM Response:** Mentioned dengue symptoms but didn’t clearly differentiate between dengue and other potential causes.  
**Issue:** Could lead to self-diagnosis without proper medical evaluation.

## Example 6: Overly Cautious Meta-prompt

**Prompt Type:** Meta-prompt  
**Query:** "Tell me about high blood sugar."  
**LLM Response:** "Certainly! To give you the most appropriate advice, it would be helpful to know what specific symptoms you are experiencing..."  
**Issue:** The response doesn't provide any information about high blood sugar, even though it's a general knowledge question that could be answered directly.

## Example 7: Redundant Clarification Requests

**Prompt Type:** Meta-prompt  
**Query:** "What are the side effects of paracetamol?"  
**LLM Response:** "Certainly! To better assist you with the right medication, it would be helpful to know what specific symptoms you are experiencing..."  
**Issue:** The response asks for symptoms when the query is a general question about medication side effects, not a request for medical advice.

## Analysis of Hallucination Patterns

1. **Zero-shot Prompting**:

   - Most prone to including off-topic information
   - Frequently adds dengue-related information regardless of query relevance
   - Tends to mix multiple topics in a single response

2. **Few-shot Prompting**:

   - More focused but sometimes too brief
   - Less prone to hallucinations but may omit important details

3. **CoT Prompting**:
   - Most reliable in staying on topic
   - Provides clear structure that helps avoid irrelevant information
   - Better at qualifying information and acknowledging limitations

4. **Meta-prompt**:
   - Overly cautious, even for general knowledge questions
   - Consistently asks for clarification when not always necessary
   - May frustrate users by not providing direct answers to straightforward questions

## Recommendations to Reduce Hallucinations

1. **For Zero-shot**:

   - Implement stricter response filtering
   - Add explicit instructions to only address the specific query
   - Consider breaking down complex queries into sub-questions

2. **For Few-shot**:

   - Include more diverse examples in the prompt
   - Ensure examples cover edge cases and common misconceptions

3. **For CoT**:

   - Encourage the model to explicitly state when it’s speculating
   - Include verification steps in the reasoning process

4. **For Meta-prompt**:
   - Implement logic to differentiate between general knowledge questions and those requiring personal medical advice
   - Provide direct answers to general knowledge questions while still being cautious with personal medical advice
   - Add a confidence threshold for when to ask for clarification

5. **General**:
   - Implement post-processing to detect and filter out off-topic information
   - Add confidence scoring for different parts of the response
   - Consider implementing a verification step for medical information
