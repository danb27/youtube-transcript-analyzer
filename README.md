# youtube-transcript-analyzer
Ask an LLM a question about a youtube video's transcript

# Example usage

```python
analyzer = YoutubeTranscriptAnalyzer()
print(
    analyzer.process(
        prompt="What are 5 high level steps for writing your own AutoGPT app?",
        # LangChain Crash Course: Build a AutoGPT app in 25 minutes!
        url="https://www.youtube.com/watch?v=MlK6SIjcjE8",
    )
)
```

Response:
```text
1. Set up the environment: Import necessary libraries, such as Lang chain, Streamlit, and OpenAI, and install required dependencies like Hugging Face, Cohere, and others.

2. Create prompt templates: Design templates for your prompts, which will help structure the input and output for your large language model. These templates can include variables to make them more dynamic and context-specific.

3. Build LLM chains: Create instances of LLM chains for each prompt template, connecting them to the large language model and specifying any additional parameters, such as temperature or memory.

4. Chain multiple outputs (optional): If you want to generate multiple outputs, use a sequential chain to connect multiple LLM chains together. This allows you to pass the output of one chain as input to the next, creating a more complex and dynamic application.

5. Build the user interface: Use Streamlit or another web framework to create a user interface for your AutoGPT app. This will allow users to input prompts, view generated outputs, and interact with the large language model in a more user-friendly way.
```

There is an async method also available. 
