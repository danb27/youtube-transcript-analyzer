# youtube-transcript-analyzer
Ask an LLM a question about a youtube video's transcript

# Example usage

```python
analyzer = YoutubeTranscriptAnalyzer()
print(
    analyzer.process(
        prompt="What programming language is used to build an AutoGPT app?",
        # LangChain Crash Course: Build a AutoGPT app in 25 minutes!
        url="https://www.youtube.com/watch?v=MlK6SIjcjE8",
    )
)

print(
    analyzer.process(
        prompt="Give me the TLDW",
        # LangChain Crash Course: Build a AutoGPT app in 25 minutes!
        url="https://www.youtube.com/watch?v=MlK6SIjcjE8",
        strict_qa=False,
    )
)
```

Responses:
```text
The programming language used to build an AutoGPT app is Python.
```

```text
The video demonstrates how to build an application using the Lang chain framework, which simplifies working with large language models like GPT. It covers creating prompt templates, using chains, incorporating memory, and adding tools like the Wikipedia API wrapper. The final application generates a YouTube video title and script based on a given topic.
```

There is an async method also available. 
