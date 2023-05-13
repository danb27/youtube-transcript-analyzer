import tiktoken
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import TokenTextSplitter


class YoutubeTranscriptAnalyzer:
    """
    This is used to interact with ChatGPT/GPT-4 and a YouTube video's transcript.
    """

    def __init__(self, model_name: str = "gpt-4", max_output_length: int = 200, context_window: int = 8000):
        """
        Create the video analyzer with details about the LLM to use

        :param model_name: gpt-3.5-turbo or gpt-4 (as of May 2023)
        :param max_output_length: Max tokens in a response's outputs
        :param context_window: Context window of the model chosen
        """
        self.chat = ChatOpenAI(model_name=model_name, temperature=0)
        self.prompt_template = PromptTemplate(
            template="""
                You are a helpful video transcript analyst assistant. 
                Your job is to analyze a video and respond to a user prompt in your own words given the transcript 
                from that video.

                TRANSCRIPT: {text}

                USER PROMPT: {prompt}

                RESPONSE: """,
            input_variables=["prompt", "text"],
        )
        self.summarizer_chain = load_summarize_chain(
            self.chat,
            chain_type="map_reduce",
            combine_prompt=self.prompt_template,
        )

        prompt_length = len(tiktoken.encoding_for_model(model_name).encode(self.prompt_template.template))

        self.splitter = TokenTextSplitter(
            chunk_size=context_window - prompt_length - max_output_length, chunk_overlap=0, model_name=model_name
        )

    def process(self, url: str, prompt: str) -> str:
        """
        Process a link and a prompt. Summarize if the transcript is too long.

        :param url: URL to the video to be analyzed
        :param prompt: Question/task/prompt to ask of the video
        """
        text = YoutubeLoader.from_youtube_url(url).load()[0].page_content
        docs = self.splitter.split_documents(self.splitter.create_documents([text]))
        if len(docs) > 1:
            return self.summarizer_chain.run(input_documents=docs, prompt=prompt)
        else:
            prompt = self.prompt_template.format_prompt(text=text, prompt=prompt)
            return self.chat(prompt.to_messages()).content

    async def aprocess(self, url: str, prompt: str) -> str:
        """
        Async version of process()
        """
        text = YoutubeLoader.from_youtube_url(url).load()[0].page_content
        docs = self.splitter.split_documents(self.splitter.create_documents([text]))
        if len(docs) > 1:
            return await self.summarizer_chain.arun(input_documents=docs, prompt=prompt)
        else:
            prompt = self.prompt_template.format_prompt(text=text, prompt=prompt)
            return (await self.chat.agenerate([prompt.to_messages()])).generations[0][0].text


if __name__ == "__main__":
    analyzer = YoutubeTranscriptAnalyzer()
    print(
        analyzer.process(
            prompt="What are 5 high level steps for writing your own AutoGPT app?",
            # LangChain Crash Course: Build a AutoGPT app in 25 minutes!
            url="https://www.youtube.com/watch?v=MlK6SIjcjE8",
        )
    )
