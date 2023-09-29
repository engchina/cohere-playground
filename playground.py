import json
import os

import gradio as gr
import cohere
from cohere.responses.classify import Example
# from langchain.llms import Cohere
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
from dotenv import load_dotenv, find_dotenv

# read local .env file
_ = load_dotenv(find_dotenv())
COHERE_API_KEY = os.environ["COHERE_API_KEY"]


# co = cohere.Client(api_key=COHERE_API_KEY)


# def chat(question, model):
#     template = """{question}"""
#
#     prompt = PromptTemplate(template=template, input_variables=["question"])
#     llm = Cohere(cohere_api_key=COHERE_API_KEY, model=model)
#     llm_chain = LLMChain(prompt=prompt, llm=llm)
#
#     return llm_chain.run(question)


async def chat_stream(question, model, citation_quality, prompt_truncation, randomness):
    if question is None or len(question) == 0:
        return
    async with cohere.AsyncClient(api_key=COHERE_API_KEY) as aio_co:
        streaming_chat = await aio_co.chat(
            message=question,
            model=model,
            stream=True,
            citation_quality=citation_quality,
            prompt_truncation=prompt_truncation,
            temperature=randomness
        )
        completion = ""
        async for token in streaming_chat:
            # print(f"chat_stream token: {token}")
            # print(f"chat_stream type(token): {type(token)}")
            if isinstance(token, cohere.responses.chat.StreamTextGeneration):
                completion += token.text
                yield gr.update(value=completion)


async def clear_chat_stream():
    yield (gr.update(value=""), gr.update(value=""), gr.update(value="accurate"),
           gr.update(value="auto"), gr.update(value=0.3))


async def generate_stream(question, model, number_of_words, randomness,
                          stop_sentence, top_k, show_likelihood):
    if question is None or len(question) == 0:
        return
    async with cohere.AsyncClient(api_key=COHERE_API_KEY) as aio_co:
        streaming_generate = await aio_co.generate(
            prompt=question,
            max_tokens=3 * number_of_words,
            model=model,
            stream=True,
            temperature=randomness,
            stop_sequences=[] if stop_sentence is None or len(stop_sentence) == 0 else stop_sentence.split(","),
            k=top_k,
            return_likelihoods="GENERATION" if show_likelihood else "NONE"
        )
        completion = ""
        async for token in streaming_generate:
            # print(f"generate_stream token: {token}")
            # print(f"generate_stream type(token): {type(token)}")
            if isinstance(token, cohere.responses.generation.StreamingText):
                completion += token.text
                yield gr.update(value=completion)


async def clear_generate_stream():
    yield (gr.update(value=""), gr.update(value=""), gr.update(value=100), gr.update(value=0.9),
           gr.update(value=""), gr.update(value=0), gr.update(value=False))


async def summarize_stream(question, model, length, format, extractiveness,
                           randomness, additional_command):
    if question is None or len(question) == 0:
        return
    async with cohere.AsyncClient(api_key=COHERE_API_KEY) as aio_co:
        summarize = await aio_co.summarize(
            text=question,
            model=model,
            length=length,
            format=format,
            extractiveness=extractiveness,
            temperature=randomness,
            additional_command=additional_command
        )

        # print(f"summarize_stream summarize: {summarize}")
        # print(f"summarize_stream type(summarize): {type(summarize)}")
        if isinstance(summarize, cohere.responses.summarize.SummarizeResponse):
            completion = summarize.summary
            yield gr.update(value=completion)


async def clear_summarize_stream():
    yield (gr.update(value=""), gr.update(value=""), gr.update(value="medium"), gr.update(value="paragraph"),
           gr.update(value="low"), gr.update(value=0.3), gr.update(value=""))


async def classify_stream(question, model, examples):
    if question is None or len(question) == 0:
        return
    examples = json.loads(examples)
    examples = [Example(example["text"], example["label"]) for example in examples]
    async with cohere.AsyncClient(api_key=COHERE_API_KEY) as aio_co:
        classify = await aio_co.classify(
            inputs=[question],
            model=model,
            examples=examples
        )
        # print(f"classify_stream classify: {classify}")
        # print(f"classify_stream type(classify): {type(classify)}")
        if isinstance(classify, cohere.responses.classify.Classifications):
            completion = classify[0].prediction
            yield gr.update(value=completion)


async def clear_classify_stream():
    yield gr.update(value=""), gr.update(value=""), gr.update(value="")


async def embed_stream(question, model, truncate):
    if question is None or len(question) == 0:
        return
    async with cohere.AsyncClient(api_key=COHERE_API_KEY) as aio_co:
        texts = [question]
        # print(f"texts: {texts}")
        embed = await aio_co.embed(
            texts=texts,
            model=model,
            truncate=truncate
        )

        # print(f"embed: {embed}")
        # print(f"type(embed): {type(embed)}")
        if isinstance(embed, cohere.responses.embeddings.Embeddings):
            completion = embed.embeddings
            yield gr.update(value=completion)


async def clear_embed_stream():
    yield gr.update(value=""), gr.update(value=""), gr.update(value="END")


with gr.Blocks() as demo:
    gr.Markdown(value="# (Unofficial) Cohere Playground")
    with gr.Tabs() as tabs:
        with gr.TabItem(label="Chat"):
            with gr.Row():
                with gr.Column():
                    answer_text = gr.Textbox(label="Answer", lines=10, max_lines=10, show_copy_button=True)
                    question_text = gr.Textbox(label="Question", lines=2)
            with gr.Row():
                with gr.Column():
                    clear_button = gr.Button(value="Clear", label="Clear")
                with gr.Column():
                    chat_button = gr.Button(value="Send", label="Send", variant="primary")
            with gr.Row():
                with gr.Column():
                    model_text = gr.Dropdown(label="Model",
                                             choices=["command", "command-nightly",
                                                      "command-light", "command-light-nightly"],
                                             value="command",
                                             )
            with gr.Row():
                with gr.Column():
                    with gr.Accordion("Advanced Parameters", open=False):
                        with gr.Row():
                            with gr.Column():
                                citation_quality_text = gr.Dropdown(label="CITATION QUALITY",
                                                                    choices=["accurate", "fast"],
                                                                    value="accurate",
                                                                    interactive=True
                                                                    )
                            with gr.Column():
                                prompt_truncation_text = gr.Dropdown(label="PROMPT TRUNCATION",
                                                                     choices=["auto", "off"],
                                                                     value="auto",
                                                                     interactive=True
                                                                     )
                            with gr.Column():
                                randomness_text = gr.Slider(label="RANDOMNESS(Temperature)",
                                                            minimum=0,
                                                            maximum=2,
                                                            step=0.1,
                                                            value=0.3,
                                                            interactive=True
                                                            )
            with gr.Row():
                with gr.Column():
                    gr.Examples(examples=["Can you give me a global market overview of the solar panels?",
                                          "Gather business intelligence on the Chinese markets",
                                          "Summarize recent news about the North American tech job market",
                                          "Give me a rundown of AI startups in the productivity space"],
                                inputs=question_text)
            clear_button.click(clear_chat_stream,
                               inputs=[],
                               outputs=[question_text, answer_text, citation_quality_text, prompt_truncation_text,
                                        randomness_text])
            chat_button.click(chat_stream,
                              inputs=[question_text, model_text, citation_quality_text, prompt_truncation_text,
                                      randomness_text],
                              outputs=answer_text)

        with gr.TabItem(label="Generate"):
            with gr.Row():
                with gr.Column():
                    answer_text = gr.Textbox(label="Answer", lines=10, max_lines=10, show_copy_button=True)
                    question_text = gr.Textbox(label="Question", lines=2)
            with gr.Row():
                with gr.Column():
                    clear_button = gr.Button(value="Clear", label="Clear")
                with gr.Column():
                    generate_button = gr.Button(value="Send", label="Send", variant="primary")
            with gr.Row():
                with gr.Column():
                    model_text = gr.Dropdown(label="Model",
                                             choices=["command", "command-nightly",
                                                      "command-light", "command-light-nightly",
                                                      "base", "base-light"],
                                             value="command"
                                             )
            with gr.Row():
                with gr.Column():
                    number_of_words_text = gr.Slider(label="~NUMBER OF WORDS(1 word is about 3 tokens)",
                                                     minimum=0,
                                                     maximum=1365,
                                                     step=1,
                                                     value=100,
                                                     interactive=True
                                                     )
                with gr.Column():
                    randomness_text = gr.Slider(label="RANDOMNESS(Temperature)",
                                                minimum=0,
                                                maximum=2,
                                                step=0.1,
                                                value=0.9,
                                                interactive=True
                                                )
                with gr.Column():
                    stop_sentence_text = gr.Textbox(label="STOP SEQUENCE", lines=1, interactive=True,
                                                    placeholder="Input sequences seperated by ,")
            with gr.Row():
                with gr.Column():
                    with gr.Accordion("Advanced Parameters", open=False):
                        with gr.Row():
                            with gr.Column():
                                top_k_text = gr.Slider(label="TOP-K",
                                                       minimum=0,
                                                       maximum=500,
                                                       step=1,
                                                       value=0,
                                                       interactive=True
                                                       )
                            with gr.Column():
                                show_likelihood_text = gr.Checkbox(label="Show likelihood",
                                                                   value=False,
                                                                   interactive=True
                                                                   )

            with gr.Row():
                with gr.Column():
                    gr.Examples(examples=["Write a LinkedIn post about starting a career in tech:",
                                          """Suggest three alternative titles with a better marketing copy for the \
following blog.

The tone of the alternative titles is: Inspirational
Blog Title: Learning to Play Chess
Alternative Blog Titles:""",
                                          """Write 5 titles for a blog ideas for the keywords "large language model" \
or "text generation" """,
                                          """write a blog outline for a blog titled "How Transformers made Large \
Language models possible" """],
                                inputs=question_text)
            clear_button.click(clear_generate_stream,
                               inputs=[],
                               outputs=[question_text, answer_text, number_of_words_text, randomness_text,
                                        stop_sentence_text, top_k_text, show_likelihood_text])
            generate_button.click(generate_stream,
                                  inputs=[question_text, model_text, number_of_words_text, randomness_text,
                                          stop_sentence_text, top_k_text, show_likelihood_text],
                                  outputs=answer_text)

        with gr.TabItem(label="Summarize"):
            with gr.Row():
                with gr.Column():
                    answer_text = gr.Textbox(label="Answer", lines=10, max_lines=10, show_copy_button=True)
                    question_text = gr.Textbox(label="Question", lines=2)
            with gr.Row():
                with gr.Column():
                    clear_button = gr.Button(value="Clear", label="Clear")
                with gr.Column():
                    summarize_button = gr.Button(value="Send", label="Send", variant="primary")
            with gr.Row():
                with gr.Column():
                    model_text = gr.Dropdown(label="Model",
                                             choices=["command", "command-nightly",
                                                      "command-light", "command-light-nightly"],
                                             value="command"
                                             )
            with gr.Row():
                with gr.Column():
                    length_text = gr.Dropdown(label="LENGTH",
                                              choices=[("Auto", "auto"), ("Short(1-2 sentences)", "short"),
                                                       ("Medium(3-5 sentences)", "medium"),
                                                       ("Long(4 or more sentences)", "long")],
                                              value="medium",
                                              interactive=True
                                              )
                with gr.Column():
                    format_text = gr.Dropdown(label="FORMAT",
                                              choices=[("Auto", "auto"), ("Paragraph", "paragraph"),
                                                       ("Bullets", "bullets")],
                                              value="paragraph",
                                              interactive=True
                                              )
            with gr.Row():
                with gr.Column():
                    with gr.Accordion("Advanced Parameters", open=False):
                        with gr.Row():
                            with gr.Column():
                                extractiveness_text = gr.Dropdown(label="EXTRACTIVENESS",
                                                                  choices=[("Auto", "auto"), ("Low", "low"),
                                                                           ("Medium", "medium"), ("High", "high")],
                                                                  value="auto",
                                                                  interactive=True
                                                                  )
                            with gr.Column():
                                randomness_text = gr.Slider(label="RANDOMNESS(Temperature)",
                                                            minimum=0,
                                                            maximum=2,
                                                            step=0.1,
                                                            value=0.3,
                                                            interactive=True
                                                            )
                            with gr.Column():
                                additional_command_text = gr.Textbox(label="ADDITIONAL COMMAND", lines=1,
                                                                     placeholder="focusing on action points",
                                                                     info="A free-form instruction for modifying "
                                                                          "how the summaries get generated. "
                                                                          "Should complete the sentence "
                                                                          "\"Generate a summary _\". "
                                                                          "Eg. \"focusing on the next steps\" "
                                                                          "or \"written by Yoda\"",
                                                                     interactive=True)
            with gr.Row():
                with gr.Column():
                    gr.Examples(examples=["""Summarize this dialogue:
Customer: Please connect me with a support agent.
AI: Hi there, how can I assist you today?
Customer: I forgot my password and lost access to the email affiliated to my account. Can you please help me?
AI: Yes of course. First I'll need to confirm your identity and then I can connect you with one of our support agents.
TLDR: A customer lost access to their account.
--
Summarize this dialogue:
AI: Hi there, how can I assist you today?
Customer: I want to book a product demo.
AI: Sounds great. What country are you located in?
Customer: I'll connect you with a support agent who can get something scheduled for you.
TLDR: A customer wants to book a product demo.
--
Summarize this dialogue:
AI: Hi there, how can I assist you today?
Customer: I want to get more information about your pricing.
AI: I can pull this for you, just a moment.
TLDR:""",
                                          """Passage: Is Wordle getting tougher to solve? Players seem to be \
convinced that the game has gotten harder in recent weeks ever since The \ 
New York Times bought it from developer Josh Wardle in late January. The \
Times has come forward and shared that this likely isn’t the case. That \
said, the NYT did mess with the back end code a bit, removing some \
offensive and sexual language, as well as some obscure words There is a \ 
viral thread claiming that a confirmation bias was at play. One Twitter \
user went so far as to claim the game has gone to “the dusty section of the \ 
dictionary” to find its latest words. TLDR: Wordle has not gotten more \
difficult to solve.

--

Passage: ArtificialIvan, a seven-year-old, London-based payment and expense management software company, has raised \
$190 million in Series C funding led by ARG Global, with participation from D9 Capital Group and Boulder Capital. \
Earlier backers also joined the round, including Hilton Group, Roxanne Capital, Paved Roads Ventures, Brook Partners, \ 
and Plato Capital. TLDR: ArtificialIvan has raised $190 million in Series C funding.

--

Passage: The National Weather Service announced Tuesday that a freeze warning is in effect for the Bay Area, 
with freezing temperatures expected in these areas overnight. Temperatures could fall into the mid-20s to low 30s in 
some areas. In anticipation of the hard freeze, the weather service warns people to take action now. TLDR:""",
                                          """It's an exciting day for the development community. Cohere's \
state-of-the-art language AI is now available through Amazon SageMaker. \
This makes it easier for developers to deploy Cohere's pre-trained \
generation language model to Amazon SageMaker, an end-to-end machine \
learning (ML) service. Developers, data scientists, and business analysts \
use Amazon SageMaker to build, train, and deploy ML models quickly and \
easily using its fully managed infrastructure, tools, and workflows. \

At Cohere, the focus is on language. The company's mission is to enable developers and businesses to add language AI \
to their technology stack and build game-changing applications with it. Cohere helps developers and businesses \
automate a wide range of tasks, such as copywriting, named entity recognition, paraphrasing, text summarization, \
and classification. The company builds and continually improves its general-purpose large language models (LLMs), \
making them accessible via a simple-to-use platform. Companies can use the models out of the box or tailor them to \
their particular needs using their own custom data.

Developers using SageMaker will have access to Cohere's Medium generation language model. The Medium generation model \
excels at tasks that require fast responses, such as question answering, copywriting, or paraphrasing. The Medium \
model is deployed in containers that enable low-latency inference on a diverse set of hardware accelerators available \
on AWS, providing different cost and performance advantages for SageMaker customers.

"Amazon SageMaker provides the broadest and most comprehensive set of services that eliminate heavy lifting from each \
step of the machine learning process," said Rajneesh Singh, General Manager AI/ML at Amazon Web Services. "We're \
excited to offer Cohere's general purpose large language model with Amazon SageMaker. Our joint customers can now \
leverage the broad range of Amazon SageMaker services and integrate Cohere's model with their applications for \
accelerated time-to-value and faster innovation." "As Cohere continues to push the boundaries of language AI, \
we are excited to join forces with Amazon SageMaker," said Saurabh Baji, Senior Vice President of Engineering at \
Cohere. "This partnership will allow us to bring our advanced technology and innovative approach to an even wider \
audience, empowering developers and organizations around the world to harness the power of language AI and stay ahead \
of the curve in an increasingly competitive market."

The Cohere Medium generation language model available through SageMaker, provide developers with three key benefits:

Build, iterate, and deploy quickly – Cohere empowers any developer (no NLP, ML, or AI expertise required) to quickly \
get access to a pre-trained, state-of-the-art generation model that understands context and semantics at \
unprecedented levels. This high-quality, large language model reduces the time-to-value for customers by providing an \
out-of-the-box solution for a wide range of language understanding tasks. Private and secure – With SageMaker, \
customers can spin up containers serving Cohere's models without having to worry about their data leaving these \
self-managed containers. Speed and accuracy – Cohere's Medium model offers customers a good balance across quality, \
cost, and latency. Developers can easily integrate the Cohere Generate endpoint into apps using a simple API and SDK. \
Get started with Cohere in SageMaker Developers can use the visual interface of the Amazon SageMaker JumpStart \
foundation models to test Cohere's models without writing a single line of code. You can evaluate the model on your \
specific language understanding task and learn the basics of using generative language models. See Cohere's \
documentation and blog for various tutorials and tips-and-tricks related to language modeling.


Deploy the SageMaker endpoint using a notebook Cohere has packaged Medium models, along with an optimized, \
low-latency inference framework, in containers that can be deployed as SageMaker inference endpoints. Cohere's \
containers can be deployed on a range of different instances (including ml.p3.2xlarge, ml.g5.xlarge, \
and ml.g5.2xlarge) that offer different cost/performance trade-offs. These containers are currently available in two \
Regions: us-east-1 and eu-west-1. Cohere intends to expand its offering in the near future, including adding to the \
number and size of models available, the set of supported tasks (such as the endpoints built on top of these models), \
the supported instances, and the available Regions.

To help developers get started quickly, Cohere has provided Jupyter notebooks that make it easy to deploy these \
containers and run inference on the deployed endpoints. With the preconfigured set of constants in the notebook, \
deploying the endpoint can be easily done with only a couple of lines of code as shown in the following example:


After the endpoint is deployed, users can use Cohere's SDK to run inference. The SDK can be installed easily from \
PyPI as follows:


It can also be installed from the source code in Cohere's public SDK GitHub repository.

After the endpoint is deployed, users can use the Cohere Generate endpoint to accomplish multiple generative tasks, \
such as text summarization, long-form content generation, entity extraction, or copywriting. The Jupyter notebook and \
GitHub repository include examples demonstrating some of these use cases.


Conclusion The availability of Cohere natively on SageMaker via the AWS Marketplace represents a major milestone in \
the field of NLP. The Cohere model's ability to generate high-quality, coherent text makes it a valuable tool for \
anyone working with text data.

If you're interested in using Cohere for your own SageMaker projects, you can now access it on SageMaker JumpStart. \
Additionally, you can reference Cohere's GitHub notebook for instructions on deploying the model and accessing it \
from the Cohere Generate endpoint."""],
                                inputs=question_text)
            clear_button.click(clear_summarize_stream,
                               inputs=[],
                               outputs=[question_text, answer_text, length_text, format_text, extractiveness_text,
                                        randomness_text, additional_command_text])
            summarize_button.click(summarize_stream,
                                   inputs=[question_text, model_text, length_text, format_text, extractiveness_text,
                                           randomness_text, additional_command_text],
                                   outputs=answer_text)

        with gr.TabItem(label="Classify"):
            with gr.Row():
                with gr.Column():
                    answer_text = gr.Textbox(label="Answer", lines=10, max_lines=10, show_copy_button=True)
                    examples_text = gr.Textbox(label="Examples", lines=2)
                    question_text = gr.Textbox(label="Question", lines=2)
            with gr.Row():
                with gr.Column():
                    clear_button = gr.Button(value="Clear", label="Clear")
                with gr.Column():
                    classify_button = gr.Button(value="Send", label="Send", variant="primary")
            with gr.Row():
                with gr.Column():
                    model_text = gr.Dropdown(label="Model",
                                             choices=[
                                                 "embed-multilingual-v2.0", "embed-english-v2.0",
                                                 "embed-english-light-v2.0"
                                             ],
                                             value="embed-english-v2.0"
                                             )
            with gr.Row():
                with gr.Column():
                    gr.Examples(examples=[json.dumps([{"text": "I want to set up a recurring monthly transfer "
                                                               "between my chequing and savings account. How do I "
                                                               "do this?",
                                                       "label": "Savings accounts (chequing & savings)"},
                                                      {
                                                          "text": "I would like to add my wife to my current chequing "
                                                                  "account so"
                                                                  "it's a joint account. What do I need to do?",
                                                          "label": "Savings accounts (chequing & savings)"},
                                                      {
                                                          "text": "Can I set up automated payment for my bills?",
                                                          "label": "Savings accounts (chequing & savings)"},
                                                      {
                                                          "text": "Interest rates are going up - does this impact the "
                                                                  "interest rate in my savings account?",
                                                          "label": "Savings accounts (chequing & savings)"},
                                                      {
                                                          "text": "What is the best option for a student savings "
                                                                  "account?",
                                                          "label": "Savings accounts (chequing & savings)"},
                                                      {
                                                          "text": "My family situation is changing and I need to "
                                                                  "update my risk profile for my equity investments",
                                                          "label": "Investments"},
                                                      {
                                                          "text": "My family situation is changing and I need to "
                                                                  "update my risk profile for my equity investments",
                                                          "label": "Investments"},
                                                      {
                                                          "text": "How can I change my beneficiaries of my investment "
                                                                  "accounts?",
                                                          "label": "Investments"},
                                                      {
                                                          "text": "Is crypto an option for my investment account?",
                                                          "label": "Investments"},
                                                      {
                                                          "text": "How can I minimize my tax exposure?",
                                                          "label": "Taxes"},
                                                      {
                                                          "text": "I'm going to be late filing my ${currentYear - 1} "
                                                                  "tax returns. Is there a penalty?",
                                                          "label": "Taxes"},
                                                      {
                                                          "text": "I'm going to have a baby in November - what "
                                                                  "happens to my taxes?",
                                                          "label": "Taxes"},
                                                      {
                                                          "text": "I'd like to increase my monthly RRSP contributions "
                                                                  "to my RRSP",
                                                          "label": "RRSP"},
                                                      {
                                                          "text": "I want to take advantage of the First Time Home "
                                                                  "Buyers program and take money out of my RRSP. How "
                                                                  "does the program work?",
                                                          "label": "RRSP"},
                                                      {
                                                          "text": "What is the ${currentYear} RRSP limit?",
                                                          "label": "RRSP"}
                                                      ])
                                          ],
                                inputs=examples_text)
                    gr.Examples(examples=["How can I set up a 3rd party contribution to my RRSP?",
                                          "Where can I find my unused RRSP contribution?",
                                          "How do I link my return with my partners?",
                                          "Do I need to complete a return if I moved to Canada this year?",
                                          "Can I set up a business account on your platform?"],
                                inputs=question_text)
            clear_button.click(clear_classify_stream,
                               inputs=[],
                               outputs=[question_text, answer_text, examples_text])
            classify_button.click(classify_stream,
                                  inputs=[question_text, model_text, examples_text],
                                  outputs=answer_text)

        with gr.TabItem(label="Embed"):
            with gr.Row():
                with gr.Column():
                    answer_text = gr.Textbox(label="Answer", lines=10, max_lines=10, show_copy_button=True)
                    question_text = gr.Textbox(label="Question", lines=2)
            with gr.Row():
                with gr.Column():
                    clear_button = gr.Button(value="Clear", label="Clear")
                with gr.Column():
                    embed_button = gr.Button(value="Send", label="Send", variant="primary")
            with gr.Row():
                with gr.Column():
                    model_text = gr.Dropdown(label="Model",
                                             choices=["embed-multilingual-v2.0", "embed-english-v2.0",
                                                      "embed-english-light-v2.0", "search-english-beta-2023-04-02",
                                                      # "rerank-multilingual-v2.0", "rerank-english-v2.0"
                                                      ],
                                             value="embed-english-v2.0"
                                             )
                    truncate_text = gr.Dropdown(label="TRUNCATE",
                                                choices=[("NONE", "NONE"), ("LEFT", "START"), ("RIGHT", "END")],
                                                value="END"
                                                )
            with gr.Row():
                with gr.Column():
                    gr.Examples(examples=["NATO Parliamentary Assembly declares Russia to be a ‘terrorist state’",
                                          "Qatar Bans Beer Sales at World Cup Stadiums",
                                          "Biden calls 'emergency' meeting after missile hits Poland"],
                                inputs=question_text)
            clear_button.click(clear_embed_stream,
                               inputs=[],
                               outputs=[question_text, answer_text, truncate_text])
            embed_button.click(embed_stream,
                               inputs=[question_text, model_text, truncate_text],
                               outputs=answer_text)

demo.queue()
if __name__ == "__main__":
    demo.launch()
