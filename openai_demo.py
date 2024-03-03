import os
import json
from openai import OpenAI

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=os.getenv('openai_api_key'),
)


def get_completion(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


lamp_review = """
        Needed a nice lamp for my bedroom, and this one had \
        additional storage and not too high of a price point. \
        Got it fast. The string to our lamp broke during the \
        transit and the company happily sent over a new one. \
        Came within a few days as well. It was easy to put \
        together. I had a missing part, so I contacted their \
        support and they very quickly got me the missing piece! \
        Lumina seems to me to be a great company that cares \
        about their customers and products!!
    """


def example_1():
    """
    这是使用 OPENAI API 编程的第一个例子；
    演示在prompt中使用分隔符。
    :return:
    """
    text = f"""
    You should express what you want a model to do by \
    providing instrucitions that are as clear and \
    specific as you can possibly make them. \
    This will guide the model towards the desired output, \
    and reduce the chances of receiving irrelevant \
    or incorrect responses. Don't confuse writing a \
    clear prompt with writing a short prompt. \
    In many cases, longer prompts provide more clarity \
    and context for the model, which can lead to \
    more detailed and relevant outputs.
    """

    prompt1 = f"""
    Summarize the text delimited by triple backticks \
    into a single sentence.
    ```{text}```
    """
    return get_completion(prompt1)  # 返回str类型的response


def example_2():
    """
    这个例子演示结构化输出。
    :return:
    """
    prompt2 = f"""
    Generate a list of three made-up book titles along \
    with their authors and genres.
    Provide them in JSON format with the following keys:
    book_id, title, author, genre.
    """
    response2 = get_completion(prompt2)
    return json.loads(response2)  # 返回由dict类型组成的list


def example_3():
    """
    Tactic 4: Few-shot prompting 少量示例提示。
    要求LLM重写指令，并按照指定的格式进行输出。
    本例的输出如下：
    Step 1 - Get some water boiling.
    Step 2 - Grab a cup and put a tea bag in it.
    Step 3 - Pour the hot water over the tea bag.
    Step 4 - Let the tea steep for a bit.
    Step 5 - Take out the tea bag.
    Step 6 - Add sugar or milk to taste.
    Step 7 - Enjoy your delicious cup of tea.

    :return:
    """
    text_3 = f"""
        Making a cup of tea is easy! First, you need to get some \
        water boiling. While that's happening, \
        grab a cup and put a tea bag in it. Once the water is \
        hot enough, just pour it over the tea bag. \
        Let it sit for a bit so the tea can steep. After a \
        few minutes, take out the tea bag. If you \
        like, you can add some sugar or milk to taste. \
        And that's it! You've got yourself a delicious \
        cup of tea to enjoy.
    """

    prompt_3 = f"""
       You will be provided with text delimited by triple quotes.
       If it contains a sequence of instructions, \
       re-write those instructions in the following format:
    
       Step 1 - ...
       Step 2 - ...
       ...
       Step N - ...
    
       If the text does not contain a sequence of instructions, \
       then simply write \"No steps provided.\"
    
       \"\"\"{text_3}\"\"\"
       """
    return get_completion(prompt_3)


def example_4():
    """
    LLM inferring: 情感分析，positive or negative ?
    :return:
    """
    prompt_4 = f"""
        What is the sentiment of the following product review,
        which is delimited with triple backticks?
        
        Give your answer as a single word, either "positive" \
        or "negative".
    
        Review text: ```{lamp_review}```
    """
    return get_completion(prompt_4)


def example_5():
    """
    LLM inferring: 情感分析，信息提取，NLP
    指定JSON格式输出，指定key值，限定部分value的取值。
    :return:
    """
    prompt_5 = f"""
        Identify the following items from the review text:
        - Sentiment (positive or negative)
        - Is the reviewer expressing anger? (true or false)
        - Item purchased by reviewer
        - Company that made the item
    
        The review is delimited with triple backticks. \
        Format your response as a JSON object with \
        "Sentiment", "Anger", "Item" and "Brand" as the keys.
        If the information isn't present, use "unknown" \
        as the value.
        Make your response as short as possible.
        Format the Anger value as a boolean.
    
        Review text: ```{lamp_review}```
    """
    return json.loads(get_completion(prompt_5))
