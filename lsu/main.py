import datetime
import json
import logging
import os
import time

import numpy as np
import pandas as pd
import requests
from functions import functions, run_function
from openai import OpenAI
from questions import answer_question
from replicate import Client
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

CODE_PROMPT = """
Here are two input:output examples for code generation. Please use these and follow the styling for future requests that you think are pertinent to the request. Make sure All HTML is generated with the JSX flavoring.

// SAMPLE 1
// A Blue Box with 3 yellow cirles inside of it that have a red outline
<div style={{
  backgroundColor: 'blue',
  padding: '20px',
  display: 'flex',
  justifyContent: 'space-around',
  alignItems: 'center',
  width: '300px',
  height: '100px',
}}>
  <div style={{
    backgroundColor: 'yellow',
    borderRadius: '50%',
    width: '50px',
    height: '50px',
    border: '2px solid red'
  }}></div>
  <div style={{
    backgroundColor: 'yellow',
    borderRadius: '50%',
    width: '50px',
    height: '50px',
    border: '2px solid red'
  }}></div>
  <div style={{
    backgroundColor: 'yellow',
    borderRadius: '50%',
    width: '50px',
    height: '50px',
    border: '2px solid red'
  }}></div>
</div>

// SAMPLE 2
// A RED BUTTON THAT SAYS 'CLICK ME'
<button style={{
  backgroundColor: 'red',
  color: 'white',
  padding: '10px 20px',
  border: 'none',
  borderRadius: '50px',
  cursor: 'pointer'
}}>
  Click Me
</button>
"""
tg_bot_token = os.environ["TG_BOT_TOKEN"]
replicate_token = os.environ["REPLICATE_API_TOKEN"]
client = Client(api_token=replicate_token)
model = client.models.get("meta/llama-2-70b-chat")
version = model.versions.get(
    "2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1"
)

# df = pd.read_csv("processed/embeddings.csv", index_col=0)
# df["embeddings"] = df["embeddings"].apply(eval).apply(np.array)

openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant that answers questions.",
    },
    {"role": "system", "content": CODE_PROMPT},
]

# Assistants begin

current_dir = os.path.dirname(os.path.abspath(__file__))
scraped_csv_path = os.path.join(current_dir, "processed", "scraped.csv")

mdn_scrape_file = openai.files.create(
    file=open(scraped_csv_path, "rb"), purpose="assistants"
)
assistant = openai.beta.assistants.create(
    name="Telegram Bot",
    instructions=CODE_PROMPT,
    tools=[
        {"type": "code_interpreter"},
        {"type": "retrieval"},
        {"type": "function", "function": functions[0]},
        {"type": "function", "function": functions[1]},
    ],
    model="gpt-4-0125-preview",
    file_ids=[
        mdn_scrape_file.id,
    ],
)
# ---
THREAD = openai.beta.threads.create()


def generate_prompt(messages):
    return "\n".join(
        f"[INST] {message['text']} [/INST]" if message["isUser"] else message["text"]
        for message in messages
    )


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


# The meta/llama-2-70b-chat model can stream output as it's running.
# The predict method returns an iterator, and you can iterate over that output.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!"
    )


async def mozilla(update: Update, context: ContextTypes.DEFAULT_TYPE):
    answer = answer_question(df, question=update.message.text)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=answer)


async def transcribe_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Safety Check
    voice_id = update.message.voice.file_id
    if voice_id:
        file = await context.bot.get_file(voice_id)
        await file.download_to_drive(f"voice_note_{voice_id}.ogg")
        await update.message.reply_text("Voice note downloaded, transcribing now")
        audio_file = open(f"voice_note_{voice_id}.ogg", "rb")
        transcript = openai.audio.transcriptions.create(
            model="whisper-1", file=audio_file
        )
        message = openai.beta.threads.messages.create(
            thread_id=THREAD.id, role="user", content=transcript.text
        )
        run = openai.beta.threads.runs.create(
            thread_id=THREAD.id, assistant_id=assistant.id
        )
        await update.message.reply_text(
            f"Transcript finished:\n {transcript.text}\n processing request"
        )
        run = wait_on_run(run, THREAD)
        # if we did a function call, run the function and update the thread's state
        if run.status == "requires_action":
            print(run.required_action.submit_tool_outputs.tool_calls)
            tool_call = run.required_action.submit_tool_outputs.tool_calls[0]
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            response = run_function(name, args)
            if name in ("svg_to_png_bytes"):
                await context.bot.send_photo(
                    chat_id=update.effective_chat.id, photo=response
                )
            if name in ("generate_image"):
                await context.bot.send_photo(
                    chat_id=update.effective_chat.id, photo=response.content
                )
                run = openai.beta.threads.runs.cancel(
                    thread_id=THREAD.id, run_id=run.id
                )
                run = wait_on_run(run, THREAD)
                return
            run = openai.beta.threads.runs.submit_tool_outputs(
                thread_id=THREAD.id,
                run_id=run.id,
                tool_outputs=[
                    {"tool_call_id": tool_call.id, "output": json.dumps(str(response))}
                ],
            )
            run = wait_on_run(run, THREAD)
        # Retrieve the message object
        messages = openai.beta.threads.messages.list(
            thread_id=THREAD.id, order="asc", after=message.id
        )
        # Extract the message content
        message_content = messages.data[0].content[0].text
        await context.bot.send_message(
            chat_id=update.effective_chat.id, text=message_content.value
        )


# Asynchronous function to handle chat interactions
async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Append the user's message to the messages list
    messages.append({"role": "user", "content": update.message.text})

    # Generate an initial response using GPT-3.5 model
    initial_response = openai.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, functions=functions
    )
    initial_response_message = initial_response.choices[0].message.content
    final_response = None

    # Check if the initial response contains a function call
    if initial_response_message and initial_response_message.get("function_call"):
        # Extract the function name and arguments
        name = initial_response_message.function_call.name
        args = json.loads(initial_response_message.function_call.arguments)

        # Run the corresponding function
        function_response = run_function(name, args)

        # if 'svg_to_png_bytes' function, send a photo and return as there's nothing else to do
        if name == "svg_to_png_bytes":
            await context.bot.send_photo(
                chat_id=update.effective_chat.id, photo=function_response
            )
            return

        # Generate the final response
        final_response = openai.chat.completions.create(
            model="gpt-3.5-turbo-0613",
            messages=[
                *messages,
                initial_response_message,
                {
                    "role": "function",
                    "name": initial_response_message["function_call"]["name"],
                    "content": json.dumps(function_response),
                },
            ],
        )
        final_answer = final_response.choices[0].message.content

        # Send the final response if it exists
        if final_answer:
            messages.append({"role": "assistant", "content": final_answer})
            await context.bot.send_message(
                chat_id=update.effective_chat.id, text=final_answer
            )
        else:
            # Send an error message if something went wrong
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="something wrong happened, please try again",
            )
    else:
        # If no function call, send the initial response
        messages.append(initial_response_message)
        await context.bot.send_message(
            chat_id=update.effective_chat.id, text=initial_response_message["content"]
        )
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!"
    )


# async def chat2(update: Update, context: ContextTypes.DEFAULT_TYPE):
#   # Add User Message
#   message_history.append({"isUser": True, "text": update.message.text})
#   prompt = generate_prompt(message_history)
#   prediction = client.predictions.create(version=version,
#                                          input={"prompt": prompt})
#   await context.bot.send_message(chat_id=update.effective_chat.id,
#                                  text=prediction.status)
#   prediction.wait()
#   await context.bot.send_message(chat_id=update.effective_chat.id,
#                                  text=prediction.status)
#   output = prediction.output
#   human_readable_output = ''.join(output).strip()
#   await context.bot.send_message(chat_id=update.effective_chat.id,
#                                  text=human_readable_output)
#   #Add AI Message
#   message_history.append({"isUser": False, "text": human_readable_output})


def wait_on_run(run, thread):
    while run.status == "queued" or run.status == "in_progress":
        print(run.status)
        run = openai.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run


def pretty_print(messages):
    print("# Messages")
    for m in messages:
        print(f"{m.role}: {m.content[0].text.value}")
    print()


async def assistant_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = openai.beta.threads.messages.create(
        thread_id=THREAD.id, role="user", content=update.message.text
    )
    run = openai.beta.threads.runs.create(
        thread_id=THREAD.id, assistant_id=assistant.id
    )
    run = wait_on_run(run, THREAD)
    # if we did a function call, run the function and update the thread's state
    if run.status == "requires_action":
        print(run.required_action.submit_tool_outputs.tool_calls)
        tool_call = run.required_action.submit_tool_outputs.tool_calls[0]
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        response = run_function(name, args)
        if name in ("svg_to_png_bytes"):
            await context.bot.send_photo(
                chat_id=update.effective_chat.id, photo=response
            )
        if name in ("generate_image"):
            await context.bot.send_photo(
                chat_id=update.effective_chat.id, photo=response.content
            )
            run = openai.beta.threads.runs.cancel(thread_id=THREAD.id, run_id=run.id)
            run = wait_on_run(run, THREAD)
            return
        run = openai.beta.threads.runs.submit_tool_outputs(
            thread_id=THREAD.id,
            run_id=run.id,
            tool_outputs=[
                {"tool_call_id": tool_call.id, "output": json.dumps(str(response))}
            ],
        )
        run = wait_on_run(run, THREAD)
    # Retrieve the message object
    messages = openai.beta.threads.messages.list(
        thread_id=THREAD.id, order="asc", after=message.id
    )
    # Extract the message content
    message_content = messages.data[0].content[0].text
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text=message_content.value
    )


if __name__ == "__main__":
    application = ApplicationBuilder().token(tg_bot_token).build()

    start_handler = CommandHandler("start", start)
    mozilla_handler = CommandHandler("mozilla", mozilla)
    # chat_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), chat)
    chat_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), assistant_chat)
    # chat_handler_2 = MessageHandler(filters.TEXT & (~filters.COMMAND), chat2)
    # This handler will be triggered for voice messages
    voice_handler = MessageHandler(filters.VOICE, transcribe_message)

    # application.add_handler(chat_handler)
    application.add_handler(chat_handler)
    application.add_handler(voice_handler)
    application.add_handler(mozilla_handler)
    application.add_handler(start_handler)

    application.run_polling()
