import asyncio
from asyncio import gather
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from task._constants import OPENAI_API_KEY
from task.user_client import UserClient

# TODO:
# Before implementation open the `flow_diagram.png` to see the flow of app

BATCH_SYSTEM_PROMPT = """
You are a user search assistant. Your task is to find users from the provided list that match the search criteria.

INSTRUCTIONS:
1. Analyze the user question to understand what attributes/characteristics are being searched for
2. Examine each user in the context and determine if they match the search criteria
3. For matching users, extract and return their complete information
4. Be inclusive - if a user partially matches or could potentially match, include them

OUTPUT FORMAT:
- If you find matching users: Return their full details exactly as provided, maintaining the original format
- If no users match: Respond with exactly "NO_MATCHES_FOUND"
- If uncertain about a match: Include the user with a note about why they might match
"""

FINAL_SYSTEM_PROMPT = """
You are a helpful assistant that provides comprehensive answers based on user search results.

INSTRUCTIONS:
1. Review all the search results from different user batches
2. Combine and deduplicate any matching users found across batches
3. Present the information in a clear, organized manner
4. If multiple users match, group them logically
5. If no users match, explain what was searched for and suggest alternatives
"""

USER_PROMPT = """
## USER DATA:
{context}

## SEARCH QUERY: 
{query}
"""


class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
        self.batch_tokens = []

    def add_tokens(self, tokens: int):
        self.total_tokens += tokens
        self.batch_tokens.append(tokens)

    def get_summary(self):
        return {
            'total_tokens': self.total_tokens,
            'batch_count': len(self.batch_tokens),
            'batch_tokens': self.batch_tokens
        }


# TODO:
# 1. Create ChatOpenAI client
#    hint: api_version set as empty string if you gen an error that indicated that api_version cannot be None
# 2. Create TokenTracker
chat_client = ChatOpenAI(
    api_key=OPENAI_API_KEY
)
token_tracker = TokenTracker()


def parse_user(user: dict[str, Any]) -> str:
    return f"""
        User:
            id: {user['id']}
            name: {user['name']}
            surname: {user['surname']}
            email: {user['email']}
            phone: {user['phone']}
            date_of_birth: {user['date_of_birth']}
            address:
                country: {user['address']['country']}
                city: {user['address']['city']}
                street: {user['address']['street']}
                flat_house: {user['address']['flat_house']}
            gender: {user['gender']}
            company: {user['company']}
            salary: {user['salary']}
            about_me: {user['about_me']}
            credit_card:
                num: {user['credit_card']['num']}
                cvv: {user['credit_card']['cvv']}
                exp_date: {user['credit_card']['exp_date']}
            created_at: {user['created_at']}
            \n
    """


def join_context(context: list[dict[str, Any]]) -> str:
    # TODO:
    # You cannot pass raw JSON with user data to LLM (" sign), collect it in just simple string or markdown.
    # You need to collect it in such way:
    # User:
    #   name: John
    #   surname: Doe
    #   ...

    return ",".join(list(map(parse_user, context)))


async def generate_response(system_prompt: str, user_message: str) -> str:
    print("Processing...")
    # TODO:
    # 1. Create messages array with system prompt and user message
    # 2. Generate response (use `ainvoke`, don't forget to `await` the response)
    # 3. Get usage (hint, usage can be found in response metadata (its dict) and has name 'token_usage', that is also
    #    dict and there you need to get 'total_tokens')
    # 4. Add tokens to `token_tracker`
    # 5. Print response content and `total_tokens`
    # 5. return response content
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]
    response = await chat_client.ainvoke(
        model="gpt-4o-mini",
        input=messages
    )

    usage = response.response_metadata['token_usage']['total_tokens']
    token_tracker.add_tokens(usage)
    print('----------------')
    print(f'AI Response: {response.content}')
    print(f"Tokens used: {token_tracker.get_summary()}")
    print('----------------')
    return response.content


async def main():
    print("Query samples:")
    print(" - Do we have someone with name John that loves traveling?")

    user_question = input("> ").strip()
    if user_question:
        print("\n--- Searching user database ---")

        client = UserClient()
        users = client.get_all_users()
        batch_size = 100
        tasks = []
        for i in range(0, len(users), batch_size):
            context = join_context(users[i:i + batch_size])

            tasks.append(generate_response(
                BATCH_SYSTEM_PROMPT,
                USER_PROMPT.format(context=context, query=user_question)
            ))

        responses = await gather(*tasks)
        filtered_responses = list(filter(lambda response: 'NO_MATCHES_FOUND' not in response, responses))
        combined_result = "\n\n".join(filtered_responses)
        final_response = await generate_response(FINAL_SYSTEM_PROMPT, f"you need to make augmentation of retrieved result and user question: \n {combined_result}")
        print('----------------')
        print(f'Final Response: {final_response}')
        print(f'Tokens used: {token_tracker.get_summary()}')
        print('------------------')


# TODO:
# 1. Get all users (use UserClient)
# 2. Split all users on batches (100 users in 1 batch). We need it since LLMs have its limited context window
# 3. Prepare tasks for async run of response generation for users batches:
#       - create array tasks
#       - iterate through `user_batches` and call `generate_response` with these params:
#           - BATCH_SYSTEM_PROMPT (system prompt)
#           - User prompt, you need to format USER_PROMPT with context from user batch and user question
# 4. Run task asynchronously, use method `gather` form `asyncio`
# 5. Filter results on 'NO_MATCHES_FOUND' (see instructions for BATCH_SYSTEM_PROMPT)
# 5. If results after filtration are present:
#       - combine filtered results with "\n\n" spliterator
#       - generate response with such params:
#           - FINAL_SYSTEM_PROMPT (system prompt)
#           - User prompt: you need to make augmentation of retrieved result and user question
# 6. Otherwise prin the info that `No users found matching`
# 7. In the end print info about usage, you will be impressed of how many tokens you have used. (imagine if we have 10k or 100k users 😅)

if __name__ == "__main__":
    asyncio.run(main())

# The problems with No Grounding approach are:
#   - If we load whole users as context in one request to LLM we will hit context window
#   - Huge token usage == Higher price per request
#   - Added + one chain in flow where original user data can be changed by LLM (before final generation)
# User Question -> Get all users -> ‼️parallel search of possible candidates‼️ -> probably changed original context -> final generation
