import asyncio
from typing import Any, Optional, Dict, List

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr, BaseModel, Field
from task._constants import OPENAI_API_KEY
from task.user_client import UserClient


# TODO: Info about app:
# HOBBIES SEARCHING WIZARD
# Searches users by hobbies and provides their full info in JSON format:
#   Input: `I need people who love to go to mountains`
#   Output:
#     ```json
#       "rock climbing": [{full user info JSON},...],
#       "hiking": [{full user info JSON},...],
#       "camping": [{full user info JSON},...]
#     ```
# ---
# 1. Since we are searching hobbies that persist in `about_me` section - we need to embed only user `id` and `about_me`!
#    It will allow us to reduce context window significantly.
# 2. Pay attention that every 5 minutes in User Service will be added new users and some will be deleted. We will at the
#    'cold start' add all users for current moment to vectorstor and with each user request we will update vectorstor on
#    the retrieval step, we will remove deleted users and add new - it will also resolve the issue with consistency
#    within this 2 services and will reduce costs (we don't need on each user request load vectorstor from scratch and pay for it).
# 3. We ask LLM make NEE (Named Entity Extraction) https://cloud.google.com/discover/what-is-entity-extraction?hl=en
#    and provide response in format:
#    {
#       "{hobby}": [{user_id}, 2, 4, 100...]
#    }
#    It allows us to save significant money on generation, reduce time on generation and eliminate possible
#    hallucinations (corrupted personal info or removed some parts of PII (Personal Identifiable Information)). After
#    generation we also need to make output grounding (fetch full info about user and in the same time check that all
#    presented IDs are correct).
# 4. In response we expect JSON with grouped users by their hobbies.
# ---
# This sample is based on the real solution where one Service provides our Wizard with user request, we fetch all
# required data and then returned back to 1st Service response in JSON format.
# ---
# Useful links:
# Chroma DB: https://docs.langchain.com/oss/python/integrations/vectorstores/index#chroma
# Document#id: https://docs.langchain.com/oss/python/langchain/knowledge-base#1-documents-and-document-loaders
# Chroma DB, async add documents: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.aadd_documents
# Chroma DB, get all records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.get
# Chroma DB, delete records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.delete
# ---
# TASK:
# Implement such application as described on the `flow.png` with adaptive vector based grounding and 'lite' version of
# output grounding (verification that such user exist and fetch full user info)
class HobbyExtraction(BaseModel):
    hobbies: Dict[str, List[str]] = Field(
        description="Dictionary mapping hobby name to list of user_ids"
    )


SYSTEM_PROMPT = """
You are a precise hobby extraction assistant.
You will receive a list of user bios with their IDs.
Your task:
- Identify all **hobbies** or **interest activities** mentioned in those bios.
- Return a JSON object mapping each hobby to the list of user IDs who have that hobby.

Rules:
- Only use hobbies explicitly present in the text.
- Keep hobby names concise and lowercase (e.g., "rock climbing", "yoga", "skiing").
- Do NOT include personal details or summaries.
- If no hobbies found, return empty JSON {}.

Example:
Input:
1. [id=5] "I love hiking and camping in the mountains."
2. [id=9] "Rock climbing and hiking are my weekend activities."
Output:
{
  "hobbies": {
    "hiking": ["5", "9"],
    "camping": ["5"],
    "rock climbing": ["9"]
  }
}
Return only valid JSON.
"""

USER_PROMPT = """
## RAG CONTEXT:
{context}

## USER QUESTION: 
{query}
"""


class HobbiesSearchWizard:
    def __init__(self, embeddings: OpenAIEmbeddings, llm_client: ChatOpenAI):
        self.embeddings = embeddings
        self.llm_client = llm_client
        self.vectorstore: Chroma = self._create_vectorstore()
        self.user_client = UserClient()
        self._initialize_vector_store()

    def _create_vectorstore(self) -> Chroma:
        vector_store = Chroma(
            collection_name="users_hobbies",
            embedding_function=self.embeddings,
        )
        return vector_store

    def _get_all_users_as_documents(self) -> list[Document]:
        users = self.user_client.get_all_users()
        documents: list[Document] = []
        for user in users:
            documents.append(Document(id=user['id'], page_content=user['about_me']))
        return documents

    def _create_batch_store(self, batch: list[Document]) -> list[str]:
        return self.vectorstore.add_documents(batch)

    def _initialize_vector_store(self):
        documents = self._get_all_users_as_documents()
        batch_size = 100
        batches = [
            documents[i:i + batch_size]
            for i in range(0, len(documents), batch_size)
        ]
        tasks = [self._create_batch_store(batch) for batch in batches]

    def format_user_document(self, id: str, about_me: str) -> str:
        return f"""
                User:
                    id: {id}
                    about_me: {about_me}
                    \n
            """

    def retrieve_context(self, query: str, k=10):
        vector_data = self.vectorstore.get()
        existing_ids = set(vector_data.get("ids", []))

        users_from_server = self.user_client.get_all_users()
        server_ids = {str(user["id"]) for user in users_from_server}  # cast to str to match vectorstore

        to_delete = list(existing_ids - server_ids)
        to_create = [user for user in users_from_server if str(user["id"]) not in existing_ids]

        if to_delete:
            print(f"Deleting {len(to_delete)} users from vectorstore...")
            self.vectorstore.delete(to_delete)

        if to_create:
            print(f"Adding {len(to_create)} new users to vectorstore...")
            documents_to_feed = [
                Document(id=str(user["id"]), page_content=user["about_me"])
                for user in to_create
            ]
            self.vectorstore.add_documents(documents_to_feed)

        print(f"Vectorstore synced. Total users now: {len(server_ids)}")

        search_result = self.vectorstore.similarity_search_with_relevance_scores(query=query, k=k)
        users_strs = []
        for res, score in search_result:
            users_strs.append(self.format_user_document(res.id, res.page_content))
            print('\n----------------')
            print(f'Retrieved user: {res.page_content}')
            print(f'Retrieved score: {score}')
            print('\n----------------')
        return '\n\n'.join(users_strs)

    async def get_answer(self, user_query: str) -> str:
        parser = PydanticOutputParser(pydantic_object=HobbyExtraction)
        context = self.retrieve_context(user_query)
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=USER_PROMPT.format(context=context, query=user_query))
        ]).partial(format_instructions=parser.get_format_instructions())
        response = (prompt | self.llm_client | parser).invoke({})
        print('\n-----------------')
        print(f'AI response: {response}')
        print('\n-----------------')
        return await self.out_grounding(response.hobbies)

    async def out_grounding(self, users_hobbies: Dict[str, List[str]]):
        user_hobbies = []
        for hobbie, ids in users_hobbies.items():
            for id in ids:
                try:
                    user = await self.user_client.get_user(int(id))
                    user_hobbies.append(f'{hobbie}: {user}')
                except Exception:
                    print('Fail to fetch user with id {}'.format(id))

        return '\n\n'.join(user_hobbies)


embeddings = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY,
    model='text-embedding-3-small',
    dimensions=384
)
llm_client = ChatOpenAI(
    api_key=OPENAI_API_KEY
)

search_wizard = HobbiesSearchWizard(
    embeddings=embeddings,
    llm_client=llm_client,
)


async def main():
    while (True):
        user_query = input("Please describe hobbies to find matching profiles: ")

        response = await search_wizard.get_answer(user_query)
        print('\n-----------------')
        print(f'Final grounded answer: {response}')
        print('\n-----------------')


asyncio.run(main())
# I need people who love to go to mountains
