import asyncio
from typing import Any
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pydantic import SecretStr
from task._constants import OPENAI_API_KEY
from task.user_client import UserClient

# TODO:
# Before implementation open the `vector_based_grounding.png` to see the flow of app

# TODO:
# Provide System prompt. Goal is to explain LLM that in the user message will be provide rag context that is retrieved
# based on user question and user question and LLM need to answer to user based on provided context
SYSTEM_PROMPT = """
You are an intelligent assistant that answers user questions using only the information provided in the retrieved context.

Your task:
- The user message will contain two parts:
  1. "context" — text retrieved from external sources relevant to the user’s question.
  2. "question" — the actual question from the user.

Your goal:
- Read and understand the provided context.
- Use only this context to answer the question.
- If the context does not contain enough information to confidently answer, explicitly state that the information is not available in the provided context.
- Do not invent or assume facts beyond the context.
- Be clear, concise, and accurate in your responses.

Format reminder:
User message example:
<context>
[retrieved information...]
</context>
<question>
[user’s actual question]
</question>
"""

# TODO:
# Should consist retrieved context and user question
USER_PROMPT = """
## RAG CONTEXT:
{context}

## USER QUESTION: 
{query}
"""


def format_user_document(user: dict[str, Any]) -> str:
    # TODO:
    # Prepare context from users JSONs in the same way as in `no_grounding.py` `join_context` method (collect as one string)
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


class UserRAG:
    def __init__(self, embeddings: OpenAIEmbeddings, llm_client: ChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = None
        self.user_client = UserClient()

    async def __aenter__(self):
        print("🔎 Loading all users...")
        # TODO:
        # 1. Get all users (use UserClient)
        # 2. Prepare array of Documents where page_content is `format_user_document(user)` (you need to iterate through users)
        # 3. call `_create_vectorstore_with_batching` (don't forget that its async) and setup it as obj var `vectorstore`
        users = self.user_client.get_all_users()
        documents = list(map(lambda user: Document(page_content=format_user_document(user)), users))
        self.vectorstore = await self._create_vectorstore_with_batching(documents=documents)
        print("✅ Vectorstore is ready.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def _create_vectorstore_with_batching(self, documents: list[Document], batch_size: int = 100):
        # TODO:
        # 1. Split all `documents` on batches (100 documents in 1 batch). We need it since Embedding models have limited context window
        # 2. Iterate through document batches and create array with tasks that will generate FAISS vector stores from documents:
        #    https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.faiss.FAISS.html#langchain_community.vectorstores.faiss.FAISS.afrom_documents
        # 3. Gather tasks with asyncio
        # 4. Create `final_vectorstore` via merge of all vector stores:
        #    https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.faiss.FAISS.html#langchain_community.vectorstores.faiss.FAISS.merge_from
        # 6. Return `final_vectorstore`
        batches = [
            documents[i:i + batch_size]
            for i in range(0, len(documents), batch_size)
        ]

        async def create_batch_store(batch):
            return await FAISS.afrom_documents(batch, self.embeddings)

        tasks = [create_batch_store(batch) for batch in batches]

        vector_stores = await asyncio.gather(*tasks)

        final_vectorstore = vector_stores[0]
        for store in vector_stores[1:]:
            final_vectorstore.merge_from(store)

        return final_vectorstore

    async def retrieve_context(self, query: str, k: int = 10, score: float = 0.1) -> str:
        # TODO:
        # 1. Make similarity search:
        #    https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.faiss.FAISS.html#langchain_community.vectorstores.faiss.FAISS.similarity_search_with_relevance_scores
        # 2. Create `context_parts` empty array (we will collect content here)
        # 3. Iterate through retrieved relevant docs (pay attention that its tuple (doc, relevance_score)) and:
        #       - add doc page content to `context_parts` and then print score and content
        # 4. Return joined context from `context_parts` with `\n\n` spliterator (to enhance readability)
        results: list[tuple[Document, float]] = self.vectorstore.similarity_search_with_relevance_scores(query=query,
                                                                                                         k=k,
                                                                                                         score_threshold=score)
        context_parts = []
        for doc, relevance_score in results:
            print('\n----------------')
            print(f"Document: {doc.page_content}")
            print(f"Relevance Score: {relevance_score}")
            print('\n----------------')
            context_parts.append(doc.page_content)
        return "\n\n".join(context_parts)

    def augment_prompt(self, query: str, context: str) -> str:
        # TODO: Make augmentation for USER_PROMPT via `format` method
        return USER_PROMPT.format(query=query, context=context)

    def generate_answer(self, augmented_prompt: str) -> str:
        # TODO:
        # 1. Create messages array with:
        #       - system prompt
        #       - user prompt
        # 2. Generate response
        #    https://python.langchain.com/docs/integrations/chat/openai/#invocation
        # 3. Return response content
        return self.llm_client.invoke(
            input=[
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=augmented_prompt)
            ]
        ).content


async def main():
    # TODO:
    # 1. Create OpenAIEmbeddings
    #    https://api.python.langchain.com/en/latest/embeddings/langchain_openai.embeddings.base.OpenAIEmbeddings.html
    #    embedding model 'text-embedding-3-small'
    #    I would recommend to set up dimensions as 384
    # 2. Create ChatOpenAI
    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model='text-embedding-3-small',
        dimensions=384
    )
    llm_client = ChatOpenAI(
        api_key=OPENAI_API_KEY
    )

    async with UserRAG(embeddings, llm_client) as rag:
        print("Query samples:")
        print(" - I need user emails that filled with hiking and psychology")
        print(" - Who is John?")
        while True:
            user_question = input("> ").strip()
            if user_question.lower() in ['quit', 'exit']:
                break
            # TODO:
            # 1. Retrieve context
            # 2. Make augmentation
            # 3. Generate answer and print it
            context = await rag.retrieve_context(query=user_question)
            augmented_prompt = rag.augment_prompt(query=user_question, context=context)
            response = rag.generate_answer(augmented_prompt)
            print('\n------------')
            print(f'AI Response: {response}')
            print('\n------------')


asyncio.run(main())

# The problems with Vector based Grounding approach are:
#   - In current solution we fetched all users once, prepared Vector store (Embed takes money) but we didn't play
#     around the point that new users added and deleted every 5 minutes. (Actually, it can be fixed, we can create once
#     Vector store and with new request we will fetch all the users, compare new and deleted with version in Vector
#     store and delete the data about deleted users and add new users).
#   - Limit with top_k (we can set up to 100, but what if the real number of similarity search 100+?)
#   - With some requests works not so perfectly. (Here we can play and add extra chain with LLM that will refactor the
#     user question in a way that will help for Vector search, but it is also not okay in the point that we have
#     changed original user question).
#   - Need to play with balance between top_k and score_threshold
# Benefits are:
#   - Similarity search by context
#   - Any input can be used for search
#   - Costs reduce
