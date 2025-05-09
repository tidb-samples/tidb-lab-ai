#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import dotenv
from openai import OpenAI
import streamlit as st
from pytidb import TiDBClient
from pydantic import BaseModel

dotenv.load_dotenv()


class QuestionSQLResponse(BaseModel):
    question: str
    sql: str
    markdown: str


st.markdown("## 📖 Text2SQL")

db = TiDBClient.connect(
    host=os.getenv("SERVERLESS_CLUSTER_HOST"),
    port=int(os.getenv("SERVERLESS_CLUSTER_PORT")),
    username=os.getenv("SERVERLESS_CLUSTER_USERNAME"),
    password=os.getenv("SERVERLESS_CLUSTER_PASSWORD"),
    database=os.getenv("SERVERLESS_CLUSTER_DATABASE_NAME"),
    enable_ssl=True,
)
oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

for item in ["generated", "past"]:
    if item not in st.session_state:
        st.session_state[item] = []

table_definitions = []
current_database = db._db_engine.url.database
for table_name in db.table_names():
    table_definitions.append(db.query(f"SHOW CREATE TABLE `{table_name}`").to_rows()[0])


def on_submit():
    user_input = st.session_state.user_input
    if user_input:
        response = (
            oai.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": f"""
                        You are a very senior database administrator who can write SQL very well,
                        please write MySQL SQL to answer user question,
                        Use backticks to quote table names and column names,
                        here are some table definitions in database,
                        the database name is {current_database}\n\n"""
                        + "\n".join("|".join(t) for t in table_definitions),
                    },
                    {"role": "user", "content": f"Question: {user_input}\n"},
                ],
                response_format=QuestionSQLResponse,
            )
            .choices[0]
            .message.parsed
        )
        st.session_state.past.append(user_input)

        if "insert" in response.sql.lower() or "update" in response.sql.lower():
            st.error(
                "The generated SQL is not a SELECT statement, please check it carefully before running it."
            )
            st.stop()

        # Execute the SQL query and set the result
        answer = None
        try:
            rows = db.query(response.sql).to_rows()
            sql_result = "\n".join(str(row) for row in rows)

            answer = (
                oai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a markdown formatter, format the user input to markdown, format the data row into markdown tables.",
                        },
                        {
                            "role": "user",
                            "content": f"""
                        Question: {response.question}\n\n
                        SQL: {response.sql}\n\n
                        Markdown: {response.markdown}\n\n
                        Result: {sql_result}""",
                        },
                    ],
                )
                .choices[0]
                .message.content
            )
            st.session_state.generated.append(answer)
        except Exception as e:
            st.session_state.generated.append(f"Error: {e}")


st.markdown("##### User Query")
with st.container():
    st.chat_input(
        "Input your question, e.g. how many tables in the database?",
        key="user_input",
        on_submit=on_submit,
    )

    chat_placeholder = st.empty()
    with chat_placeholder.container():
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            with st.chat_message("user"):
                st.write(st.session_state["past"][i])
            with st.chat_message("assistant"):
                st.write(st.session_state["generated"][i])