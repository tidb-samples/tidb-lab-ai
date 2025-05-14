#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import boto3
import streamlit as st
from pytidb import TiDBClient


st.markdown("## 📖 Text2SQL")

db = TiDBClient.connect(
    host=os.getenv("SERVERLESS_CLUSTER_HOST"),
    port=int(os.getenv("SERVERLESS_CLUSTER_PORT")),
    username=os.getenv("SERVERLESS_CLUSTER_USERNAME"),
    password=os.getenv("SERVERLESS_CLUSTER_PASSWORD"),
    database=os.getenv("SERVERLESS_CLUSTER_DATABASE_NAME"),
    enable_ssl=True,
)

client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2"
)

for item in ["generated", "past"]:
    if item not in st.session_state:
        st.session_state[item] = []

table_definitions = []
current_database = db._db_engine.url.database
for table_name in db.table_names():
    table_definitions.append(db.query(f"SHOW CREATE TABLE `{table_name}`").to_rows()[0])

def format_answer(answer: str) -> str:
    # Handle ```.* specifically
    if answer.startswith("```"):
        # Find the first newline after the opening ```
        first_newline = answer.find("\n")
        if first_newline != -1:
            # Remove everything from start to the newline
            answer = answer[first_newline:].strip()

    # Remove closing ``` if present
    if answer.endswith("```"):
        answer = answer[:-len("```")].strip()

    return answer

def on_submit():
    user_input = st.session_state.user_input
    if user_input:
        generated_sql = (
            client.converse(
                modelId="us.amazon.nova-pro-v1:0",
                messages=[
                    {"role": "user", "content": [{"text": f"Question: {user_input}\n Respond in SQL directly, without any other text and format."}]},
                ],
                system=[
                    {"text": f"""You are a very senior database administrator who can write SQL very well,
                        please write MySQL SQL to answer user question,
                        Use backticks to quote table names and column names,
                        here are some table definitions in database,
                        the database name is {current_database}\n\n"""
                        + "\n".join("|".join(t) for t in table_definitions)}
                ],
            )["output"]["message"]["content"][0]["text"]
        )

        generated_sql = format_answer(generated_sql)
        st.session_state.past.append(user_input)

        if "insert" in generated_sql.lower() or "update" in generated_sql.lower():
            st.error(
                "The generated SQL is not a SELECT statement, please check it carefully before running it."
            )
            st.stop()

        # Execute the SQL query and set the result
        answer = None
        try:
            rows = db.query(generated_sql).to_rows()
            sql_result = "\n".join(str(row) for row in rows)

            answer = (
                client.converse(
                    modelId="us.amazon.nova-pro-v1:0",
                    messages=[
                        {"role": "user", "content": [{"text": f"""
                        Question: {user_input}\n\n
                        SQL: {generated_sql}\n\n
                        Result: {sql_result}"""}]},
                    ],
                    system=[
                        {"text": """You are a markdown formatter,
                            format the Question and SQL to markdown text,
                            format the data row into markdown tables.
                            NOTE: Without any other format like ``` etc.
                            """}
                    ],
                )["output"]["message"]["content"][0]["text"]
            )
            answer = format_answer(answer)
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