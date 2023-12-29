
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import google.generativeai as genai
import google.ai.generativelanguage as glm
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re
from langchain.tools import DuckDuckGoSearchRun
from PIL import Image
from langchain import LLMMathChain
import os
import io


os.environ["GOOGLE_API_KEY"] ="AIzaSyDqKzb2p4ItiEEao-oim5IcGgAifOtv6do"

search = DuckDuckGoSearchRun()
# search = SerpAPIWrapper()
search = Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events,years,dates,time,weather,name,meaning"
    )

llm = ChatGoogleGenerativeAI(model="gemini-pro")
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
math_tool = Tool.from_function(
        func=llm_math_chain.run,
        name="Calculator",
        description="Useful for when you need to answer questions about math. This tool is only for math questions and nothing else. Only input math expressions.",
    )
# Set up the base template
template = """Answer the following questions as best you can. 
iformation about yourself is your name is CODA you are able to answer any question you are developed by Koustav powered by Gemini.
only if you dont know any answer try to use the Search tool to get the information and answer accordingly
you can access external websites via Search tool
you can get previous conversation data from Conversations
note you have access to internet via Search tool
You have access to the following tools:

{tools}

Use the following format:


Conversations:{chat_history}
Question: the input question you must answer
Thought: you should only think about what to do if you need to
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat n times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}"""

memory_key = "chat_history"
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key=memory_key)
# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

prompt = CustomPromptTemplate(
    template=template,
    tools=[search,math_tool],
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "chat_history","intermediate_steps"]
)

class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

def get_res(q):
    output_parser = CustomOutputParser()
    llm = ChatGoogleGenerativeAI(model="gemini-pro")

    memory = st.session_state.memory
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["Observation"],
        allowed_tools=tool_names,
    )
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True,memory=memory)
    z = agent_executor.run(q)
    return z
def image_to_byte_array(image: Image) -> bytes:
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format=image.format)
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

# st.image("./Google-Gemini-AI-Logo.png", width=200)
st.write("")

gemini_pro, gemini_vision = st.tabs(["CODA Web", "CODA Vision"])


def main():
    with gemini_pro:
        st.header("Interact with CODA")
        st.write("")

        prompt = st.text_input("prompt please...", placeholder="Prompt", label_visibility="visible")
        # model = genai.GenerativeModel("gemini-pro")

        if st.button("SEND", use_container_width=True):
            response = get_res(prompt)

            st.write("")
            st.header(":blue[Response]")
            st.write("")

            st.markdown(response)

    with gemini_vision:
        st.header("Interact with CODA Vision")
        st.write("")

        image_prompt = st.text_input("Interact with the Image", placeholder="Prompt", label_visibility="visible")
        uploaded_file = st.file_uploader("Choose and Image", accept_multiple_files=False,
                                         type=["png", "jpg", "jpeg", "img", "webp"])

        if uploaded_file is not None:
            st.image(Image.open(uploaded_file), use_column_width=True)

            st.markdown("""
                <style>
                        img {
                            border-radius: 10px;
                        }
                </style>
                """, unsafe_allow_html=True)

        if st.button("GET RESPONSE", use_container_width=True):
            model = genai.GenerativeModel("gemini-pro-vision")

            if uploaded_file is not None:
                if image_prompt != "":
                    image = Image.open(uploaded_file)
                    z= glm.Content(
                            parts=[
                                glm.Part(text=image_prompt),
                                glm.Part(
                                    inline_data=glm.Blob(
                                        mime_type="image/jpeg",
                                        data=image_to_byte_array(image)
                                    )
                                )
                            ]
                        )
                    # print(z)

                    response = model.generate_content(
                        glm.Content(
                            parts=[
                                glm.Part(text=image_prompt),
                                glm.Part(
                                    inline_data=glm.Blob(
                                        mime_type="image/jpeg",
                                        data=image_to_byte_array(image)
                                    )
                                )
                            ]
                        )
                    )

                    response.resolve()

                    st.write("")
                    st.write(":blue[Response]")
                    st.write("")

                    st.markdown(response.text)

                else:
                    st.write("")
                    st.header(":red[Please Provide a prompt]")

            else:
                st.write("")
                st.header(":red[Please Provide an image]")


if __name__ == "__main__":
    main()
