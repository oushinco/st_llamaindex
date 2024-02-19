
from datetime import datetime, date
import pandas as pd
import os
import streamlit as st
import json
import random

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
# from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from typing import List, Union, Optional
from dateutil import parser
import re

import requests


# from llama_index.llms import LlamaCPP
# from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt


# llm = LlamaCPP(
  
#     # model_url=model_url,

#     model_path=model_path,

    
#     temperature=0.0,
#     max_new_tokens=1024,
    
#     # llama2 has a context window of 4096 tokens
#     context_window=3900,  # note, this sets n_ctx in the model_kwargs below
    
#     # kwargs to pass to __call__()
#     generate_kwargs={},
    
#     # kwargs to pass to __init__()
#     # set to at least 1 to use GPU
#     # model_kwargs={"n_gpu_layers": 1}, 
#     model_kwargs={"n_gpu_layers": -1},
    
#     # transform inputs into Llama2 format
#     messages_to_prompt=messages_to_prompt,
#     completion_to_prompt=completion_to_prompt,
#     verbose=True,
# )

# llm_chatopenai = ChatOpenAI(
#                     model="llama-13b-2", 
#                     openai_api_base="http://oia2wla6:7861/v1",
#                     openai_api_key="EMPTY",
#                     temperature=0,
#                     ) 


st.set_page_config(layout="wide") 

os.environ['no_proxy'] = 'ppldsa02ui.police.nsw.gov.au, 10.1.120.194, 10.17.161.65, oia2wla6'
df_ori = pd.read_csv(r'MB_narr.csv')





    
    
    

# Sample JSON data
data_json = """
{
    "entities": [
        {
            "id": 1,
            "name": "Example Name",
            "type": "Person",
            "titles": [
                "Mr", "SC"
            ],
            "age": 35,
            "nickname": "Example Nickname"
        },
        {
            "id": 2,
            "name": "Example Hotel",
            "type": "Organization"
        },
        {
            "id": 3,
            "name": "Another Name",
            "type": "Person",
            "titles": [
                "Mr"
            ],
            "age": 23
        }
    ],
    "relationships": [
        {
            "source": 1,
            "target": 2,
            "relation_type": "Ownership",
            "relation_date": "07-10-2012"
        },
        {
            "source": 1,
            "target": 3,
            "relation_type": "Contacted"
        }
    ]
}
"""
def auto_trim_tokens(text, max):
    api_base = "http://oia2wla6:7862"
    api_endpoint = f"{api_base}/auto_trim" 
    headers = {
        # "Authorization": "EMPTY",  
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "max_tokens": max
    }
  
    response = requests.post(api_endpoint, headers=headers, json=payload)

    if response.status_code == 200:
        data = response.json()
        return data
  
    else:
        return None


# Base class for entities
class Entity(BaseModel):
    # id: int = Field(default=None, description="Assign a number as ID to the identified entity.")
    name: Optional[str] = Field(default=None, description="The name of the identified entity.")
    type: Optional[str] = Field(default=None, description="The type of the identified entity.")
    

# Subclasses for specific types of entities
class Person(Entity):

    type: str = Field(default="Person", description="The type of the entity.")
    fullname: Optional[str] = Field(default=None, description="The fullname of the person, without title.")
    titles: Optional[List[str]] = Field(default=None, description="A list of titles associated with the person.")
    age: Optional[int] = Field(default=None, description="The age of the person.")
    id_number: Optional[int] = Field(default=None, description="Any id linked to the person, for example, CNI number")
    # nickname: Optional[str] = Field(default=None, description="The person's nickname.")

class Organization(Entity):

    type: str = Field(default="Organization", description="The type of the entity.")   
    pass

class Phone_Number(Entity):

    type: str = Field(default="Phone_Number", description="The type of the entity.")
    name: Optional[str] = Field(default=None, description="The phone number.")

    @validator('name', allow_reuse=True)
    def validate_phone_number(cls, v):
        # Remove spaces from the phone number
        cleaned_number = re.sub(r'\s+', '', v)

        # Regular expression for a simple phone number validation
        phone_regex = re.compile(r'^\+?1?\d{7,15}$')
        if not phone_regex.match(cleaned_number):
            raise ValueError('Invalid phone number format')

        return cleaned_number



class Address(Entity):

    type: str = Field(default="Address", description="The type of the entity.")    
    pass

# Relationship class
class Relationship(BaseModel):
    # source: Optional[int] = Field(default=None, description="The ID of the source entity.")
    # target: Optional[int] = Field(default=None, description="The ID of the target entity.")
    # source_id: Optional[int] = Field(default=None, description="The ID of the source entity.")
    # target_id: Optional[int] = Field(default=None, description="The ID of the target entity.")
    source: Optional[str] = Field(default=None, description="The name of the source entity.")
    target: Optional[str] = Field(default=None, description="The name of the target entity.")
    relation_type: Optional[str] = Field(default=None, description="The type of relationship between the source and target entity.")
    relation_date: Optional[str] = Field(default=None, description="The date of the relationship.")
    additional_info: Optional[dict] = Field(default={}, description="Any additional information about the relationship.")

    @validator('relation_date', pre=True, allow_reuse=True)
    def format_date(cls, value):
        if value is None:
            return value
        try:
            parsed_date = parser.parse(value)
            return parsed_date.strftime("%d-%m-%Y")
        except (ValueError, TypeError):
            raise ValueError("Invalid date format")
    
    
# Main class
class EagleIModel(BaseModel):
    entities: List[Union[Person, Organization, Phone_Number, Address]] = Field(
        default=[], 
        description="A list of entities, which can be a mix of different types such as Person, Organization, Phone Number, and Address."
    )
    relationships: List[Relationship] = Field(
        default=[], 
        description="A list of relationships between the entities."
    )
  
  
  

    
llm = ChatOpenAI(
                    model="llama-13b-2", 
                    openai_api_base="http://oia2wla6:7861/v1",
                    openai_api_key="EMPTY",
                    temperature=0,
                    # max_tokens=4096,
                         
                    ) 




def process_text(text):

    
    EagleI_query = f"Here is the narrative: \n<{text}>."

    EagleI_parser = PydanticOutputParser(pydantic_object=EagleIModel)

    EagleI_prompt = PromptTemplate(
        template="Given the input of a police narrative, extract the entities and relationships according to the format instructions below. \n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": EagleI_parser.get_format_instructions()},
    )

    
    the_prompt_str = EagleI_prompt.invoke(({"query": EagleI_query})).text
    
    ent_chain = EagleI_prompt | llm 

    
    try:
        ent_output = ent_chain.invoke({"query": EagleI_query})
        
        the_resp_string = ent_output.content
      
        ent_output_parsed = EagleI_parser.invoke(ent_output)
        
        json_out = ent_output_parsed.json(indent=4, exclude_unset=True)
  
        

    except Exception as e:
        json_out = data_json
        the_resp_string = "None"
        
        print("An error occurred during parsing:", str(e))
        
    print('get error from model and this is example json')
  

    return json_out, the_prompt_str, the_resp_string


import json

def parse_json(s):
    s = s[next(idx for idx, c in enumerate(s) if c in "{["):]
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        return json.loads(s[:e.pos])
    
    
    
def process_text_llama_index(text):


    EagleI_query = f"Here is the narrative: \n<{text}>."

    EagleI_parser = PydanticOutputParser(pydantic_object=EagleIModel)

    EagleI_prompt = PromptTemplate(
        template="Given the input of a police narrative, extract the entities and relationships according to the format instructions below. \n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": EagleI_parser.get_format_instructions()},
    )

    
    the_prompt_str = EagleI_prompt.invoke(({"query": EagleI_query})).text
    
    

    
    try:
        
        response = llm.complete(the_prompt)

        
        the_resp_string = response.text
      
        json_out = parse_json(the_resp_string)
  
        

    except Exception as e:
        json_out = data_json
        the_resp_string = "None"
        
        print("An error occurred during parsing:", str(e))
        
    print('get error from model and this is example json')
  

    return json_out, the_prompt_str, the_resp_string


def highlight_ent(text):
    """ Function to highlight text with color and background """
    return f"<span style='color: blue; background-color: red;'><b>{text}</b></span>"

def highlight_rel(text):
    """ Function to highlight text with color and background """
    return f"<span style='color: blue; background-color: yellow;'><b>{text}</b></span>"

# def display_entity(entity):
#     entity_type = highlight_ent(entity['type'])
#     entity_name = highlight_ent(entity['name'])
#     st.markdown(f"**{entity_type}:** {entity_name}", unsafe_allow_html=True)
#     if 'titles' in entity:
#         st.write(f"Titles: {', '.join(entity['titles'])}")
#     if 'age' in entity:
#         st.write(f"Age: {entity['age']}")
#     if 'nickname' in entity:
#         st.write(f"Nickname: {entity['nickname']}")

def display_entity(entity):
    st.subheader(f"{entity['type']}: {entity['name']}")
    if 'titles' in entity:
        st.write(f"Titles: {', '.join(entity['titles'])}")
    if 'age' in entity:
        st.write(f"Age: {entity['age']}")
    if 'nickname' in entity:
        st.write(f"Nickname: {entity['nickname']}")

def find_entity_by_id(entities, entity_id):
    """Find an entity by its ID."""
    for entity in entities:
        if entity['id'] == entity_id:
            return entity
    return None 


def display_relationship(relationship, entities):

    source_entity = find_entity_by_id(entities, relationship['source'])
    target_entity = find_entity_by_id(entities, relationship['target'])
    
    if source_entity is not None and target_entity is not None:
        source_name = highlight_rel(source_entity['name'])
        target_name = highlight_rel(target_entity['name'])
        relation_type = highlight_rel(relationship['relation_type'])
        
        if 'relation_date' in relationship and relationship['relation_date']:
            relation_info = f"{source_name} has a {relation_type} relationship with {target_name} since {relationship['relation_date']}."
        else:
            relation_info = f"{source_name} has a {relation_type} relationship with {target_name}."
        
        st.markdown(relation_info, unsafe_allow_html=True)
    else:
        print("One or both of the entities in this relationship were not found.")
        
        
def display_relationship_csv(relationship):
    
    if 'relation_date' in relationship and relationship['relation_date']:
        relation_info = f"{relationship['source']} has a {relationship['relation_type']} relationship with {relationship['target']} since {relationship['relation_date']}."
    else:
        relation_info = f"{relationship['source']} has a {relationship['relation_type']} relationship with {relationship['target']}."

    st.markdown(relation_info, unsafe_allow_html=True)





def display_json_as_buttons(element, key=None):
    if isinstance(element, dict):
        for k, v in element.items():
            display_json_as_buttons(v, k)
    elif isinstance(element, list):
        for item in element:
            display_json_as_buttons(item, key)
    else:
        st.button(f"{key}: {element}" if key else str(element))

def show_json(data_json):
    st.json(data_json)


def show_text(data_json):
    data = json.loads(data_json)
    entities = data["entities"]
    relationships = data["relationships"]
    # display_json_as_buttons(data)

    st.header("Entities")
    for entity in entities:
        display_entity(entity)

    st.header("Relationships")
    for relationship in relationships:
        # display_relationship(relationship, entities)
        display_relationship_csv(relationship)


def show_table(data_json):

    data = json.loads(data_json)

    entities_df = pd.DataFrame(data["entities"])

    entities_df['titles'] = entities_df['titles'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

    relationships_df = pd.DataFrame(data["relationships"])
  
    entity_names = {entity['id']: entity['name'] for entity in data["entities"]}
    relationships_df['source_name'] = relationships_df['source'].map(entity_names)
    relationships_df['target_name'] = relationships_df['target'].map(entity_names)

    st.title("JSON Data Display")

    st.header("Entities")
    st.table(entities_df)

    st.header("Relationships")
    # st.table(relationships_df[['source_name', 'relation_type', 'target_name', 'relation_date']])
    if 'relation_date' in relationships_df.columns:
        st.table(relationships_df[['source_name', 'relation_type', 'target_name', 'relation_date']])
    else:
        st.table(relationships_df[['source_name', 'relation_type', 'target_name']])
    
def get_tables(df):


    # entity_columns = ['id', 'name', 'type', 'titles', 'age', 'nickname', 'narrative_id', 'narrative_text', 'error_message']
    # relationship_columns = ['source_id', 'target_id', 'source', 'target', 'relation_type', 'relation_date', 'narrative_id', 'narrative_text', 'error_message']
    entity_columns = ['name', 'type', 'titles', 'age', 'nickname', 'ProductID', 'InvestigationID', 'narrative_text', 'error_message']
    relationship_columns = ['source', 'target', 'relation_type', 'relation_date', 'ProductID', 'InvestigationID', 'narrative_text', 'error_message']


    # Initialize lists to collect data for each DataFrame
    all_entities = []
    all_relationships = []

    # Loop through each row in your DataFrame
    for index, row in df.iterrows():
        narrative_text = row['Ori_Narr']
        # narrative_id = f"{row['ProductID']}-{row['InvestigationID']}"  # Creating a unique identifier
        ProductID = row['ProductID']
        InvestigationID = row['InvestigationID']

        # Attempt to parse the JSON, handle empty or malformed JSON
        try:
            json_data = json.loads(row['json']) if isinstance(row['json'], str) and not pd.isna(row['json']) else {}
        except (json.JSONDecodeError, TypeError):
            json_data = {}

        # Process entities
        if 'entities' in json_data and isinstance(json_data['entities'], list):
            for entity in json_data['entities']:
                if isinstance(entity, dict):
                    # Use dict comprehension to only include expected columns, add narrative and error columns
                    entity_data = {col: entity.get(col) for col in entity_columns[:-2]}  # Exclude narrative_text and error_message
                    entity_data.update({'ProductID': ProductID, 'InvestigationID': InvestigationID, 'narrative_text': narrative_text, 'error_message': None})
                    # entity_data.update({'narrative_id': narrative_id, 'narrative_text': narrative_text, 'error_message': None})
                    all_entities.append(entity_data)
                else:
                    # Handle unexpected entity format
                    all_entities.append({col: None for col in entity_columns})
                    all_entities[-1].update({'ProductID': ProductID, 'InvestigationID': InvestigationID, 'error_message': f"Unexpected entity format at row {index}"})
        else:
            # Handle missing or non-list 'entities'
            all_entities.append({col: None for col in entity_columns})
            all_entities[-1].update({'ProductID': ProductID, 'InvestigationID': InvestigationID, 'error_message': f"No entities found or invalid format at row {index}"})

        # Process relationships
        if 'relationships' in json_data and isinstance(json_data['relationships'], list):
            for relationship in json_data['relationships']:
                if isinstance(relationship, dict):
                    # Use dict comprehension to only include expected columns, add narrative and error columns
                    relationship_data = {col: relationship.get(col) for col in relationship_columns[:-2]}  # Exclude narrative_text and error_message
                    relationship_data.update({'ProductID': ProductID, 'InvestigationID': InvestigationID, 'narrative_text': narrative_text, 'error_message': None})
                    all_relationships.append(relationship_data)
                else:
                    # Handle unexpected relationship format
                    all_relationships.append({col: None for col in relationship_columns})
                    all_relationships[-1].update({'ProductID': ProductID, 'InvestigationID': InvestigationID, 'error_message': f"Unexpected relationship format at row {index}"})
        else:
            # Handle missing or non-list 'relationships'
            all_relationships.append({col: None for col in relationship_columns})
            all_relationships[-1].update({'ProductID': ProductID, 'InvestigationID': InvestigationID, 'error_message': f"No relationships found or invalid format at row {index}"})

    # Create DataFrames with predefined columns to ensure consistency
    entities_df = pd.DataFrame(all_entities, columns=entity_columns)
    relationships_df = pd.DataFrame(all_relationships, columns=relationship_columns)
    
    return entities_df, relationships_df

def show_table_csv(entities_df, relationships_df):
    
  
    st.title("JSON Data Display")

    st.header("Entities")
    st.table(entities_df)

    st.header("Relationships")

    if 'relation_date' in relationships_df.columns:
        st.table(relationships_df[['source', 'relation_type', 'target', 'relation_date']])
    else:
        st.table(relationships_df[['source', 'relation_type', 'target']])







def main():
    st.title("AI Based Entity Extractor")
    # File uploader widget
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read the CSV file into a Pandas dataframe
        df = pd.read_csv(uploaded_file)
                    
        row_number = st.number_input('Enter row number', min_value=1, max_value=len(df), value=1) - 1  # Adjust for zero indexing
         # Input for specifying the JSON column name, assuming you might not know it beforehand
        json_column = st.text_input('Enter JSON column name', value='json')
             
   
        if st.button("Show json"):
            
            if json_column in df.columns:
                try:
                    json_string = df.at[row_number, json_column]
                    # result_json = json.loads(json_string)
                    result_json = json_string

                    the_prompt = df.at[row_number, 'Ori_Narr']
                    the_response = df.at[row_number, 'Ent_out']
                    st.session_state['result_json'] = result_json
                    st.session_state['the_prompt'] = the_prompt
                    st.session_state['the_response'] = the_response
                    
                    
                except KeyError:
                    st.error(f"Row number {row_number + 1} is out of range.")
                except Exception as e:
                    st.error(f"Error extracting JSON: {str(e)}")
            else:
                st.error("Column name not found. Please check the column name.")
            
             
        
    else:
        user_choice = st.radio(
            "Here is an EagleI narrative:",
            ("I'm Feeling Lucky", "I want to write my own"),
            horizontal=True
        )

        if user_choice == "I'm Feeling Lucky":
            if 'lucky_text' not in st.session_state or st.session_state['user_choice'] != user_choice:
                # Select a new random narrative
                st.session_state['lucky_text'] = df_ori['text'][random.randint(0, len(df_ori) - 1)]
            st.session_state['user_input'] = st.text_area("The Narrative:", height=200, value=st.session_state['lucky_text'])
        else:
            # Provide a text area for user input if not feeling lucky or to preserve user's manual input
            if 'user_choice' not in st.session_state or st.session_state['user_choice'] != user_choice:
                st.session_state['user_input'] = ""
            st.session_state['user_input'] = st.text_area("The Narrative:", height=200, value=st.session_state['user_input'])

        # Update the user choice in session state to detect changes
        st.session_state['user_choice'] = user_choice

        user_input = st.session_state['user_input']
        
        # col1, col2, col3, col4, col5 = st.columns(5)
        # with col1:
        #     show_prompt_checkbox = st.checkbox("Show Prompt", value=True)
        # with col2:
        #     show_response_checkbox = st.checkbox("Show Response", value=True)
        # with col3:
        #     show_json_checkbox = st.checkbox("Show JSON", value=True)
        # with col4:
        #     show_text_checkbox = st.checkbox("Show Text", value=True)
        # with col5:
        #     show_table_checkbox = st.checkbox("Show Table", value=True)
            

        if st.button("Process the above text"):
            if user_input:
                with st.spinner('Processing your text, please wait...'):
                    # result_json, the_prompt, the_response = process_text_llama_index(user_input)
                    result_json = data_json
                    the_prompt = 'testing prompt'
                    the_response = 'testing response'
                    st.session_state['result_json'] = result_json
                    st.session_state['the_prompt'] = the_prompt
                    st.session_state['the_response'] = the_response
            else:
                st.error("Please enter some text to process.")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        show_prompt_checkbox = st.checkbox("Show Prompt", value=True)
    with col2:
        show_response_checkbox = st.checkbox("Show Response", value=True)
    with col3:
        show_json_checkbox = st.checkbox("Show JSON", value=True)
    with col4:
        show_text_checkbox = st.checkbox("Show Text", value=True)
    with col5:
        show_table_checkbox = st.checkbox("Show Table", value=True)
   
    # Display results if available
    if 'result_json' in st.session_state:
        result_json = st.session_state['result_json']
        the_prompt = st.session_state['the_prompt']
        the_response = st.session_state['the_response']

        if show_prompt_checkbox:
            with st.container():
                st.markdown("### Prompt")
                st.markdown("---")
                st.write(the_prompt)
        
        if show_response_checkbox:
            with st.container():
                st.markdown("### Response")
                st.markdown("---")
                st.write(the_response)
        
        if show_json_checkbox:
            with st.container():
                st.markdown("### JSON")
                st.markdown("---")
                st.json(result_json)
        
        if show_text_checkbox:
            with st.container():
                st.markdown("### Text")
                st.markdown("---")
                show_text(result_json)  
        
        if show_table_checkbox:
            with st.container():
                st.markdown("### Table")
                st.markdown("---")
                show_table_csv(result_json)





if __name__ == "__main__":
    main()